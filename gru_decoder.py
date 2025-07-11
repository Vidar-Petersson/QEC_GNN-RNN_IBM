import os
import time
import torch
import wandb

from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.utils.rnn import pad_packed_sequence
from torch_geometric.nn import global_mean_pool

from tqdm import tqdm

from args import Args
from utils import (
    GraphConvLayer, TrainingLogger, group,
    standard_deviation, generate_batches_async,
    save_checkpoint, load_checkpoint
)
from graph_creator import GraphCreator
os.environ["WANDB_SILENT"] = "True"

class GRUDecoder(nn.Module):
    """
    A QEC decoder combining a GNN and an RNN.
    """
    def __init__(self, args: Args):
        super().__init__()
        self.args = args
        
        features = list(zip(args.embedding_features[:-1], args.embedding_features[1:]))
        self.embedding =  nn.ModuleList([GraphConvLayer(a, b) for a, b in features])

        self.rnn = nn.GRU(
            args.embedding_features[-1],
            args.hidden_size, num_layers=args.n_gru_layers,
            batch_first=True
        )

        self.decoder = nn.Sequential(
            nn.Linear(args.hidden_size, 1),
            nn.Sigmoid()
        )

    def _embed_graph(self, x, edge_index, edge_attr, batch_labels):
        for layer in self.embedding:
            x = layer(x, edge_index, edge_attr)
        return global_mean_pool(x, batch_labels)

    def forward(self, x, edge_index, edge_attr, batch_labels, label_map):
        # Run embedding + group
        x = self._embed_graph(x, edge_index, edge_attr, batch_labels)
        x = group(x, label_map)

        # GRU output: out_packed is packed sequence, h is final hidden state
        out_packed, h = self.rnn(x)

        # Unpack the output to get predictions over all chunks
        # out shape: [batch_size, g_actual, hidden_size]
        out, _ = pad_packed_sequence(out_packed, batch_first=True)

        # Apply decoder to get chunkwise predictions (e.g. for time-resolved loss)
        # predictions shape: [batch_size, g_actual]
        predictions = self.decoder(out).squeeze(-1)

        # Get final prediction from the last hidden layer for each sample
        # h shape: [n_layers, batch_size, hidden_size]
        # h[-1] is the final layer's output → shape: [batch_size, hidden_size]
        # final_prediction shape: [batch_size, 1]
        final_prediction = self.decoder(h[-1])

        # Return both time-resolved and final prediction
        return predictions, final_prediction


    def train_model(self, logger: TrainingLogger | None = None, save: str | None = None) -> None:
        local_log = isinstance(logger, TrainingLogger)
        best_model = self.state_dict()

        if self.args.log_wandb:
            wandb.init(project="GNN-RNN-repetition_code", name = save, config = self.args)

        if local_log:
            logger.on_training_begin(self.args)
        
        gc = GraphCreator(self.args)
        gc.train_val_split()
        
        optim = torch.optim.Adam(self.parameters(), lr=self.args.lr)
        schedule = lambda epoch: max(0.95 ** epoch, self.args.min_lr / self.args.lr)
        scheduler = LambdaLR(optim, lr_lambda=schedule)

        # Early stopping setup
        patience = self.args.patience
        best_val_loss = float('inf')
        no_improve = 0

        # ----- NYTT: ladda pretrained eller återuppta träning -----
        start_epoch = 1
        if self.args.pretrained_checkpoint:
            print(f"Loading pretrained weights from {self.args.pretrained_checkpoint}...")
            ckpt_epoch = load_checkpoint(self,
                self.args.pretrained_checkpoint,
                optimizer=optim if self.args.resume else None,
                scheduler=scheduler if self.args.resume else None,
                resume=self.args.resume
            )
            if self.args.resume:
                start_epoch = ckpt_epoch + 1
                print(f"Resuming training from epoch {start_epoch}")
        # -----------------------------------------------------------

        validation_batches = gc.generate_batches(mode="validation")

        # Starta första async-genereringen
        thread, get_next_batches = generate_batches_async(gc, mode="training")
        
        for i in range(1, self.args.n_epochs + 1):
            if local_log:
                logger.on_epoch_begin(i)
        
            epoch_train_loss, epoch_train_acc = 0.0, 0.0
            epoch_val_loss,   epoch_val_acc   = 0.0, 0.0
            epoch_val_log_acc_num = 0
            data_time, model_time = 0, 0
            
            self.train()
            t0 = time.perf_counter()
            thread.join()
            train_batches = get_next_batches()
            # Starta batchgenerering för nästa epok parallellt
            thread, get_next_batches = generate_batches_async(gc, mode="training")
            t1 = time.perf_counter()
            data_time = t1 - t0
        
            for batch in train_batches:
                optim.zero_grad()
    
                x, edge_index, batch_labels, label_map, edge_attr, aligned_flips, lengths, last_label = batch
                # Forward pass through the model
                # out has shape [B, g_actual], where:
                #   B = batch size
                #   g_actual = maximum number of non-empty chunks in batch
                # (can vary between batches, <= t - dt + 2)

                out, final_prediction = self.forward(x, edge_index, edge_attr, batch_labels, label_map)

                if self.args.train_all_times:
                    # Create a boolean mask of shape [B, g_actual] indicating valid chunk positions
                    # For each batch element b, mask[b, i] = True if i < lengths[b]
                    # lengths[b] is the number of non-empty chunks for batch element b
                    mask = torch.arange(out.size(1), device=out.device)[None, :] < lengths[:, None]

                    # Compute binary cross-entropy loss for each element without reduction
                    # loss_raw has shape [B, g_actual], matching the shape of out and aligned_flips
                    loss_raw = nn.functional.binary_cross_entropy(out, aligned_flips, reduction='none')

                    # Apply the mask to zero out the loss from padded (non-existent) chunks
                    # Then compute the mean loss over all valid elements
                    loss = (loss_raw * mask).sum() / mask.sum()
                else:
                    # If not training all times, we only consider the final label
                    loss = nn.functional.binary_cross_entropy(final_prediction, last_label)

                # Backpropagation and optimization step
                loss.backward()
                optim.step()

                # Statistics
                epoch_train_loss += loss.item()
                epoch_train_acc += (torch.sum(torch.round(final_prediction) == last_label) / torch.numel(last_label)).item()

            model_time = time.perf_counter() - t1

            # — Valideringsfas —
            self.eval()
            with torch.no_grad():
                for batch in validation_batches:
                    x, edge_index, batch_labels, label_map, edge_attr, aligned_flips, lengths, last_label = batch
                    out, final_prediction = self.forward(x, edge_index, edge_attr, batch_labels, label_map)
                    if self.args.train_all_times:
                        mask     = torch.arange(out.size(1), device=out.device)[None, :] < lengths[:, None]
                        loss_raw = nn.functional.binary_cross_entropy(out, aligned_flips, reduction='none')
                        loss     = (loss_raw * mask).sum() / mask.sum()
                    else:
                        loss = nn.functional.binary_cross_entropy(final_prediction, last_label)

                    epoch_val_loss += loss.item()
                    epoch_val_acc  += (torch.sum(torch.round(final_prediction) == last_label) / torch.numel(last_label)).item()
                    epoch_val_log_acc_num += torch.sum(torch.round(final_prediction) == last_label).item()
            
            epoch_train_loss /= len(train_batches)
            epoch_train_acc  /= len(train_batches)
            epoch_val_loss   /= len(validation_batches)
            epoch_val_acc    /= len(validation_batches)
            epoch_val_log_acc = (epoch_val_log_acc_num + gc.val_num_trivial) / gc.val_size

            scheduler.step()

            metrics = {
                "train_loss":    epoch_train_loss,
                "train_acc":     epoch_train_acc,
                "val_loss":      epoch_val_loss,
                "val_acc":       epoch_val_acc,
                "val_log_acc":   epoch_val_log_acc,
                "learning_rate": scheduler.get_last_lr()[0],
                "data_time":     data_time,
                "model_time":    model_time
            }

            if self.args.log_wandb:
                wandb.log(metrics)
            if local_log:
                logger.on_epoch_end(logs=metrics)

            # Early stopping & checkpointing
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                no_improve = 0
                if save:
                    ckpt_path = f"./models/{save}.pt"
                    save_checkpoint(self, ckpt_path, optim, scheduler, i)
                    print(f"Saved new best model (log.acc.={epoch_val_log_acc:.5f} loss={epoch_val_loss}) at epoch {i} → {ckpt_path}")
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"Early stopping triggered: no improvement in {patience} epochs.")
                    break
        

        if local_log:
            logger.on_training_end()


    def test_model(self, dataset: GraphCreator, n_iter=1000, verbose=True):
        """
        Evaluates the model by feeding n_iter batches to the decoder and 
        calculating the mean and standard deviation of the accuracy. 
        """
        self.eval()
        accuracy_list = torch.zeros(n_iter)
        data_time, model_time = 0, 0
        for i in tqdm(range(n_iter), disable=not verbose):
            t0 = time.perf_counter()
            x, edge_index, batch_labels, label_map, edge_attr, aligned_flips, lengths, last_label = dataset.generate_batch()
            t1 = time.perf_counter() 
            out, final_prediction = self.forward(x, edge_index, edge_attr, batch_labels, label_map)
            t2 = time.perf_counter()
            accuracy_list[i] = (torch.sum(torch.round(final_prediction) == last_label) / torch.numel(last_label)).item()
            data_time += t1 - t0
            model_time += t2 - t1
        accuracy = accuracy_list.mean()
        std = standard_deviation(accuracy, n_iter * dataset.batch_size)
        if verbose:
            print(f"Accuracy: {accuracy:.4f}, data time = {data_time:.3f}, model time = {model_time:.3f}")
        return accuracy, std