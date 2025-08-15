import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))  # Add parent dir for imports

import time
import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from args import Args
from training_utils import (
    GraphConvLayer, TrainingLogger, group,
    generate_batches_async, save_checkpoint, load_checkpoint
)
from decoder_binary.graph_creator import GraphCreator
import wandb
os.environ["WANDB_SILENT"] = "True"


class GRUDecoder(nn.Module):
    def __init__(self, args: Args):
        super().__init__()
        self.args = args

        # Dropout-layer för GNN embeddings
        self.dropout_gnn = nn.Dropout(p=0.2)

        # Bygg GNN embedding-lager
        features = list(zip(args.embedding_features[:-1], args.embedding_features[1:]))
        self.embedding = nn.ModuleList([GraphConvLayer(a, b) for a, b in features])

        # GRU med dropout mellan lager
        self.rnn = nn.GRU(
            args.embedding_features[-1],
            args.hidden_size,
            num_layers=args.n_gru_layers,
            dropout=0.3,
            batch_first=True
        )

        # Slutlig decoder
        self.decoder = nn.Sequential(
            nn.Linear(args.hidden_size, 1),
            nn.Sigmoid()
        )

    def _embed_graph(self, x, edge_index, edge_attr, batch_labels):
        for layer in self.embedding:
            x = layer(x, edge_index, edge_attr)
            x = self.dropout_gnn(x)
        return global_mean_pool(x, batch_labels)

    def forward(self, x, edge_index, edge_attr, batch_labels, label_map):
        x = self._embed_graph(x, edge_index, edge_attr, batch_labels)
        x = group(x, label_map)
        out_packed, h = self.rnn(x)
        out, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)
        predictions = self.decoder(out).squeeze(-1)
        final_prediction = self.decoder(h[-1])
        return predictions, final_prediction

    def train_model(self, logger: TrainingLogger | None = None, save: str | None = None) -> None:
        if self.args.log_wandb:
            wandb.init(project="GNN-RNN-repetition-code", name=save, config=self.args)

        if isinstance(logger, TrainingLogger):
            logger.on_training_begin(self.args)

        gc = GraphCreator(self.args)
        gc.train_val_split()

        # AdamW med weight decay
        optim = AdamW(self.parameters(), lr=self.args.lr, weight_decay=1e-4)

        # ReduceLROnPlateau baserat på val_loss
        scheduler = ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=5, min_lr=self.args.min_lr)

        best_val_acc = 0
        no_improve = 0
        start_epoch = 1

        # Ladda ev. pretrained
        if self.args.pretrained_checkpoint:
            ckpt_epoch = load_checkpoint(
                self, self.args.pretrained_checkpoint,
                optimizer=optim if self.args.resume else None,
                scheduler=scheduler if self.args.resume else None,
                resume=self.args.resume
            )
            if self.args.resume:
                start_epoch = ckpt_epoch + 1

        # Mindre batch size för bättre generalisering
        self.args.batch_size = 512

        validation_batches = list(gc.generate_batches(mode="validation"))
        validation_batches = [tuple(t.to(self.args.device) for t in batch) for batch in validation_batches]

        for epoch in range(start_epoch, self.args.n_epochs + 1):
            if isinstance(logger, TrainingLogger):
                logger.on_epoch_begin(epoch)

            self.train()
            epoch_train_loss, epoch_train_acc = 0.0, 0.0
            num_train_batches = 0

            _, get_next_batch = generate_batches_async(gc, mode="training", max_prefetch=5)
            t1 = time.perf_counter()

            while True:
                batch = get_next_batch()
                if batch is None:
                    break

                optim.zero_grad()
                batch = [t.to(self.args.device, non_blocking=True) for t in batch]
                x, edge_index, batch_labels, label_map, edge_attr, aligned_flips, lengths, last_label = batch

                out, final_prediction = self.forward(x, edge_index, edge_attr, batch_labels, label_map)

                if self.args.train_all_times:
                    mask = torch.arange(out.size(1), device=out.device)[None, :] < lengths[:, None]
                    loss_raw = nn.functional.binary_cross_entropy(out, aligned_flips, reduction='none')
                    loss = (loss_raw * mask).sum() / mask.sum()
                else:
                    loss = nn.functional.binary_cross_entropy(final_prediction, last_label)

                loss.backward()
                optim.step()

                epoch_train_loss += loss.item()
                epoch_train_acc += (torch.sum(torch.round(final_prediction) == last_label) / torch.numel(last_label)).item()
                num_train_batches += 1

            model_time = time.perf_counter() - t1

            # Validering
            self.eval()
            val_total_correct, val_total_loss, val_total_elements = 0, 0.0, 0
            with torch.no_grad():
                for batch in validation_batches:
                    x, edge_index, batch_labels, label_map, edge_attr, aligned_flips, lengths, last_label= batch
                    out, final_prediction = self.forward(x, edge_index, edge_attr, batch_labels, label_map)

                    if self.args.train_all_times:
                        mask = torch.arange(out.size(1), device=out.device)[None, :] < lengths[:, None]
                        loss_raw = nn.functional.binary_cross_entropy(out, aligned_flips, reduction='none')
                        loss = (loss_raw * mask).sum() / mask.sum()
                    else:
                        loss = nn.functional.binary_cross_entropy(final_prediction, last_label)

                    val_total_loss += loss.item()
                    val_total_correct += torch.sum(torch.round(final_prediction) == last_label).item()
                    val_total_elements += last_label.numel()

                # Fysisk accuracy (alla samples)
                epoch_val_acc = val_total_correct / val_total_elements
                epoch_val_loss = val_total_loss / len(validation_batches)

                # Logical accuracy (med triviala inkluderade)
                epoch_val_log_acc = (val_total_correct + gc.val_num_trivial) / gc.val_size

            scheduler.step(epoch_val_loss)

            metrics = {
                "train_loss":    epoch_train_loss,
                "train_acc":     epoch_train_acc,
                "val_loss":      epoch_val_loss,
                "val_acc":       epoch_val_acc,
                "val_log_acc":   epoch_val_log_acc,
                "learning_rate": scheduler.get_last_lr()[0],
                "model_time":    model_time
            }

            if self.args.log_wandb:
                wandb.log(metrics)
            if isinstance(logger, TrainingLogger):
                logger.on_epoch_end(logs=metrics)

            # Early stopping
            if epoch_val_acc > best_val_acc:
                best_val_acc = epoch_val_acc
                no_improve = 0
                if save:
                    ckpt_path = f"./models/{save}.pt"
                    save_checkpoint(self, ckpt_path, optim, scheduler, epoch)
                    print(f"Saved new best model (log.acc.={epoch_val_log_acc:.5f} loss={epoch_val_loss}) at epoch {epoch} → {ckpt_path}")
            else:
                no_improve += 1
                if no_improve >= self.args.patience:
                    print(f"Early stopping: no improvement in {self.args.patience} epochs.")
                    break
    
    def test_model(self) -> tuple:
        """
        Utvärderar modellen på test-datasetet och beräknar fysisk och logisk accuracy
        med konsekvent viktning över alla exempel.
        """

        gc = GraphCreator(self.args)
        gc.train_val_split()
        gc.print_info()

        test_batches = list(gc.generate_batches(mode="validation"))
        test_batches = [tuple(t.to(self.args.device) for t in batch) for batch in test_batches]

        self.eval()
        total_loss = 0.0
        total_correct = 0
        total_elements = 0
        data_time = 0.0
        model_time = 0.0

        all_final_preds = []
        all_last_labels = []

        with torch.no_grad():
            for batch in tqdm(test_batches, desc="Evaluating model on test batches"):
                t0 = time.perf_counter()
                x, edge_index, batch_labels, label_map, edge_attr, aligned_flips, lengths, last_label = batch
                t1 = time.perf_counter()

                out, final_pred = self.forward(x, edge_index, edge_attr, batch_labels, label_map)
                t2 = time.perf_counter()

                if self.args.train_all_times:
                    mask = torch.arange(out.size(1), device=out.device)[None, :] < lengths[:, None]
                    loss_raw = nn.functional.binary_cross_entropy(out, aligned_flips, reduction='none')
                    loss = (loss_raw * mask).sum() / mask.sum()
                else:
                    loss = nn.functional.binary_cross_entropy(final_pred, last_label)

                total_loss += loss.item()
                total_correct += torch.sum(torch.round(final_pred) == last_label).item()
                total_elements += last_label.numel()

                data_time  += (t1 - t0)
                model_time += (t2 - t1)

                all_final_preds.append(final_pred.cpu())
                all_last_labels.append(last_label.cpu())

        # Physical accuracy (global viktning)
        physical_acc = total_correct / total_elements

        # Logical accuracy (inkluderar triviala)
        logical_acc = (total_correct + gc.val_num_trivial) / gc.val_size

        avg_loss = total_loss / len(test_batches)

        print(
            f"Test Result → Loss: {avg_loss:.4f}, "
            f"Physical Acc: {physical_acc:.4f}, "
            f"Logical Acc: {logical_acc:.4f} "
            f"(data_time={data_time:.3f}s, model_time={model_time:.3f}s)"
        )

        all_final_preds = torch.cat(all_final_preds)
        all_last_labels = torch.cat(all_last_labels)

        return avg_loss, physical_acc, logical_acc