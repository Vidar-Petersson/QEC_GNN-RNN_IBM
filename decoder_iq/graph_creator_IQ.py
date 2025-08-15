import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))  # Add parent dir for imports

import numpy as np
import torch
from tqdm import tqdm
from args import Args
from torch_geometric.nn.pool import knn_graph
from decoder_iq.dataloader_ibm_IQ import IBMSampler
from data_analysis.data_characteristics import *

class GraphCreator:
    """
    Class that is used to generate graphs of errors that occur in quantum computers. 
    Call generate_batches() to generate batches of graphs.
    """
    def __init__(self, args: Args):
        self.device = args.device
        self.error_rates = args.error_rates 
        self.batch_size = args.batch_size
        self.t = args.t
        self.dt = args.dt 
        self.distance = args.distance
        self.n_stabilizers = self.distance - 1
        self.sliding = args.sliding
        self.k = args.k
        self.norm = args.norm
        self.simulator = args.simulator_backend
        self.val_fraction = args.val_fraction
        self.seed = args.seed
        self.threshold = args.detection_threshold

        self.IBMSampler = IBMSampler(args)
        self.detections_probs, self.flips_probs = self.IBMSampler.load_jobdata(verbose=True) # Includes trivial syndromes, size as original file
        self.detector_coordinates = self._generate_detector_coordinates(self.distance, self.t)

    @staticmethod
    def _generate_detector_coordinates(d, t):
        d -= 1
        x = np.tile(np.arange(d), t)            # shape: [d*t]
        times = np.repeat(np.arange(t), d)      # shape: [d*t]
        return np.stack((x, times), axis=1)     # shape: [d*t, 2]

    def get_sliding_window(self, node_features: list[np.ndarray], sampler_t: int
                        ) -> tuple[list[np.ndarray], np.ndarray]:
        """
        Applies a sliding window to the input node features in time,
        segmenting each shot's data into overlapping time chunks.

        This is used to divide each graph (shot) into smaller graph segments
        that span dt rounds of the circuit. The result is a per-chunk 
        representation suitable for sequential processing (e.g., in an RNN).

        Args:
            node_features: List of length batch_size. Each element is an array of 
                shape [n_i, 3] containing the node features (x, y, t) for a single
                shot (i.e., detection events).
            sampler_t: The number of rounds used in the circuit (i.e., full time duration, = t).

        Returns:
            A tuple (node_features, chunk_labels):
                node_features: Modified list where each entry's coordinates are mapped 
                    into chunk-local time and reordered to align with chunk boundaries.
                chunk_labels: A 1D array indicating to which chunk (window) each 
                    node in the batch belongs. This is later used for pooling and batching.
        Note:
            There are g = t - dt + 2 chunks for each shot. 
        """
        dt = self.dt
        g = sampler_t - dt + 2  # Number of chunks

        updated_node_features = []
        all_chunk_labels = []

        j_values = np.arange(g)[:, None]  # Shape: [g, 1]

        for coordinates in node_features:
            times = coordinates[:, -1][None, :]  # Shape: [1, num_points]

            # Mask: [g, num_points] where True if time falls in window [j, j+dt)
            mask = (times >= j_values) & (times < j_values + dt)

            chunk_idx, point_idx = np.where(mask)
            sorted_idx = np.argsort(chunk_idx)

            selected_coords = coordinates[point_idx[sorted_idx]].copy()
            selected_coords[:, -1] -= chunk_idx[sorted_idx]  # Convert to chunk-local time

            updated_node_features.append(selected_coords)
            all_chunk_labels.append(chunk_idx[sorted_idx])

        return updated_node_features, np.concatenate(all_chunk_labels)


    def get_node_features(self, detections: np.ndarray, detection_probs: np.ndarray):
        """
        Converts detection event indices into physical node features and assigns 
        them to batch and chunk labels, optionally applying a sliding window.

        Args:
            detections: Boolean array of shape [batch_size, s] where s is the total
                number of detectors (t * number of stabilizers). Each entry indicates
                whether a detection event occurred at a given space-time location.

        Returns:
            node_features: ndarray of shape [n, 2] where each row is (x, t).
                - x, t: spatial and temporal position of a detection event
            batch_labels: ndarray of shape [n], mapping each node to a batch element
            chunk_labels: ndarray of shape [n], mapping each node to a time chunk (graph)
        """
        # Decode syndrome indices into (x, t) coordinates using precomputed detector layout
        coords_list = [self.detector_coordinates[s] for s in detections]  # varje elem: [n_i, 2]
        probs_list  = [dp[s]                    for dp, s in zip(detection_probs, detections)]

        if self.sliding:
            # Apply a sliding window over time to divide events into overlapping chunks
            # Returns updated node_features with local time coordinates and chunk_labels
            #coords_list, chunk_labels = self.get_sliding_window(coords_list, self.t)


            dt = self.dt
            g  = self.t - dt + 2
            j_values = np.arange(g)[:, None]   # shape [g,1]

            new_coords = []
            new_probs  = []
            all_chunks = []

            for coords, probs in zip(coords_list, probs_list):
                # coords: [n_i, 2], probs: [n_i]
                times = coords[:,1][None, :]     # shape [1, n_i]
                mask  = (times >= j_values) & (times < j_values + dt)
                chunk_idx, point_idx = np.where(mask)
                order = np.argsort(chunk_idx)

                # välj ut och justera coords
                sel = point_idx[order]
                chunks = chunk_idx[order]
                c2 = coords[sel].copy()
                c2[:,1] -= chunks            # gör tiderna lokala inom fönstret

                # välj ut motsvarande probs
                p2 = probs[sel]              # shape [ (#sel), ]

                new_coords.append(c2)
                new_probs.append(p2)
                all_chunks.append(chunks)

            coords_list  = new_coords
            probs_list   = new_probs
            chunk_labels = np.concatenate(all_chunks)
        else:
            chunk_labels = np.concatenate([
                coords[:, 1] // self.dt for coords in coords_list
            ])
            for coords in coords_list:
                coords[:, 1] %= self.dt

        # Construct a batch_labels array that repeats batch indices according to number of events
        # Example: if shot 0 has 3 events and shot 1 has 5, this will be [0, 0, 0, 1, 1, 1, 1, 1]
        batch_labels = np.repeat(np.arange(len(coords_list)), [len(c) for c in coords_list])
        # Combine all node features into a single array [total_nodes, 3]
        #node_features = np.vstack(coords_list).astype(np.float32)  # shape [N,2]
        node_features = np.vstack([np.hstack([coords_list[i], probs_list[i][:,None]]) for i in range(len(coords_list))]).astype(np.float32) # shape [N,3]
        return node_features, batch_labels, chunk_labels

    
    def get_edges(self, node_features: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute graph edges and their weights. Everything stays on GPU.

        Args:
            node_features: [num_nodes, feature_dim] on GPU
            labels: [num_nodes] batch labels for each node

        Returns:
            edge_index: [2, num_edges]
            edge_attr: [num_edges]
        """

        edge_index = knn_graph(node_features, k=self.k, batch=labels, loop=False)

        row, col = edge_index  # shape: [num_edges]
        diffs = node_features[row] - node_features[col]  # shape: [num_edges, dim]
        
        # Efficient norm computation (inf-norm ≈ max(|x|))
        if self.norm == torch.inf:
            dists = diffs.abs().max(dim=1).values
        elif self.norm == 2:
            dists = torch.norm(diffs, p=2, dim=1)
        else:
            raise ValueError(f"Unsupported norm: {self.norm}")

        # Avoid division by zero
        dists = dists.clamp(min=1e-10)
        edge_attr = 1 / (dists ** 2)

        return edge_index, edge_attr

    def align_labels_to_outputs(self, label_map: torch.Tensor, flips_full: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Aligns logical flip labels to GRU outputs.

        Given indices of valid chunks (`label_map`), this returns their corresponding
        labels (`flips_full`) in a [B, max_len]-shaped tensor, ordered by batch and
        chunk position. Padding is applied to match the longest sequence.

        Parameters
        ----------
        label_map : torch.Tensor
            Tensor of shape [n, 2] with (batch_idx, chunk_idx) for valid chunks.
        flips_full : torch.Tensor
            Tensor of shape [B, g] with flip labels for all chunks.

        Returns
        -------
        aligned_flips : torch.Tensor
            Tensor of shape [B, max_len] with labels ordered and padded.
        lengths : torch.Tensor
            Tensor of shape [B] with number of valid chunks per batch.
        """
        B = int(label_map[:, 0].max().item()) + 1  # Number of batches
        lengths = torch.bincount(label_map[:, 0].long(), minlength=B)  # Number of chunks per batch
        max_len = lengths.max().item()  # Longest sequence length (for padding)

        batch_idx = label_map[:, 0].long()  # Batch indices of valid chunks
        chunk_idx = label_map[:, 1].long()  # Chunk indices within each batch

        # Sort by batch index to group chunks belonging to the same batch
        sorted_indices = torch.argsort(batch_idx)
        batch_idx = batch_idx[sorted_indices]
        chunk_idx = chunk_idx[sorted_indices]

        # Compute the position of each chunk within its batch using a range per group
        _, counts = batch_idx.unique_consecutive(return_counts=True)
        pos_in_batch = torch.cat([
            torch.arange(c) for c in counts
        ])

        # Create the output tensor and place the flip labels at the correct position
        aligned_flips = torch.zeros(B, max_len)
        aligned_flips[batch_idx, pos_in_batch] = flips_full[batch_idx, chunk_idx]

        return aligned_flips, lengths  # lengths can be used for masking later

    def train_val_split(self):

        num_total = self.detections_probs.shape[0]
        self.val_size = int(num_total * self.val_fraction)

        rng  = np.random.default_rng(self.seed)
        perm = rng.permutation(num_total)

        # Indexera direkt med permuteringen
        val_idx   = perm[:self.val_size]
        train_idx = perm[self.val_size:]

        self.train_detections_probs, self.train_flips_probs = self.detections_probs[train_idx], self.flips_probs[train_idx] 
        self.val_detections_probs, self.val_flips_probs     = self.detections_probs[val_idx], self.flips_probs[val_idx]

        self.train_detections = (self.train_detections_probs >= self.threshold)
        self.val_detections   = (self.val_detections_probs >= self.threshold)
        self.train_flips = (self.train_flips_probs >= 0.5)
        self.val_flips = (self.val_flips_probs >= 0.5)

        train_triv_syndrome_mask = np.any(self.train_detections, axis=1)
        val_triv_syndrome_mask   = np.any(self.val_detections, axis=1)
        self.val_num_trivial     = np.sum(~val_triv_syndrome_mask)

        self.train_detections, self.train_flips           = self.train_detections[train_triv_syndrome_mask], self.train_flips[train_triv_syndrome_mask] 
        self.train_detections_soft, self.train_flips_soft = self.train_detections_probs[train_triv_syndrome_mask], self.train_flips_probs[train_triv_syndrome_mask] 
        self.val_detections, self.val_flips               = self.val_detections[val_triv_syndrome_mask], self.val_flips[val_triv_syndrome_mask]
        self.val_detections_soft, self.val_flips_soft     = self.val_detections_probs[val_triv_syndrome_mask], self.val_flips_probs[val_triv_syndrome_mask]

    def print_info(self):
        print("--------------------")
        print(f"Train/val-split: {self.train_detections.shape[0]} / {self.val_detections.shape[0]}")
        analyze_class_balance(self.train_flips, self.val_flips)
        analyze_pdet_time(self.train_detections)
        print("--------------------")

    def generate_batches(self, mode: str = "training"):
        """
        Generates batches of graphs from the entire dataset, where each batch 
        contains self.batch_size datapoints (i.e., shots).

        Returns:
            List of batches. Each batch is a tuple:
                node_features, edge_index, labels, label_map,
                edge_attr, aligned_flips, lengths, last_label
        """

        if mode == "validation":
            detections = self.val_detections
            detections_soft = self.val_detections_soft
            flips = self.val_flips
            flips_soft = self.val_flips_soft
        elif mode == "training":
            detections = self.train_detections
            detections_soft = self.train_detections_soft
            flips = self.train_flips
            flips_soft = self.train_flips_soft
        else:
            raise NotImplementedError
            
        perm = np.random.permutation(detections.shape[0])
        detections = detections[perm]
        detections_soft = detections_soft[perm]
        flips = flips[perm]
        flips_soft = flips_soft[perm]

        num_total = detections.shape[0]
        batch_size = self.batch_size
        flips = flips[:, self.dt - 1:]  # shape: [batch_size, g - 1], where g = t - dt + 2
        flips_soft = flips_soft[:, self.dt - 1:]  # shape: [batch_size, g - 1], where g = t - dt + 2
        flips = torch.from_numpy(flips).float()
        flips_soft = torch.from_numpy(flips_soft).float()
        # Append the last label one more time to get [B, g]
        last_label = flips[:, -1:]  # shape [B, 1]
        last_label_soft = flips_soft[:, -1:]  # shape [B, 1]
        flips = torch.cat([flips, last_label], dim=1)  # shape [B, g]
        flips_soft = torch.cat([flips_soft, last_label_soft], dim=1)  # shape [B, g]


        for i in tqdm(range(0, num_total, batch_size), desc=f"Generating {mode} batches", disable=True):
            det_batch = detections[i:i+batch_size]
            det_batch_soft = detections_soft[i:i+batch_size]
            flips_batch = flips[i:i+batch_size]
            flips_batch_soft = flips_soft[i:i+batch_size]

            # Extract graph structure and labels for non-empty chunks
            node_features, batch_labels, chunk_labels = self.get_node_features(det_batch, det_batch_soft)
            node_features = torch.from_numpy(node_features)

            # Map each unique (batch, chunk) pair to a unique graph index
            label_map = np.array(list(zip(batch_labels, chunk_labels)))
            label_map, counts = np.unique(label_map, axis=0, return_counts=True)
            labels = np.repeat(np.arange(counts.shape[0]), counts).astype(np.int64)
            label_map = torch.from_numpy(label_map).long().float()
            labels = torch.from_numpy(labels).to(node_features.device) # Ensure node_features and labels are on the same device for knn_graph

            # Extract graph edges and attributes
            edge_index, edge_attr = self.get_edges(node_features, labels)
            # align labels with chunk indices: 
            aligned_flips, lengths = self.align_labels_to_outputs(label_map, flips_batch)
            last_label_batch = flips_batch[:, -1:]
            last_label_batch_soft = flips_batch_soft[:, -1:]

            yield (
                node_features,
                edge_index,
                labels,
                label_map,
                edge_attr,
                aligned_flips, # Bara för train all times
                lengths, # -:-
                last_label_batch,
                last_label_batch_soft
            )

if __name__ == "__main__":
    args = Args(t=6, distance=3, sliding=True, dt=2, simulator_backend=False, sub_dir="/iq_data")
    gc = GraphCreator(args)
    gc.train_val_split()
    gc.print_info()
    val_bathces = list(gc.generate_batches(mode="validation"))
    train_batches = list(gc.generate_batches(mode="training"))

    #print(f"Generated {len(train_batches)} training batches of size {args.batch_size}")