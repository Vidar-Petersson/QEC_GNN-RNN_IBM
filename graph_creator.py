import numpy as np
import torch
from tqdm import tqdm
import time
from args import Args
from torch_geometric.nn.pool import knn_graph
from dataloader_ibm import IBMSampler

class GraphCreator:
    """
    Class that is used to generate graphs of errors that occur in quantum computers. 
    Call generate_batch() to generate batches of graphs.
    """
    def __init__(self, args: Args):
        self.device = args.device
        self.error_rates = args.error_rates 
        self.batch_size = args.batch_size
        self.t = args.t[0]
        self.dt = args.dt 
        self.distance = args.distance
        self.n_stabilizers = self.distance - 1
        self.sliding = args.sliding
        self.k = args.k
        self.norm = args.norm
        self.simulator = args.simulator_backend
        self.val_fraction = args.val_fraction
        
        t0 = time.perf_counter() 
        self.IBMSampler = IBMSampler(distance=self.distance, t=self.t, simulator=self.simulator)
        self.syndromes, self.flips = self.IBMSampler.load_jobdata() # Includes trivial syndromes, size as original file
        self.filename = self.IBMSampler.filename
        
        self.syndromes, self.flips = self.syndromes[:50000,:], self.flips[:50000,:] # Only for light testing
        trivial_syndrome_mask = np.any(self.syndromes, axis=1) # Mask for trivial syndromes where no detection event happend
        t1 = time.perf_counter()
        print(f"Loaded IBM jobdata {self.filename} (d={self.distance}, t={self.t}) with {self.syndromes.shape[0]} shots ({np.mean(~trivial_syndrome_mask)*100:.1f}% trivial) in {t1-t0:.2f} s.")
        self.syndromes, self.flips = self.syndromes[trivial_syndrome_mask], self.flips[trivial_syndrome_mask]


        def _generate_detector_coordinates(d, t): # Lägga denna separat?
            d -= 1
            col0 = np.tile(np.arange(d), t)
            col1 = np.zeros(d * t, dtype=np.int64)
            col2 = np.repeat(np.arange(t), d)
            return np.stack((col0, col1, col2), axis=1)

        self.detector_coordinates = _generate_detector_coordinates(self.distance, self.t)
        self.stabilizer_mask = np.ones((1, self.distance-1), dtype=np.uint8) # Mask for type of stabiliser, not exactly needed for the repetition code
        
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


    def get_node_features(self, syndromes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Converts detection event indices into physical node features and assigns 
        them to batch and chunk labels, optionally applying a sliding window.

        Args:
            syndromes: Boolean array of shape [batch_size, s] where s is the total
                number of detectors (t * number of stabilizers). Each entry indicates
                whether a detection event occurred at a given space-time location.

        Returns:
            node_features: ndarray of shape [n, 5] where each row is (x, y, t, type_x, type_z).
                - x, y, t: spatial and temporal position of a detection event
                - type_x, type_z: one-hot encoding of stabilizer type
            batch_labels: ndarray of shape [n], mapping each node to a batch element
            chunk_labels: ndarray of shape [n], mapping each node to a time chunk (graph)
        """

        # Decode syndrome indices into (x, y, t) coordinates using precomputed detector layout
        # Result: list of arrays, one per shot, each with shape [num_events_in_shot, 3]
        node_features = [self.detector_coordinates[s] for s in syndromes]

        if self.sliding:
            # Total number of rounds used in this circuit
            sampler_t = self.t
            # Apply a sliding window over time to divide events into overlapping chunks
            # Returns updated node_features with local time coordinates and chunk_labels
            node_features, chunk_labels = self.get_sliding_window(node_features, sampler_t)
        
        # Construct a batch_labels array that repeats batch indices according to number of events
        # Example: if shot 0 has 3 events and shot 1 has 5, this will be [0, 0, 0, 1, 1, 1, 1, 1]
        batch_labels = np.repeat(np.arange(len(node_features)), [len(i) for i in node_features])

        # Combine all node features into a single array [total_nodes, 3]
        node_features = np.vstack(node_features)
        
        if not self.sliding:
            # If sliding window is not used, manually compute chunk index and local time:
            #   - chunk = t // dt
            #   - local_t = t % dt
            chunk_labels = node_features[:, -1] // self.dt
            node_features[:, -1] = node_features[:, -1] % self.dt

        # Determine stabilizer type at each (x, y) coordinate using the precomputed mask
        stabilizer_type = self.stabilizer_mask[node_features[:, 1], node_features[:, 0]] == 3
        stabilizer_type = stabilizer_type[:, np.newaxis]  # Shape: [n, 1]

        # Add one-hot stabilizer type to feature vector: [x, y, t, is_Z, is_X]
        node_features = np.hstack((node_features, stabilizer_type, ~stabilizer_type)).astype(np.float32) # Ta bort stabilisatortypen i framtiden för att minska antalet element per nod

        return node_features, batch_labels, chunk_labels

    
    def get_edges(self, node_features: np.ndarray, labels) -> tuple[np.ndarray, np.ndarray, int]:
        """
        Returns edges between nodes. The edges are of shape [n_edges, 2].

        Use ord=torch.inf for the supremum norm, ord=2 for euclidean norm.
        """
        # Compute edges.
        edge_index = knn_graph(node_features, self.k, batch=labels)

        # Compute the distances between the nodes:
        delta = node_features[edge_index[1]] - node_features[edge_index[0]]
        edge_attr = torch.linalg.norm(delta, ord=self.norm, dim=1) # by default self.norm = torch.inf

        # Inverse square of the norm between two nodes.
        edge_attr = 1 / edge_attr ** 2

        return edge_index, edge_attr
    
    def align_labels_to_outputs(self, label_map: torch.Tensor, flips_full: torch.Tensor) -> torch.Tensor:
        """
        Given label_map and full logical flips, return a label tensor aligned
        with the packed GRU output (i.e., labels only for non-empty chunks, in GRU order).

        Args:
            label_map: Tensor of shape [n_graphs, 2], with (batch_idx, chunk_idx)
            flips_full: Tensor of shape [B, g], with one label per possible chunk

        Returns:
            aligned_labels: Tensor of shape [B, L], aligned with GRU output
        """
        B = int(label_map[:, 0].max().item()) + 1 

        lengths = torch.bincount(label_map[:, 0].long(), minlength=B)  # number of real chunks per batch
        max_len = lengths.max().item()

        aligned_flips = torch.zeros(B, max_len, device=self.device)
        offsets = torch.zeros(B, dtype=torch.long, device=self.device)

        for i in range(label_map.size(0)):
            b = int(label_map[i, 0])
            t = int(label_map[i, 1])
            pos = offsets[b].item()
            aligned_flips[b, pos] = flips_full[b, t]
            offsets[b] += 1

        return aligned_flips, lengths  # counts = lengths for masking

    def train_val_split(self, seed=None):

        num_total = self.syndromes.shape[0]
        val_size = int(num_total * self.val_fraction)

        rng = np.random.default_rng(seed)
        perm = rng.permutation(num_total)

        # Indexera direkt med permuteringen
        val_idx = perm[:val_size]
        train_idx = perm[val_size:]

        self.train_syndromes, self.train_flips = self.syndromes[train_idx], self.flips[train_idx] 
        self.val_syndromes, self.val_flips = self.syndromes[val_idx], self.flips[val_idx]

    def generate_batch(self, mode: str = "validation"):
        """
        Generates batches of graphs from the entire dataset, where each batch 
        contains self.batch_size datapoints (i.e., shots).

        Returns:
            List of batches. Each batch is a tuple:
                node_features, edge_index, labels, label_map,
                edge_attr, aligned_flips, lengths, last_label
        """

        if mode == "validation":
            syndromes = self.val_syndromes
            flips = self.val_flips
        elif mode == "training":
            syndromes = self.train_syndromes
            flips = self.train_flips

        all_batches = []
        perm = np.random.permutation(syndromes.shape[0])
        syndromes = syndromes[perm]
        flips = flips[perm]
        
        # Keep only labels at chunk boundaries (i.e., end of each chunk)
        flips = flips[:, self.dt - 1:]  # shape: [batch_size, g - 1], where g = t - dt + 2
        flips = torch.from_numpy(flips).to(dtype=torch.float32, device=self.device)
        # Append the last label one more time to get [B, g]
        last_label = flips[:, -1:]  # shape [B, 1]
        flips = torch.cat([flips, last_label], dim=1)  # shape [B, g]

        num_total = syndromes.shape[0]
        batch_size = self.batch_size

        for i in tqdm(range(0, num_total, batch_size)):
            synd_batch = syndromes[i:i+batch_size]
            flips_batch = flips[i:i+batch_size]

            # Hoppa över om batchen blir mindre än batch_size (t.ex. sista)
            if synd_batch.shape[0] < batch_size:
                break

            # Extract graph structure and labels for non-empty chunks
            node_features, batch_labels, chunk_labels = self.get_node_features(synd_batch)
            node_features = torch.from_numpy(node_features)

            # Map each unique (batch, chunk) pair to a unique graph index
            label_map = np.array(list(zip(batch_labels, chunk_labels)))
            label_map, counts = np.unique(label_map, axis=0, return_counts=True)
            labels = np.repeat(np.arange(counts.shape[0]), counts).astype(np.int64)
            label_map = torch.from_numpy(label_map)
            labels = torch.from_numpy(labels)

            # Extract graph edges and attributes
            edge_index, edge_attr = self.get_edges(node_features, labels)

            # align labels with chunk indices: 
            aligned_flips, lengths = self.align_labels_to_outputs(label_map, flips_batch)

            # Move everything to the appropriate device
            node_features = node_features.to(self.device)
            labels = labels.to(self.device)
            label_map = label_map.to(dtype=torch.float32, device=self.device)
            edge_index = edge_index.to(self.device)
            edge_attr = edge_attr.to(self.device)
            lengths = lengths.to(self.device)
            last_label_batch = flips_batch[:, -1:]

            all_batches.append((
                node_features,
                edge_index,
                labels,
                label_map,
                edge_attr,
                aligned_flips,
                lengths,
                last_label_batch
            ))

        return all_batches

if __name__ == "__main__":
    args = Args(error_rates=[0.002], t=[21], distance=3, sliding=True, dt=2, simulator_backend=True)
    dataset = GraphCreator(args)
    # node_features, edge_index, labels, label_map, edge_attr, aligned_flips, lengths, last_label = dataset.generate_batch()
    t0 = time.perf_counter()
    batches = dataset.generate_batch()
    print(f"Generated {len(batches)} batches of size {args.batch_size}")
    print(f"First batch node_features shape: {batches[0][0].shape}")
    print(f"{time.perf_counter() - t0:.3f} seconds")