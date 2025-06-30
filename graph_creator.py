import numpy as np
import torch
from tqdm import tqdm
import time, itertools
from args import Args
from torch_geometric.nn.pool import knn_graph
from dataloader_ibm import IBMSampler

class GraphCreator:
    """
    Class that is used to generate graphs of errors that occur in quantum computers. 
    Call generate_batch() to generate a batch of graphs.
    """
    def __init__(self, args: Args):
        self.device = args.device
        self.error_rates = args.error_rates 
        self.batch_size = 1 # hardcoded args.batch_size
        self.t = args.t[0]
        self.dt = args.dt 
        self.distance = args.distance
        self.n_stabilizers = self.distance - 1
        self.sliding = args.sliding
        self.k = args.k
        self.norm = args.norm
        self.simulator = args.simulator_backend
        
        if not self.sliding:
            for t in args.t:
                assert t % args.dt == args.dt - 1 # Syftet med detta?

        # combinations = list(itertools.product(self.error_rates, self.t))
        # if len(combinations) > 1:
        #     print("Warning multiple errors and time steps inserted. Choosing only first to sample.")
        #     self.t = self.t[0]
        #     # Ignoring errors in the implementation with IBM backend
        
        t0 = time.perf_counter() 
        self.IBMSampler = IBMSampler(distance=self.distance, t=self.t, simulator=self.simulator)
        self.syndromes, self.flips = self.IBMSampler.load_jobdata() # Includes trivial syndromes, size as original file
        self.filename = self.IBMSampler.filename
        t1 = time.perf_counter()
        self.trivial_syndrome_mask = np.any(self.syndromes, axis=1) # Mask for trivial syndromes where no detection event happend
        print(f"Loaded IBM jobdata (d={self.distance}, t={self.t}) with {self.flips.shape[0]} shots ({np.mean(~self.trivial_syndrome_mask)*100:.1f}% trivial) in {t1-t0:.2f} s.")

        def _generate_detector_coordinates(d, t): # Lägga denna separat?
            d -= 1
            col0 = np.tile(np.arange(d), t)
            col1 = np.zeros(d * t, dtype=np.int64)
            col2 = np.repeat(np.arange(t), d)
            return np.stack((col0, col1, col2), axis=1)

        self.detector_coordinates = _generate_detector_coordinates(self.distance, self.t)
        self.stabilizer_mask = np.ones((1, self.distance-1), dtype=np.uint8) # Mask for type of stabiliser, not needed for repetition code
        
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
            sampler_idx: Index into the list of circuits/samplers. This determines 
                which surface code configuration (e.g., error rate, number of rounds) 
                was used for this batch.

        Returns:
            node_features: ndarray of shape [n, 5] where each row is (x, y, t, type_x, type_z).
                - x, y, t: spatial and temporal position of a detection event
                - type_x, type_z: one-hot encoding of stabilizer type
            batch_labels: ndarray of shape [n], mapping each node to a batch element
            chunk_labels: ndarray of shape [n], mapping each node to a time chunk (graph)
        Note:
            - When setting the number of QEC rounds to t, stim will return t + 1 X
            stabilizers, and t - 1 Z stabilizers. In total, there are t + 1 'time 
            steps'. This means, that there are g = t - dt + 2 chunks for each shot. 
        """

        # Decode syndrome indices into (x, y, t) coordinates using precomputed Stim detector layout
        # Result: list of arrays, one per shot, each with shape [num_events_in_shot, 3]
        node_features = [self.detector_coordinates[s] for s in syndromes] #Borde vara OK men dubbelkolla logiken

        if self.sliding:
            # Total number of rounds used in this circuit
            sampler_t = self.t # self.circuits[sampler_idx].num_detectors // self.n_stabilizers
            # Apply a sliding window over time to divide events into overlapping chunks
            # Returns updated node_features with local time coordinates and chunk_labels
            node_features, chunk_labels = self.get_sliding_window(node_features, sampler_t)
        
        # Construct a batch_labels array that repeats batch indices according to number of events
        # Example: if shot 0 has 3 events and shot 1 has 5, this will be [0, 0, 0, 1, 1, 1, 1, 1]
        # batch_labels = np.repeat(np.arange(len(node_features)), [len(i) for i in node_features])
        num_nodes_per_shot = np.array([len(i) for i in node_features])
        batch_indices = np.arange(len(node_features)).repeat(num_nodes_per_shot)
        batch_labels = batch_indices

        # Combine all node features into a single array [total_nodes, 3]
        node_features = np.vstack(node_features)
        
        if not self.sliding:
            # If sliding window is not used, manually compute chunk index and local time:
            #   - chunk = t // dt
            #   - local_t = t % dt
            chunk_labels = node_features[:, -1] // self.dt
            node_features[:, -1] = node_features[:, -1] % self.dt

        # Determine stabilizer type at each (x, y) coordinate using the precomputed mask
        # stabilizer_mask == 3 indicates Z stabilizer; else it's X stabilizer
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
        # B = int(label_map[:, 0].max().item()) + 1 
        B = max(int(label_map[:, 0].max().item()) + 1, 1)

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


    def generate_batch(self):
        """
        Generates a batch of graphs. 

        Returns: 
            node_features: tensor of shape [n, 5] ([x, y, t, (stabilizer type)]).
            edge_index: tensor of shape [n_edges, 2]. Represents the edges, 
                i.e. the adjacency matrix. 
            labels: tensor of shape [n]. Represents which node features belong
                to which combination of batch element and chunk. 
                This is used when computing global_mean_pool following
                graph convolutions. The reason being there is no 
                explicit batch dimension. Therefore, a list of 
                labels is needed to keep track of which node features
                belong to which batch element. Further, each batch element
                consists of multiple graphs, or chunks. Therefore, an integer
                is assigned to each combination of batch element and chunk.
            label_map: tensor of shape [n_graphs]. Maps labels to
                [batch element, chunk].  
            edge_attr: tensor of shape [n_edges]. Represents the edge weights.
            flips: tensor of shape [batch_size, g]. Indicates if a logical 
                bit- or phase-flip has occurred at the end of each chunk.
            last_label: tensor of shape [batch_size, 1]. Indicates if a logical 
                bit- or phase-flip has occurred at the end of the whole circuit.

        Note:
            - When setting the number of QEC rounds to t, stim will return t + 1 X
              stabilizers, and t - 1 Z stabilizers. In total, there are t + 1 'time 
              steps'. This means, that there are g = t - dt + 2 chunks for each shot. 
            - Only logical observables measured at the end of each chunk are kept,
              i.e., we discard the first `dt` entries and keep the final g entries.
            - Because stim gives t logical observables, we copy the last label to
              the last chunk ending at the last perfect stabilizer measurement
        """
        
        batches = []
        for i in range(0, self.syndromes.shape[0] - 1, 2):
            syndromes_n = self.syndromes[i:i+2]
            flips_n = self.flips[i:i+2]

            print(syndromes_n)
            print(flips_n)
            
            # Keep only labels at chunk boundaries (i.e., end of each chunk) 
            flips_n = flips_n[:, self.dt - 1:]  # shape: [batch_size, g - 1], where g = t - dt + 2
            flips_n = torch.from_numpy(flips_n).to(dtype=torch.float32, device=self.device)
            last_label = flips_n[:, -1:]  # shape [B, 1]
            flips_n = torch.cat([flips_n, last_label], dim=1)  # Append the last label one more time to get [B, g] # OK men varför?

            # Extract graph structure and labels for non-empty chunks
            node_features, batch_labels, chunk_labels = self.get_node_features(syndromes_n)
            node_features = torch.from_numpy(node_features)

            # Map each unique (batch, chunk) pair to a unique graph index
            # label_map = np.array(list(zip(batch_labels, chunk_labels)))
            label_map = np.stack((batch_labels, chunk_labels), axis=1)

            label_map, counts = np.unique(label_map, axis=0, return_counts=True)
            labels = np.repeat(np.arange(counts.shape[0]), counts).astype(np.int64)
            label_map = torch.from_numpy(label_map)
            labels = torch.from_numpy(labels)

            # Extract graph edges and attributes
            edge_index, edge_attr = self.get_edges(node_features, labels)

            # align labels with chunk indices: 
            aligned_flips, lengths = self.align_labels_to_outputs(label_map, flips_n)

            # Move everything to the appropriate device
            node_features = node_features.to(self.device)
            labels = labels.to(self.device)
            label_map = label_map.to(dtype=torch.float32, device=self.device)
            edge_index = edge_index.to(self.device)
            edge_attr = edge_attr.to(self.device)
            lengths = lengths.to(self.device)

            batches.append((
                node_features,
                edge_index,
                labels,
                label_map,
                edge_attr,
                aligned_flips,
                lengths,
                last_label
            ))

        return batches
    
if __name__ == "__main__":
    args = Args(error_rates=[0.002], t=[21], distance=3, sliding=True, dt=2, simulator_backend=True)
    dataset = GraphCreator(args)
    t0 = time.perf_counter()
    # node_features, edge_index, labels, label_map, edge_attr, aligned_flips, lengths, last_label = dataset.generate_batch()
    batches = dataset.generate_batch()
    print(batches)
    print(f"{time.perf_counter() - t0:.3f} seconds")