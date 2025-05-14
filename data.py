import stim 
import numpy as np
import numpy.typing as npt
import torch
from tqdm import tqdm
import time, itertools
import matplotlib.pyplot as plt
from enum import Enum
from args import Args
from torch_geometric.nn.pool import knn_graph

class FlipType(Enum):
    BIT = 1
    PHASE = 2

class Dataset:
    """
    Class that is used to generate graphs of errors that occur
    in quantum computers. 

    Call generate_batch() to generate a batch of graphs.

    References:
    https://github.com/quantumlib/Stim/blob/main/doc/getting_started.ipynb
    https://github.com/LangeMoritz/GNN_decoder 
    """
    def __init__(self, args: Args, flip: FlipType = FlipType.BIT):
        self.device = args.device
        self.error_rates = args.error_rates 
        self.batch_size = args.batch_size
        self.t = args.t
        self.dt = args.dt 
        self.distance = args.distance
        self.n_stabilizers = self.distance ** 2 - 1
        self.sliding = args.sliding
        self.k = args.k
        self.seed = args.seed
        self.norm = args.norm
        
        if not self.sliding:
            for t in args.t:
                assert t % args.dt == args.dt - 1 
        if flip is FlipType.BIT:
            self.code_task = "surface_code:rotated_memory_z"
        elif flip is FlipType.PHASE:
            self.code_task = "surface_code:rotated_memory_x"
        else: 
            raise AttributeError("Unknown flip type.")
        self.__init_circuit()

    def __init_circuit(self):
        """
        Creates circuits that can be sampled from. 
        """
        combinations = itertools.product(self.error_rates, self.t)
        
        self.circuits = [
            stim.Circuit.generated(
                self.code_task,
                distance=self.distance,
                rounds=t,
                after_clifford_depolarization=error_rate,
                before_round_data_depolarization=error_rate,
                before_measure_flip_probability=error_rate,
                after_reset_flip_probability=error_rate) 
            for error_rate, t in combinations]
        self.samplers = [circuit.compile_detector_sampler(seed=self.seed) for circuit in self.circuits]

        self.detector_coordinates = []
        for circuit in self.circuits:
            detector_coordinate = circuit.get_detector_coordinates()
            detector_coordinate = np.array(list(detector_coordinate.values()))
            detector_coordinate[:, :2] /= 2
            self.detector_coordinates.append(detector_coordinate.astype(np.int64))

        # Syndrome mask, i.e. locations of stabilizers. 
        sz = self.distance + 1
        syndrome_x = np.zeros((sz, sz), dtype=np.uint8)
        syndrome_x[::2, 1:sz - 1:2] = 1
        syndrome_x[1::2, 2::2] = 1
        syndrome_z = np.rot90(syndrome_x) * 3
        self.syndrome_mask = syndrome_x + syndrome_z
      
    def sample_syndromes(self, sampler_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns a batch of syndromes. 

        The syndromes are represented by:
            detection_array: an array of bools of size [b, s] where
                - b: batch size 
                - s: number of detectors, self.t * number of stabilizers
                Each entry indicates if there has been a detection event
                on the corresponding stabilizer.
            flips_array: an array of shape [b, 1].
                An entry is 1 if there has been a logical bit/phase flip 
                when measuring at the end of the syndrome, 0 otherwise. 
        """
        sampler = self.samplers[sampler_idx]
        detection_events_list, observable_flips_list = [], []
        # Sample until we get a batch where each element has at least
        # one detection event. 
        while len(detection_events_list) < self.batch_size: 
            detection_events, observable_flips = sampler.sample(
                shots=self.batch_size, separate_observables=True)
            shots_w_flips = np.sum(detection_events, axis=1) != 0 # only include cases where there is at least one detection event.
            detection_events_list.extend(detection_events[shots_w_flips, :])
            observable_flips_list.extend(observable_flips[shots_w_flips, :])
        detection_array = np.array(detection_events_list[:self.batch_size])
        flips_array = np.array(observable_flips_list[:self.batch_size], dtype=np.int32)
        return detection_array.astype(bool), flips_array
    
    def get_sliding_window(self, node_features: list[np.ndarray], sampler_t: int
                           ) -> tuple[list[np.ndarray], np.ndarray]:
        
        # TODO: Improve documentation on this method.
        """
        Returns a sliding wiondow representation of the node features
        """
        chunk_labels = [0] * len(node_features)  # number of batches

        for batch, coordinates in enumerate(node_features):
            times, counts = np.unique(coordinates[:, -1], return_counts=True)

            start = times < self.dt
            end = times > sampler_t - self.dt
            middle = ~(start | end)

            counts[start] *= (times + 1)[start]
            counts[middle] *= self.dt
            counts[end] *= -((times - 1) - sampler_t)[end]

            new_size = np.sum(counts)
            new_coordinates = np.zeros((new_size, 3), dtype=np.uint64)
            chunk_label = np.zeros(new_size, dtype=np.uint64)

#           position = 0           
#           for j in range(self.t-self.dt+2):

#               mask = (coordinates[:,-1] <j+self.dt) & (coordinates[:,-1] >= j)
#               chunk_size = np.sum(mask)

#               new_coordinates[position : position + chunk_size,:3] = coordinates[mask]
#               new_coordinates[position : position + chunk_size, -1] -= j
#               chunk_label[position: position + chunk_size] = j
#               position += chunk_size
#           node_features[batch] = new_coordinates
#           chunk_labels[batch] = chunk_label
            
            # Code above vectorized.
            j_values = np.arange(sampler_t - self.dt + 2)[:, None]  # Column vector for broadcasting
            time_column = coordinates[:, -1][None, :]  # Row vector for broadcasting

            mask = (time_column < j_values + self.dt) & (time_column >= j_values)  # Shape: (num_j, num_points)
            indices = np.where(mask)  # Get (j, idx) pairs where mask is True

            sorted_idx = np.argsort(indices[0])  # Sort by j to maintain order
            selected_points = coordinates[indices[1][sorted_idx]].copy()  # Extract selected rows
            selected_points[:, -1] -= indices[0][sorted_idx]  # Adjust time values

            new_coordinates[:len(selected_points)] = selected_points
            chunk_label[:len(selected_points)] = indices[0][sorted_idx]

            node_features[batch] = new_coordinates
            chunk_labels[batch] = chunk_label

        chunk_labels = np.concatenate(chunk_labels)
        return node_features, chunk_labels

    def get_node_features(self, syndromes: np.ndarray, sampler_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Broadcasts the syndromes of shape [b, s] to n feature 
        vectors [x, y, t, (stabilizer type)], and creates two
        vectors batch_labels and chunk_labels indicating to which batch and 
        chunk the feature vectors belong.
        The stabilizer type is encoded as either (0, 1) (X) or (1, 0) (Z).

        Returns:
            node_features: array of shape [n, 5]
            batch_labels: array of shape [n]
            chunk_labels: array of shape [n]
        """
        # Extract [x, y, t] from the stabilizers that detected something.
        node_features = [self.detector_coordinates[sampler_idx][s] for s in syndromes]

        if self.sliding:
            sampler_t = self.circuits[sampler_idx].num_detectors // self.n_stabilizers
            node_features, chunk_labels = self.get_sliding_window(node_features, sampler_t)

        batch_labels = np.repeat(np.arange(self.batch_size), [len(i) for i in node_features])
        node_features = np.vstack(node_features)

        if not self.sliding:
            chunk_labels = node_features[:, -1] // self.dt
            node_features[:, -1] = node_features[:, -1] % self.dt
        
        # Add the stabilizer type to the node features.
        stabilizer_type = self.syndrome_mask[node_features[:, 1], node_features[:, 0]] == 3
        stabilizer_type = stabilizer_type[:, np.newaxis]
        node_features = np.hstack((node_features, stabilizer_type, ~stabilizer_type)).astype(np.float32)
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
        edge_attr = torch.linalg.norm(delta, ord=self.norm, dim=1)

        # Inverse square of the norm between two nodes.
        edge_attr = 1 / edge_attr ** 2

        return edge_index, edge_attr

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
            flips: tensor of shape [batch size]. Indicates if a logical 
                bit- or phase-flip has occured. 1 if it has, 0 otherwise. 
        """
        # Sample syndromes
        sampler_idx = np.random.choice(len(self.samplers))
        syndromes, flips = self.sample_syndromes(sampler_idx)

        node_features, batch_labels, chunk_labels = self.get_node_features(syndromes, sampler_idx)
        node_features = torch.from_numpy(node_features)

        # Map each combination of [batch element, chunk] to an integer. 
        label_map = np.array(list(zip(batch_labels, chunk_labels)))
        label_map, counts = np.unique(label_map, axis=0, return_counts=True)
        labels = np.repeat(np.arange(counts.shape[0]), counts).astype(np.int64)
        label_map = torch.from_numpy(label_map)
        labels = torch.from_numpy(labels)
        
        # Extract edges
        edge_index, edge_attr = self.get_edges(node_features, labels)
        
        node_features = node_features.to(self.device)
        flips = torch.from_numpy(flips).to(self.device)
        labels = labels.to(self.device)
        
        label_map = label_map.to(self.device)
        edge_index = edge_index.to(self.device)
        edge_attr = edge_attr.to(self.device)

        return node_features, edge_index, labels, label_map, edge_attr, flips

    def plot_graph(self, node_features, edge_index, labels, graph_idx):
        node_features = node_features.cpu().numpy()
        features = node_features[labels == graph_idx]
        min_t, max_t = 0, self.dt - 1
        edge_mask = (edge_index[0] == np.nonzero(labels == graph_idx)).cpu().numpy()
        edges = edge_index[:, np.any(edge_mask, axis=0)]

        ax = plt.axes(projection="3d")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("t")
        ax.set_xlim(0, self.distance)
        ax.set_ylim(0, self.distance)
        ax.set_zlim(min_t, max_t)
        ax.set_zticks(range(min_t, max_t + 1))
        ax.view_init(elev=60, azim=-90, roll=0)
        ax.set_aspect('equal', adjustable='box')
        plt.gca().invert_yaxis()

        # Plotting nodes.
        c = ["red" if np.round(feature[3]) == 0 else "green" for feature in features]
        ax.scatter(*features.T[:3], c=c)
        
        # Plotting edges.
        edge_coordinates = node_features[edges].T[:3]
        plt.plot(*edge_coordinates, c="blue", alpha=0.3)
       
        # Plotting stabilizers.
        x_stabs = np.nonzero(self.syndrome_mask == 1)
        z_stabs = np.nonzero(self.syndrome_mask == 3)
        ax.scatter(x_stabs[1], x_stabs[0], min_t, c="red",  alpha=0.3, s=50, label="X stabilizers")
        ax.scatter(z_stabs[1], z_stabs[0], min_t, c="green", alpha=0.3, s=50, label="Z stabilizers")
        plt.legend()
        plt.show()
        
if __name__ == "__main__":
    args = Args(error_rates=[0.002], t=[99], sliding=True, dt=8)
    dataset = Dataset(args)
    t0 = time.perf_counter()
    node_features, edge_index, labels, label_map, edge_attr, flips = dataset.generate_batch()
    for i in tqdm(range(10)):
        dataset.plot_graph(node_features, edge_index, labels, i)
    print(f"{time.perf_counter() - t0:.3f} seconds")

