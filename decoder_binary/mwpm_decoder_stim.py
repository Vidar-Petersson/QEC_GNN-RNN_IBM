import pymatching 
from old_stuff.data_stim import Dataset
from training_utils import standard_deviation
import numpy as np
from tqdm import tqdm
import time
import torch

def test_mwpm(dataset: Dataset, n_iter=1000, verbose=True):
    """
    Evaluates MWPM by feeding it n_iter batches and calculating
    the mean and standard deviation of the accuracy. 
    """
    sampler_idx = 0 # Specifies which sampler to use. Should generally be 0 as we usually have a set error rate / syndrome length combination during evaluation.
    detector_error_model = dataset.circuits[sampler_idx].detector_error_model(decompose_errors=True)
    matcher = pymatching.Matching.from_detector_error_model(detector_error_model)
    accuracy_list = torch.zeros(n_iter)
    data_time, model_time = 0, 0
    for i in tqdm(range(n_iter), disable=not verbose):
        t0 = time.perf_counter()
        detection_array, flips_array = dataset.sample_syndromes(sampler_idx) 
        data_time += time.perf_counter() - t0
        predictions = matcher.decode_batch(detection_array) 
        model_time += time.perf_counter() - t0
        accuracy_list[i] = np.sum(predictions == flips_array) / dataset.batch_size
    accuracy = accuracy_list.mean()
    std = standard_deviation(accuracy, n_iter * dataset.batch_size)
    if verbose:
        print(f"Accuracy: {accuracy:.4f}, data time = {data_time:.3f}, model time = {model_time:.3f}")
    return accuracy, std
