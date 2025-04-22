# Importing the required libraries
import mne
import pickle
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import butter, filtfilt, hilbert
import multiprocessing
from mne import io
import pyedflib
import scipy
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable 
from tqdm import tqdm
from sklearn.decomposition import PCA
import random
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr

# Ignoring warnings
import warnings
warnings.filterwarnings("ignore")

# Emptying CUDA cache (for freeing up SCC GPU space)
import gc
gc.collect()
torch.cuda.empty_cache()

# Setting GPU/CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device

#functions for augmentaton
def add_noise(data, noise_level = 0.1):
    """
    Function to augment by adding noise
    """
    noise = np.random.normal(0, noise_level, data.shape)
    return data + noise


def shuffle_channels(data):
    """Function to augment by shuffling channels"""
    original_indices = np.arange(data.shape[0])
    new_indices = np.copy(original_indices)
    np.random.shuffle(new_indices)
    shuffled_data = data[new_indices]
    return shuffled_data, new_indices


def smooth_time_mask(data, mask_duration, fs, smoothing_factor=0.25):
    """Function to augment by smooth time masking"""
    num_samples = data.shape[1]
    mask_length = int(mask_duration * fs)
    mask_start = np.random.randint(0, num_samples - mask_length)
    mask = np.ones(num_samples)   
    smooth_window = int(mask_length * smoothing_factor)
    ramp = np.linspace(0, 1, smooth_window)
    mask[mask_start:mask_start+smooth_window] *= ramp
    mask[mask_start+smooth_window:mask_start+mask_length-smooth_window] = 0
    mask[mask_start+mask_length-smooth_window:mask_start+mask_length] *= ramp[::-1]
    return data * mask