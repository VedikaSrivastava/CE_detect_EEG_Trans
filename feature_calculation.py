import os
import math
import mne
import numpy as np
import pandas as pd
import pickle
import warnings
import bisect
from scipy import signal
from scipy.signal import butter, filtfilt, hilbert, periodogram
from scipy.integrate import simpson as simps
from collections import Counter
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from numba import njit

warnings.filterwarnings("ignore")

# Bin specifications for Shannon Entropy
bin_min = -200
bin_max = 200
binWidth = 1

# -----------------------------
# Optimized Functions
# -----------------------------
@njit
def shannonEntropy_numba(eegData, bin_min, bin_max, binWidth):
    """
    Calculate Shannon Entropy for EEG data using Numba for optimization.
    Args:
        eegData (numpy.ndarray): EEG data of shape (channels, samples, epochs).
        bin_min (int): Minimum bin value.
        bin_max (int): Maximum bin value.
        binWidth (int): Width of each bin.
    Returns:
        numpy.ndarray: Shannon entropy for each channel and epoch.
    """
    n_channels = eegData.shape[0]
    n_epochs = eegData.shape[2]
    nbins = int(bin_max - bin_min - 1)
    H = np.zeros((n_channels, n_epochs))
    for chan in range(n_channels):
        for epoch in range(n_epochs):
            counts = np.zeros(nbins)
            for sample in eegData[chan, :, epoch]:
                bin_idx = int(sample - bin_min - 1)
                if bin_idx >= 0 and bin_idx < nbins:
                    counts[bin_idx] += 1
            total = 0.0
            for count in counts:
                total += count
            if total > 0:
                for count in counts:
                    if count > 0:
                        prob = count / total
                        H[chan, epoch] -= prob * math.log2(prob / binWidth)
    return H

def regularity(x, freq):
    """
    Calculate the regularity of a signal using the method described in the paper:
    Args:
        x (numpy.ndarray): Input signal.
        freq (int): Sampling frequency of the signal.
    Returns:
        float: Regularity measure of the signal.
    """
    squared_signal = x**2
    num_wts = int(freq / 2)
    wts = np.ones(num_wts) / num_wts
    smoothed_signal = np.convolve(squared_signal, wts, mode='same')
    sorted_signal = np.sort(smoothed_signal)[::-1]
    N = len(sorted_signal)
    u = np.arange(1, N+1)
    reg = np.sqrt(np.sum(u**2 * sorted_signal) / (np.sum(sorted_signal) * N**2 * (1/3)))
    if np.isnan(reg):
        reg = 0
        print("NaN Found Regularity")
    return reg

def average_delta_band(x, freq):
    """
    Calculate the average power in the delta band (0.5-4 Hz) using Welch's method.
    Args:
        x (numpy.ndarray): Input signal.
        freq (int): Sampling frequency of the signal.
    Returns:
        float: Average power in the delta band.
    """
    win = 5 * freq
    freqs, psd = signal.welch(x, freq, nperseg=win)
    low, high = 0.5, 4
    idx = np.logical_and(freqs >= low, freqs <= high)
    freq_res = freqs[1] - freqs[0]
    delta_power = simps(psd[idx], dx=freq_res)
    if np.isnan(delta_power):
        delta_power = 0
        print("NaN Found Delta Power")
    return delta_power

def average_theta_band(x, freq):
    """
    Calculate the average power in the theta band (4-7 Hz) using Welch's method.
    Args:
        x (numpy.ndarray): Input signal.
        freq (int): Sampling frequency of the signal.
    Returns:
        float: Average power in the theta band.
    """
    win = 5 * freq
    freqs, psd = signal.welch(x, freq, nperseg=win)
    low, high = 4, 7
    idx = np.logical_and(freqs >= low, freqs <= high)
    freq_res = freqs[1] - freqs[0]
    theta_power = simps(psd[idx], dx=freq_res)
    if np.isnan(theta_power):
        theta_power = 0
        print("NaN Found Theta Power")
    return theta_power

def average_alpha_band(x, freq):
    """
    Calculate the average power in the alpha band (8-12 Hz) using Welch's method.
    Args:
        x (numpy.ndarray): Input signal.
        freq (int): Sampling frequency of the signal.
    Returns:
        float: Average power in the alpha band.
    """
    win = 5 * freq
    freqs, psd = signal.welch(x, freq, nperseg=win)
    low, high = 8, 12
    idx = np.logical_and(freqs >= low, freqs <= high)
    freq_res = freqs[1] - freqs[0]
    alpha_power = simps(psd[idx], dx=freq_res)
    if np.isnan(alpha_power):
        alpha_power = 0
        print("NaN Found Alpha Power")
    return alpha_power

def average_beta_band(x, freq):
    """
    Calculate the average power in the beta band (12-30 Hz) using Welch's method.
    Args:
        x (numpy.ndarray): Input signal.
        freq (int): Sampling frequency of the signal.
    Returns:
        float: Average power in the beta band.
    """
    win = 5 * freq
    freqs, psd = signal.welch(x, freq, nperseg=win)
    low, high = 12, 30
    idx = np.logical_and(freqs >= low, freqs <= high)
    freq_res = freqs[1] - freqs[0]
    beta_power = simps(psd[idx], dx=freq_res)
    if np.isnan(beta_power):
        beta_power = 0
        print("NaN Found Beta Power")
    return beta_power

def alpha_delta_ratio(alpha, delta):
    """
    Calculate the ratio of alpha to delta power.
    Args:
        alpha (float): Alpha power.
        delta (float): Delta power.
    Returns:
        float: Ratio of alpha to delta power.
    """
    ratio = alpha / delta if delta != 0 else 0
    if np.isnan(ratio):
        ratio = 0
        print("NaN Found Alpha-Delta Ratio")
    return ratio

def spikeFreq(eegData, duration, minNumSamples=7, stdAway=3):
    """
    Calculate the spike frequency for each channel and epoch.
    Args:
        eegData (numpy.ndarray): EEG data of shape (channels, samples, epochs).
        duration (int): Duration of each epoch in seconds.
        minNumSamples (int): Minimum number of samples for a spike.
        stdAway (int): Number of standard deviations away from the mean to consider a spike.
    Returns:
        numpy.ndarray: Spike frequency for each channel and epoch.
    """
    n_channels = eegData.shape[0]
    n_epochs = eegData.shape[2]
    H = np.zeros((n_channels, n_epochs))
    for chan in range(n_channels):
        for epoch in range(n_epochs):
            data_epoch = eegData[chan, :, epoch]
            mean = np.mean(data_epoch)
            std_val = np.std(data_epoch)
            peaks, _ = signal.find_peaks(np.abs(data_epoch - mean), height=3*std_val, width=minNumSamples)
            H[chan, epoch] = len(peaks)
    return H / duration

def burst_supression_detection(x, fs, suppression_threshold=10):
    """
    Detects burst suppression in EEG data.
    Args:
        x (numpy.ndarray): EEG data of shape (channels, samples).
        fs (int): Sampling frequency of the signal.
        suppression_threshold (int): Threshold for suppression detection.
    Returns:
        tuple: A tuple containing two lists: bursts and suppressions.
    """
    e = np.abs(signal.hilbert(x, axis=1))
    # Smooth with a moving average filter
    ME = np.array([np.convolve(el, np.ones(int(fs/4))/(fs/4), mode='same') for el in e.tolist()])
    z = (ME < suppression_threshold)
    z = fcnRemoveShortEvents(z, fs/2)
    b = fcnRemoveShortEvents(1 - z, fs/2)
    z = 1 - b
    # Find transitions
    went_high = [np.where(np.diff(chD) > 0)[0].tolist() for chD in z.tolist()]
    went_low = [np.where(np.diff(chD) < 0)[0].tolist() for chD in z.tolist()]
    bursts = get_intervals(went_high, went_low, endIdx=x.shape[1])
    supressions = get_intervals(went_low, went_high, endIdx=x.shape[1])
    return bursts, supressions

def fcnRemoveShortEvents(z, n):
    """
    Remove short events from the binary array.
    Args:
        z (numpy.ndarray): Binary array of shape (channels, samples).
        n (int): Minimum length of events to keep.
    Returns:
        numpy.ndarray: Modified binary array with short events removed.
    """
    for chan in range(z.shape[0]):
        ct = 0
        i0 = 1
        i1 = 1 
        for i in range(2, z.shape[1]):
            if z[chan, i] == z[chan, i-1]:
                ct += 1
                i1 = i
            else:
                if ct < n:
                    z[chan, i0:i1] = 0
                    z[chan, i1] = 0
                ct = 0
                i0 = i
                i1 = i
        if z[chan, 0] == 1 and z[chan, 1] == 0:
            z[chan, 0] = 0
    return z

def get_intervals(A, B, endIdx):
    """
    Get intervals from two lists of indices.
    Args:
        A (list): List of lists containing indices for the first list.
        B (list): List of lists containing indices for the second list.
        endIdx (int): End index for the intervals.
    Returns:
        list: List of intervals for each channel.
    """
    intervals = []
    for ii, A_idx_lst in enumerate(A):
        B_idx_lst = [bisect.bisect_left(B[ii], idx) for idx in A_idx_lst]
        chan_intervals = []
        for jj, idx_l in enumerate(B_idx_lst):
            if idx_l == len(B[ii]):
                chan_intervals.append((A_idx_lst[jj], endIdx))
            else:
                chan_intervals.append((A_idx_lst[jj], B[ii][idx_l]))
        intervals.append(chan_intervals)
    return intervals

def numSuppressions(eegData, fs, num_points, suppression_threshold=10):
    """
    Calculate the number of suppressions in EEG data.
    Args:
        eegData (numpy.ndarray): EEG data of shape (channels, samples, epochs).
        fs (int): Sampling frequency of the signal.
        num_points (int): Number of points to consider for suppression detection.
        suppression_threshold (int): Threshold for suppression detection.
    Returns:
        numpy.ndarray: Number of suppressions for each channel and epoch.
    """
    # Process each epoch
    bursts_all = []
    suppressions_all = []
    n_epochs = eegData.shape[2]
    for epoch in range(n_epochs):
        bursts, suppressions = burst_supression_detection(eegData[:, :, epoch], fs, suppression_threshold)
        bursts_all.append(bursts)
        suppressions_all.append(suppressions)
    n_channels = eegData.shape[0]
    numSupprs_res = np.zeros((n_channels, n_epochs))
    for chan in range(n_channels):
        for epoch in range(n_epochs):
            if chan < len(suppressions_all[epoch]):
                numSupprs_res[chan, epoch] = len(suppressions_all[epoch][chan])
            else:
                numSupprs_res[chan, epoch] = 0
    return numSupprs_res / num_points


def process_file(folder_path, file_):
    """
    Process a single EDF file to extract features.
    Args:
        folder_path (str): Path to the folder containing the EDF files.
        file_ (str): Name of the EDF file to process.
    Returns:
        tuple: A tuple containing the file name and a list of features for each channel.
    """
    try:
        raw = mne.io.read_raw_edf(os.path.join(folder_path, file_), preload=True, verbose=False)
        data = raw.get_data() * 1000000  # Convert to microVolts
        sfreq = raw.info['sfreq']

        # Define chunk size (5 minutes)
        chunk_duration = int(5 * 60 * sfreq)

        # Chunk the data for each channel
        channels_chunked = []
        for i in range(data.shape[0]):
            cur_data = data[i]
            channel_chunks = []
            for j in range(0, data.shape[1] - chunk_duration + 1, chunk_duration):
                channel_chunks.append(cur_data[j:j+chunk_duration])
            channels_chunked.append(channel_chunks)

        # Convert to NumPy array with shape: (channels, epochs, samples)
        chunked_array = np.array(channels_chunked)
        chunked_array = np.transpose(chunked_array, (0, 2, 1))
        
        # Compute features
        entropy = shannonEntropy_numba(chunked_array, bin_min, bin_max, binWidth)
        spike_freq = spikeFreq(chunked_array, duration=5)
        bs_ratio = numSuppressions(chunked_array, sfreq, num_points=30000)

        # Compute additional features chunk-wise
        n_channels = len(channels_chunked)
        n_epochs = len(channels_chunked[0])
        regs = [[] for _ in range(n_channels)]
        deltas = [[] for _ in range(n_channels)]
        thetas = [[] for _ in range(n_channels)]
        alphas = [[] for _ in range(n_channels)]
        betas = [[] for _ in range(n_channels)]
        ad_ratios = [[] for _ in range(n_channels)]

        for i in range(n_channels):
            for j in range(n_epochs):
                cur_data = channels_chunked[i][j]
                reg = regularity(cur_data, sfreq)
                delta_power = average_delta_band(cur_data, sfreq)
                theta_power = average_theta_band(cur_data, sfreq)
                alpha_power = average_alpha_band(cur_data, sfreq)
                beta_power = average_beta_band(cur_data, sfreq)
                ad_ratio_val = alpha_delta_ratio(alpha_power, delta_power)

                regs[i].append(reg)
                deltas[i].append(delta_power)
                thetas[i].append(theta_power)
                alphas[i].append(alpha_power)
                betas[i].append(beta_power)
                ad_ratios[i].append(ad_ratio_val)

        # Collate features for each channel and epoch
        features = []
        for i in range(n_channels):
            channel_features = []
            for j in range(n_epochs):
                channel_features.append([
                    entropy[i][j],
                    regs[i][j],
                    deltas[i][j],
                    thetas[i][j],
                    alphas[i][j],
                    betas[i][j],
                    ad_ratios[i][j],
                    spike_freq[i][j],
                    bs_ratio[i][j]
                ])
            features.append(channel_features)
        return file_, features
    except Exception as e:
        print(f"Error processing {file_}: {e}")
        return file_, None