# Importing the required libraries
import mne
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import butter, filtfilt, hilbert
import multiprocessing
from mne import io
import pyedflib
import os


# Ignoring all warnings
import warnings
warnings.filterwarnings("ignore")


### Helper functions ###

def bipolar(channels_wanted, act_dict):
    """
    Converting single channels to bipolar channels
    Args:
        channels_wanted (list): List of bipolar channels to be created
        act_dict (dict): Dictionary containing the EEG data for each channel
    Returns:
        new_dict (dict): Dictionary containing the bipolar channels
    """
    new_dict = {}
    # Iterating through the wanted channels list
    for i in tqdm(channels_wanted):
        new_list = []
        
        # Getting individual channel names
        ch_1 = i.split("-")[0].strip()
        ch_2 = i.split("-")[1].strip()
        
        # Getting channel data from the data dictionary
        list1 = act_dict[ch_1]
        list2 = act_dict[ch_2]

        # Subtracting the second channel from the first to create bipolar channel
        for j in range(len(list1)):
            new_list.append(list1[j] - list2[j])
        new_dict[i] = new_list

    return new_dict


# These function detect artifacts in each chunk of data from the input
def high_amplitude_artifact(epoch):
    """
    High amplitude artifact detection
    """
    return np.any(np.abs(epoch) > 500)

def small_std_artifact(epoch):
    """
    Small standard deviation artifact detection
    """
    for i in range(0, len(epoch) - 200, 100):
        cur_epoch = epoch[i:i+200]
        std_vals = np.std(cur_epoch, axis = 0)
        if std_vals < 0.2:
            return True

def fast_amplitude_change_artifact(epoch):
    """
    Fast (rapid) amplitude change artifact detection
    """
    for i in range(0, len(epoch) - 20, 10):
        cur = epoch[i:i+10]
        next_ = epoch[i+10:i+20]
        cur_avg = np.mean(cur)
        next_avg = np.mean(next_)
        if np.abs(cur_avg - next_avg) > 900:
            return True

def get_artifact_percentage_with_logging(chunk, sfreq):
    """
    Calculate the percentage of flagged epochs in a chunk of EEG data.
    Args:
        chunk (ndarray): The EEG data chunk.
        sfreq (int): The sampling frequency of the EEG data.
    Returns:
        float: The percentage of flagged epochs.
    """
    global total_flagged_epochs, total_epochs, total_duration_flagged
    global high_amplitude_flagged_epochs, small_std_flagged_epochs, fast_amplitude_change_flagged_epochs

    n_windows = int(len(chunk) // (5 * sfreq))
    artifact_windows = 0

    for i in range(n_windows):
        window = chunk[i * 5 * sfreq:(i + 1) * 5 * sfreq]
        flagged = False  # Track if the current window is flagged for any artifact

        # Check for high amplitude
        if high_amplitude_artifact(window):
            high_amplitude_flagged_epochs += 1
            artifact_windows += 1
            flagged = True

        # Check for small standard deviation
        if small_std_artifact(window):
            small_std_flagged_epochs += 1
            artifact_windows += 1
            flagged = True

        # Check for fast amplitude change
        if fast_amplitude_change_artifact(window):
            fast_amplitude_change_flagged_epochs += 1
            artifact_windows += 1
            flagged = True

        # # Log details for each artifact type in debug mode
        # if flagged:
        #     print(f"Window {i}: HighAmp={high_amplitude_artifact(window)}, "
        #           f"SmallStd={small_std_artifact(window)}, "
        #           f"FastAmpChange={fast_amplitude_change_artifact(window)}")

    # Update overall metrics
    total_flagged_epochs += artifact_windows
    total_epochs += n_windows
    total_duration_flagged += artifact_windows * 5  # Each epoch is 5 seconds

    return (artifact_windows / n_windows) * 100


def prep_process(filename, file_path, output_path):
    """
    Preprocess the EEG data
    Args:
        filename (str): The name of the file to be preprocessed
        path (str): The path to the directory containing the file
    """
    raw = mne.io.read_raw_edf(f"{file_path}/{filename}", preload=True)
    duration_seconds = raw.times[-1] - raw.times[0]
    if duration_seconds < 20*3600:
        # print(f"{filename} has less than 20 hours\nDuration:{raw.times[-1] - raw.times[0]}")
        return
        
    if duration_seconds > 24*3600:
        start_time = raw.times[0]
        end_time = start_time + 24*3600
        raw = raw.crop(tmin=start_time, tmax=end_time)
        # print(f"extracted 24 hours from {filename}\nDuration:{raw.times[-1] - raw.times[0]}\n")

    # print(f"processing {filename}\n")

    # Retrieving the new EEG data (from only the first 24 hrs)
    data = raw.get_data()

    # Converting data from microVolts to Volts
    data = data * 1000000
        
    # Creating a new EEG RawArray using this new transformed data for further processing
    info = mne.create_info(ch_names=raw.ch_names, sfreq = raw.info['sfreq'], ch_types = 'eeg')
    data = np.array(data)
    raw = mne.io.RawArray(data, info)
    
    # Retrieving the channel names and signal frequency
    ch_names = raw.ch_names
    sfreq = raw.info['sfreq']
    
    # Storing each channel's values in a dictionary with the keys being the channel names
    eeg_temp_dict = {}
    for i, channel_name in enumerate(ch_names):
        eeg_temp_dict[channel_name] = data[i]
    
    # Initializing a list containing all hte bipolar channels to be retained
    channels_wanted = ['Fp1 - F7', 'F7 - T3', 'T3 - T5', 'T5 - O1', 'Fp2 - F8', 'F8 - T4', 'T4 - T6', 'T6 - O2', 'Fp1 - F3', 'F3 - C3', 'C3 - P3', 'P3 - O1', 'Fp2 - F4', 'F4 - C4', 'C4 - P4', 'P4 - O2', 'Fz - Cz', 'Cz - Pz']

    #create bipolar channels
    eeg_channelwise_data = bipolar(channels_wanted, eeg_temp_dict)

    # Storing the newly processed bipolar channels as an EEG RawArray
    info = mne.create_info(ch_names=list(eeg_channelwise_data.keys()), sfreq = raw.info['sfreq'], ch_types = 'eeg')
    data = np.array(list(eeg_channelwise_data.values()))
    raw = mne.io.RawArray(data, info)
    
    # Applying bandpass filter to extract frequencies between 0.1Hz and 40Hz
    raw = raw.filter(l_freq=0.1, h_freq=40)
    
    # Resampling the signal to 100Hz
    new_sampling_freq = 100 
    raw = raw.resample(sfreq = new_sampling_freq)

    # Storing the data in a variable
    data = raw.get_data()

    # Storing the channel names and sampling frequency
    ch_names = raw.ch_names
    sfreq = raw.info['sfreq']

    # Iterating through each channel's data and creating 5-minute chunks
    chunk_duration = int(5 * 60 * sfreq)
    channels_chunked = []
    
    for i in range(data.shape[0]):
        cur_data = data[i]
        channels_chunked.append([])
        for j in range(0, data.shape[1] - chunk_duration + 1, chunk_duration):
            channels_chunked[i].append(cur_data[j:j+chunk_duration])

    # Creating a list of lists to stored weighted chunks
    weighted_chunks = []
    
    for i in range(data.shape[0]):
        weighted_chunks.append([])
    
    # Iterating through each chunk in the manner it was mentioned in the base paper and getting weighted chunks
    weights = []
    

    # Initialize counters for metrics
    global total_flagged_epochs, total_epochs, total_duration_flagged
    global high_amplitude_flagged_epochs, small_std_flagged_epochs, fast_amplitude_change_flagged_epochs

    total_flagged_epochs = 0
    total_epochs = 0
    total_duration_flagged = 0  # in seconds
    
    # Subcategory-specific counters
    high_amplitude_flagged_epochs = 0
    small_std_flagged_epochs = 0
    fast_amplitude_change_flagged_epochs = 0
    
    # Replace the call to `get_artifact_percentage` with the updated function
    for i in tqdm(range(len(channels_chunked)), desc='get artifact info'):
        for j in range(len(channels_chunked[0])):
            chunk = channels_chunked[i][j]
            artifact_percentage = get_artifact_percentage_with_logging(chunk, int(sfreq))
            weight = 1 - (artifact_percentage / 100)
            weights.append(weight)
            weighted_chunk = chunk * weight
            weighted_chunks[i].append(weighted_chunk)
    
    # Calculate metrics in minutes
    high_amplitude_minutes = (high_amplitude_flagged_epochs * 5) / 60
    small_std_minutes = (small_std_flagged_epochs * 5) / 60
    fast_amplitude_change_minutes = (fast_amplitude_change_flagged_epochs * 5) / 60
    total_flagged_minutes = total_duration_flagged / 60
    
    # # Print summary statistics
    # print(f"Total flagged epochs: {total_flagged_epochs}")
    # print(f"Total epochs: {total_epochs}")
    # print(f"Percentage of data removed: {100 * total_flagged_epochs / total_epochs:.2f}%")
    # print(f"Total duration flagged: {total_flagged_minutes:.2f} minutes")
    # print(f"High amplitude flagged: {high_amplitude_minutes:.2f} minutes")
    # print(f"Small standard deviation flagged: {small_std_minutes:.2f} minutes")
    # print(f"Fast amplitude change flagged: {fast_amplitude_change_minutes:.2f} minutes")
    
    # Converting the weighted chunks into a list of lists from a list of list of lists
    final_weighted_obs = []
    for i in tqdm(range(len(weighted_chunks)), desc='converting weighted chunks'):
        cur_list = []
        for j in range(len(weighted_chunks[i])):
            cur_list.extend(weighted_chunks[i][j])
        final_weighted_obs.append(cur_list)

    # Saving the preprocessed EEG data into a RawArray
    info = mne.create_info(ch_names=list(eeg_channelwise_data.keys()), sfreq = raw.info['sfreq'], ch_types = 'eeg')
    data = np.array(list(final_weighted_obs))
    raw = mne.io.RawArray(data, info)

    # Converting data from Volts to microVolts
    final_weighted_obs = np.array(final_weighted_obs)
    final_weighted_obs = final_weighted_obs / 1000000

    # Saving the preprocessed EEG data into a RawArray
    info = mne.create_info(ch_names=list(eeg_channelwise_data.keys()), sfreq = raw.info['sfreq'], ch_types = 'eeg')
    data = np.array(list(final_weighted_obs))
    raw = mne.io.RawArray(data, info)

    #Exporting the EEG into an EDF file
    mne.export.export_raw(f'{output_path}/{filename}', raw, fmt='edf', overwrite=True)
    # print("File saved")