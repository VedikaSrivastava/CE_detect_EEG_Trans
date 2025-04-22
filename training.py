# Importing required libraries
import random
import os
import numpy as np
import pickle
import pandas as pd
import mne
from tqdm import tqdm
import torch
import torch.nn as nn
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from torch.autograd import Variable
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.preprocessing import StandardScaler
import copy
import csv


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

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

# Defining the dataset class
class EEGDataset(Dataset):
    def __init__(self, input_data, output_labels):
        self.input_data = input_data
        self.output_labels = output_labels

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        inputs = self.input_data[idx]
        label = self.output_labels[idx]
        
        return inputs, label

# Function to normalize the data
def normalize_data(train_data, val_data, test_data):
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data.reshape(-1, train_data.shape[-1])).reshape(train_data.shape)
    val_data = scaler.transform(val_data.reshape(-1, val_data.shape[-1])).reshape(val_data.shape)
    test_data = scaler.transform(test_data.reshape(-1, test_data.shape[-1])).reshape(test_data.shape)
    return train_data, val_data, test_data

# Defining the model
class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, ff_dim, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, ff_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, 4*12, ff_dim)) 
        encoder_layer = nn.TransformerEncoderLayer(d_model=ff_dim, nhead=num_heads, dim_feedforward=ff_dim, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(ff_dim, 1)

    def forward(self, x):
        batch_size, num_channels, seq_len = x.shape
        x = x.view(batch_size, 4*12, 18 * 9)
        x = self.embedding(x)
        x += self.positional_encoding[:, :x.size(1), :]
        x = x.permute(1, 0, 2)
        output = self.transformer_encoder(x)
        output = output.permute(1, 0, 2)
        output = output.mean(dim=1)
        output = self.fc_out(output)
        return output


# Defining a training function
def eeg_train_model(model, dataloaders, criterion, optimizer, num_epochs):
    train_loss, val_loss, train_accuracy, val_accuracy = [], [], [], []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}:')

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            corrects = 0
            all_preds = []
            all_labels = []

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.float().to(device)
                labels = labels.float().to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels.unsqueeze(1))

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                preds = torch.sigmoid(outputs) > 0.5
                corrects += torch.sum(preds == labels.unsqueeze(1))

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = corrects.double() / len(dataloaders[phase].dataset)

            if phase == 'train':
                train_loss.append(epoch_loss)
                train_accuracy.append(epoch_acc.cpu().numpy())
            
            elif phase == 'val':
                val_loss.append(epoch_loss)
                val_accuracy.append(epoch_acc.cpu().numpy())

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

            if phase == 'val':
                print(classification_report(all_labels, all_preds))
                print()
    print()
    print('Best val loss: {:4f}'.format(best_loss))
    model.load_state_dict(best_model_wts)
    
    return model, train_loss, train_accuracy, val_loss, val_accuracy

def process_items(items):
    """
    Converts items (key, [data, label]) into inputs and labels.
    Here, each sample's data is reshaped and each channel's data is converted to a torch tensor.
    """
    inputs = []
    labels = []
    for key, value in items:
        data, label = value
        data_arr = np.array(data).reshape(np.array(data).shape[0], -1)
        # Convert each row (e.g., channel) into a tensor.
        data_list = [torch.tensor(data_arr[j]) for j in range(data_arr.shape[0])]
        inputs.append(np.array(data_list))
        labels.append(torch.tensor(label))
    return inputs, labels

# Initialize variables to track the best models
best_transformer_accuracy = 0
best_transformer_model = None
orig_features = "<dict of original data features>"
aug_features = "<dict of augmented data features>"
for fold in tqdm(range(0, 30), desc="Training and Testing folds"):
    # Work on copies so as not to disturb the original dictionaries
    orig_dict = copy.deepcopy(orig_features)
    aug_dict = copy.deepcopy(aug_features)
    
    # ====================================================
    # STEP 1: Create Test Set (from original features only)
    # ====================================================
    num_test = 20  # total number of test samples
    num_per_class = num_test // 2
    
    # Separate the original keys by class.
    test_no_ce_candidates = [k for k in orig_dict if orig_dict[k][1] == 1]  # label 1 (No CE)
    test_ce_candidates   = [k for k in orig_dict if orig_dict[k][1] == 0]  # label 0 (CE)
    
    # Randomly sample equal numbers for each class.
    test_no_ce_sample = random.sample(test_no_ce_candidates, num_per_class)
    test_ce_sample    = random.sample(test_ce_candidates, num_per_class)
    test_keys = test_no_ce_sample + test_ce_sample
    
    # If an odd number is required, pick one extra at random.
    if num_test % 2 != 0:
        extra_key = random.choice(test_no_ce_candidates + test_ce_candidates)
        test_keys.append(extra_key)
    
    # Create test_dict and remove test samples from the original dictionary.
    test_dict = {k: orig_dict[k] for k in test_keys}
    for k in test_keys:
        del orig_dict[k]

    # ====================================================
    # STEP 2: Build the Training/Validation Set by Adding Augmented Samples
    # ====================================================
    # The remaining orig_dict forms the base of the train/validation set.
    train_keys = list(orig_dict.keys())
    train_no_ce = [k for k in train_keys if orig_dict[k][1] == 1]
    train_ce    = [k for k in train_keys if orig_dict[k][1] == 0]
    n_no_ce = len(train_no_ce)
    n_ce    = len(train_ce)
    
    # We want exactly 100 augmented samples added in total.
    total_augmented_samples = 100
    a_no_ce = int(round((total_augmented_samples + (n_ce - n_no_ce)) / 2))
    a_ce    = total_augmented_samples - a_no_ce
    a_no_ce = max(a_no_ce, 0)
    a_ce    = max(a_ce, 0)
    
    # Group augmented samples by original file so that each original file gives at most one augmented sample.
    aug_no_ce_group = {}
    aug_ce_group = {}
    
    for k in aug_dict:
        # k is of the form "recordID_method.edf"
        orig_filename = k.split("_")[0] + ".edf"
        label = aug_dict[k][1]
        if label == 1:
            aug_no_ce_group.setdefault(orig_filename, []).append(k)
        else:
            aug_ce_group.setdefault(orig_filename, []).append(k)
    
    # Now that the groups are built, randomly pick one augmented sample for each original file.
    unique_aug_no_ce = [random.choice(v) for v in aug_no_ce_group.values()]
    unique_aug_ce    = [random.choice(v) for v in aug_ce_group.values()]
    
    # Randomly select the desired number of augmented samples from these unique ones.
    if len(unique_aug_no_ce) > a_no_ce:
        selected_aug_no_ce = random.sample(unique_aug_no_ce, a_no_ce)
    else:
        selected_aug_no_ce = unique_aug_no_ce
    
    if len(unique_aug_ce) > a_ce:
        selected_aug_ce = random.sample(unique_aug_ce, a_ce)
    else:
        selected_aug_ce = unique_aug_ce
    
    selected_aug_keys = selected_aug_no_ce + selected_aug_ce

    # Optionally, count augmented samples per method per class:
    aug_method_counts = {}
    for k in selected_aug_keys:
        try:
            method = k.split('_')[1].split('.')[0]
        except IndexError:
            method = "unknown"
        label = aug_dict[k][1]
        if method not in aug_method_counts:
            aug_method_counts[method] = {"No_CE": 0, "CE": 0}
        if label == 1:
            aug_method_counts[method]["No_CE"] += 1
        else:
            aug_method_counts[method]["CE"] += 1
    # Convert counts to a string for CSV logging.
    aug_method_counts_str = str(aug_method_counts)
    
    # Combine the remaining original train samples with the selected augmented samples.
    train_val_dict = copy.deepcopy(orig_dict)
    for k in selected_aug_keys:
        train_val_dict[k] = aug_dict[k]
    
    # ====================================================
    # STEP 3: Shuffle and Split Train/Val Set
    # ====================================================
    keys = list(train_val_dict.keys())
    labels = [train_val_dict[k][1] for k in keys]
    
    # Use train_test_split with stratify:
    train_keys, val_keys = train_test_split(keys, test_size=0.10)
    
    # Create train_items and val_items from the keys:
    train_items = [(k, train_val_dict[k]) for k in train_keys]
    val_items   = [(k, train_val_dict[k]) for k in val_keys]
    
    train_inputs, train_labels = process_items(train_items)
    val_inputs, val_labels     = process_items(val_items)
    test_inputs, test_labels   = process_items(list(test_dict.items()))
    
    # ====================================================
    # STEP 4: Print Class Distributions and Log to CSV
    # ====================================================
    train_no_ce_count = sum(1 for label in train_labels if label == 1)
    train_ce_count    = sum(1 for label in train_labels if label == 0)
    val_no_ce_count   = sum(1 for label in val_labels if label == 1)
    val_ce_count      = sum(1 for label in val_labels if label == 0)
    test_no_ce_count  = sum(1 for label in test_labels if label == 1)
    test_ce_count     = sum(1 for label in test_labels if label == 0)
    
    print("Test Data Classes - No CE:", test_no_ce_count, "CE:", test_ce_count)
    print("Train Data Classes - No CE:", train_no_ce_count, "CE:", train_ce_count)
    print("Validation Data Classes - No CE:", val_no_ce_count, "CE:", val_ce_count)
    print("Augmented samples added - No CE:", len(selected_aug_no_ce), "CE:", len(selected_aug_ce))
    
    # -------------------------------------------
    # Continue with normalizing data and model training...
    train_data, val_data, test_data = normalize_data(np.array(train_inputs), np.array(val_inputs), np.array(test_inputs))
    
    # Here, ensure that you use the correct labels for your EEGDataset.
    train_data = EEGDataset(input_data=train_data, output_labels=train_labels)
    val_data = EEGDataset(input_data=val_data, output_labels=val_labels)
    test_data = EEGDataset(input_data=test_data, output_labels=test_labels)
    
    batch_size = 20
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    
    dataloaders = {'train': train_loader, 'val': val_loader}
    
    print("Training Transformer Model...")
    # Define your transformer model parameters.
    input_dim = 9 * 18
    num_heads = 6
    num_layers = 6
    ff_dim = 600
    dropout = 0.6
    num_epochs = 150
    learning_rate = 8e-3
    
    transformer_model = TransformerModel(input_dim, num_heads, num_layers, ff_dim, dropout).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(transformer_model.parameters(), lr=learning_rate)
    
    transformer_model, train_loss, train_accuracy, val_loss, val_accuracy = eeg_train_model(
        transformer_model, dataloaders, criterion, optimizer, num_epochs)