# CE_detect_EEG_Trans

## Introduction
CE_detect_EEG_Trans is an EEG signal processing pipeline designed to extract robust features from raw EEG recordings, apply data augmentation, and train a transformer-based model for classification tasks. The project leverages state-of-the-art techniques in feature computation, artifact detection, and deep learning to analyze EEG data.

## Repository Structure
```
CE_detect_EEG_Trans/
├── preprocessing.py           # Performs raw data preprocessing: channel bipolar conversion, filtering, artifact detection, and chunking.
├── data_augmentation.py       # Implements augmentation strategies: noise addition, channel shuffling, and smooth time masking.
├── feature_calculation.py     # Extracts EEG features such as Shannon entropy, power bands, spike frequency, and burst suppression measures.
├── training.py                # Contains routines for dataset preparation and model training with a transformer architecture
└── README.md                 # Project documentation
```

## File Descriptions

- **feature_calculation.py:**  
  Contains optimized routines (with Numba) to compute EEG features like entropy, power in various frequency bands, and spike frequency.

- **data_augmentation.py:**  
  Provides functions to generate augmented samples by adding noise, shuffling channels, and applying smooth time masking to enhance model robustness.

- **preprocessing.py:**  
  Responsible for preparing raw EEG data. This includes converting channels to bipolar form, filtering between 0.1-40 Hz, resampling, and detecting artifacts with multiple criteria.

- **training.py:**  
  Defines a transformer-based model and the corresponding training pipeline. This file handles data normalization, train/test split, model optimization, and evaluation.

## Assumptions
- EEG data is available in EDF format with raw data provided in microvolts.
- The channel naming conventions are consistent and allow for bipolar channel creation based on a predefined list.
- Preprocessing steps (filtering, resampling, artifact detection) use fixed parameters as set in the code.
- Adequate computational resources are available for model training and preprocessing (e.g., GPU support when available).
