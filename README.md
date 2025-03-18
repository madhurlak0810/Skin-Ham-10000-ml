# Skin Cancer Classification using EfficientNetB0

This repository contains a Jupyter Notebook implementation for classifying skin cancer types using the HAM10000 dataset and the EfficientNetB0 model. The project is developed in Python and leverages deep learning techniques for image classification.

## Project Overview

The goal of this project is to classify skin lesions into various categories (e.g., melanoma, basal cell carcinoma) using dermoscopic images from the HAM10000 dataset. The EfficientNetB0 model, pre-trained on ImageNet, is fine-tuned for this multi-class classification task.

### Dataset
- **Source**: HAM10000 (Human Against Machine with 10000 training images) dataset from Kaggle.
- **Location**: `/kaggle/input/skin-cancer-mnist-ham10000/`
- **Metadata**: `HAM10000_metadata.csv` contains details such as lesion ID, image ID, diagnosis (dx), diagnosis type (dx_type), age, sex, and localization.

### Model
- **Architecture**: EfficientNetB0 with additional custom layers (Dropout, Dense, BatchNormalization).
- **Input Size**: Images resized to 32x32 pixels.
- **Optimizer**: Adam and RMSprop.
- **Callbacks**: EarlyStopping to prevent overfitting.

