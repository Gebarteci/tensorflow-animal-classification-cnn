# Advanced Animal Image Classification: Custom CNN & Transfer Learning

## Overview

This project documents an extensive journey into 10-class animal image classification using TensorFlow and Keras. It showcases an iterative development process, starting with a custom-built Convolutional Neural Network (CNN) that was progressively optimized. Following this, the project explores Transfer Learning with the `VGG16` architecture to compare approaches and potentially further enhance performance.

This repository serves as a result of experiments with data augmentation, regularization techniques, optimizer selection (Adam vs. AdamW), learning rate scheduling, and the impact of different model architectures.

**Dataset:**
The project uses the **"Animals-10" dataset from Kaggle**, which consists of 10 animal classes: cane (dog), cavallo (horse), elefante (elephant), farfalla (butterfly), gallina (chicken), gatto (cat), mucca (cow), pecora (sheep), ragno (spider), scoiattolo (squirrel).
The `Tensorflow.ipynb` notebook includes cells to download and prepare this dataset using the Kaggle API.
Dataset Link: [https://www.kaggle.com/datasets/alessiocorrado99/animals10](https://www.kaggle.com/datasets/alessiocorrado99/animals10)

## Project Structure

The project is primarily contained within the `Tensorflow.ipynb` Jupyter Notebook, which is structured as follows:
1.  Setup (Kaggle API, dataset download).
2.  Data Preparation and Augmentation definition for the Custom CNN.
3.  **Phase 1: Custom CNN Development** - Building, training, and iteratively refining a custom CNN architecture.
4.  **Phase 2: Transfer Learning with VGG16** - Implementing VGG16 for feature extraction and fine-tuning.

## Phase 1: Custom CNN Development (Peak Performance ~81%)

A custom CNN was built from scratch and optimized through various experiments.

* **Best Achieved Validation Accuracy:** ~81% (0.8082)
* **Key Optimizer:** AdamW with `learning_rate=0.0005` (using `ReduceLROnPlateau`) and `weight_decay=1e-4`.
* **Architecture Highlights:**
    * Input Rescaling (1./255) for images of size (128, 128, 3).
    * Three Convolutional Blocks: (Conv2D + BatchNormalization + ReLU + SpatialDropout2D(0.1) + MaxPooling2D). Filters: 32, 64, 128.
    * GlobalAveragePooling2D.
    * Dense block: Dense(256) + BatchNormalization + ReLU + Dropout(0.5).
    * Output layer: Dense(10, activation='softmax').
    * *Note: L2 kernel regularizers were removed from layers when using AdamW.*
* **Data Augmentation Used:**
    * `RandomFlip("horizontal")`
    * `RandomContrast(0.1)`
    * `RandomBrightness(0.1)`
* **Training:** ~80 epochs, with `EarlyStopping` (patience 10) restoring weights from the best epoch (epoch 78).
* **Key Learning:** This phase demonstrated excellent generalization, with training and validation curves tracking very closely, especially with AdamW.

### Custom CNN Performance (Best Model - ~81% Accuracy)

* **Validation Accuracy:** ~81% (0.8082)
* **Validation Loss:** ~0.584
* **Macro Average F1-score:** 0.79
* **Weighted Average F1-score:** 0.81

Validation Set Classification Report:

                precision   recall   f1-score   support            
        cane       0.76      0.80      0.78       947
     cavallo       0.81      0.75      0.78       522
    elefante       0.77      0.81      0.79       273
    farfalla       0.93      0.84      0.88       429
     gallina       0.85      0.88      0.87       593
       gatto       0.74      0.67      0.71       331
       mucca       0.76      0.69      0.72       399
      pecora       0.79      0.74      0.76       382
       ragno       0.86      0.93      0.89       979
    scoiattolo     0.74      0.75      0.75       380

    accuracy                           0.81      5235
    macro avg      0.80      0.79      0.79      5235
    weighted avg   0.81      0.81      0.81      5235

### Confusion Matrix 

| True \ Predicted | `cane` | `cavallo` | `elefante` | `farfalla` | `gallina` | `gatto` | `mucca` | `pecora` | `ragno` | `scoiattolo` |
|---|---|---|---|---|---|---|---|---|---|---|
| **`cane`** | 759 | 29 | 13 | 4 | 25 | 33 | 18 | 19 | 20 | 27 |
| **`cavallo`** | 51 | 394 | 12 | 2 | 11 | 3 | 31 | 10 | 5 | 3 |
| **`elefante`** | 17 | 10 | 220 | 1 | 3 | 5 | 3 | 6 | 7 | 1 |
| **`farfalla`** | 4 | 0 | 1 | 360 | 13 | 4 | 0 | 1 | 37 | 9 |
| **`gallina`** | 22 | 2 | 2 | 1 | 520 | 3 | 6 | 3 | 18 | 16 |
| **`gatto`** | 56 | 1 | 2 | 2 | 8 | 223 | 2 | 4 | 13 | 20 |
| **`mucca`** | 39 | 36 | 13 | 1 | 7 | 0 | 274 | 25 | 2 | 2 |
| **`pecora`** | 22 | 12 | 13 | 0 | 7 | 9 | 25 | 281 | 7 | 6 |
| **`ragno`** | 10 | 3 | 3 | 15 | 8 | 9 | 0 | 1 | 915 | 15 |
| **`scoiattolo`** | 21 | 1 | 5 | 1 | 7 | 12 | 1 | 4 | 43 | 285 |

## Phase 2: Transfer Learning with VGG16 

To explore leveraging pre-trained models, `VGG16` was implemented.

* **Base Model:** `VGG16`
    * Input Shape: `(128, 128, 3)` (using `IMG_HEIGHT`, `IMG_WIDTH` defined earlier in the notebook).
    * Weights: ImageNet.
    * `include_top=False`.
* **Strategy:**
    1.  **Feature Extraction:** The VGG16 base model was initially frozen (`base_model.trainable = False`).
    2.  **Fine-Tuning:** Subsequently, the base model was made trainable (`base_model.trainable = True`), and layers from the 100th layer onwards were unfrozen (`fine_tune_at = 100`).
* **Custom Head:**
    * `GlobalAveragePooling2D()`
    * `Dense(256, activation='relu')`
    * `Dropout(0.5)`
    * `Dense(N_CLASSES, activation='softmax')`
* **Input Data:**
    * The `train_dataset` and `validation_dataset` prepared for the custom CNN (including image resizing to (128,128) and existing data augmentations) were used.
    * **Data Augmentation on `train_dataset` (inherited from custom CNN setup):**
        * `RandomFlip("horizontal")`
        * `RandomContrast(0.1)`
        * `RandomBrightness(0.1)`
* **Optimizer & Training Stages:**
    1.  **Initial Head Training (Base Frozen):**
        * Optimizer: Adam with `learning_rate=0.001`.
        * Epochs: 10.
    2.  **Fine-Tuning (Base Partially Unfrozen):**
        * Optimizer: Adam with a lower `learning_rate=1e-5`.
        * Epochs: Continued up to a total of 20 epochs (i.e., `20 - (last_epoch_of_initial_training)` additional epochs).

### Classification Report (Transfer Learning Model - VGG16)


              precision    recall  f1-score   support              
        cane       0.84      0.89      0.86       947
     cavallo       0.85      0.85      0.85       522
    elefante       0.93      0.89      0.91       273
    farfalla       0.95      0.90      0.93       429
     gallina       0.92      0.91      0.91       593
       gatto       0.86      0.77      0.82       331
       mucca       0.83      0.83      0.83       399
      pecora       0.83      0.80      0.81       382
       ragno       0.93      0.97      0.95       979
     scoiattolo    0.86      0.82      0.84       380

    accuracy                           0.88      5235
    macro avg      0.88      0.86      0.87      5235
    weighted avg   0.88      0.88      0.88      5235


### Confusion Matrix 

| True \ Predicted | `cane` | `cavallo` | `elefante` | `farfalla` | `gallina` | `gatto` | `mucca` | `pecora` | `ragno` | `scoiattolo` |
|---|---|---|---|---|---|---|---|---|---|---|
| **`cane`** | 847 | 30 | 3 | 2 | 16 | 17 | 11 | 7 | 6 | 8 |
| **`cavallo`** | 31 | 444 | 2 | 0 | 2 | 1 | 30 | 10 | 2 | 0 |
| **`elefante`** | 6 | 10 | 244 | 0 | 0 | 2 | 2 | 6 | 1 | 2 |
| **`farfalla`** | 1 | 2 | 0 | 388 | 5 | 0 | 0 | 0 | 29 | 4 |
| **`gallina`** | 21 | 1 | 1 | 2 | 538 | 2 | 2 | 10 | 7 | 9 |
| **`gatto`** | 45 | 2 | 0 | 2 | 5 | 256 | 1 | 3 | 5 | 12 |
| **`mucca`** | 17 | 21 | 1 | 0 | 3 | 0 | 333 | 23 | 0 | 1 |
| **`pecora`** | 25 | 8 | 9 | 0 | 4 | 2 | 22 | 306 | 2 | 4 |
| **`ragno`** | 3 | 2 | 0 | 15 | 3 | 1 | 0 | 1 | 945 | 9 |
| **`scoiattolo`** | 16 | 3 | 2 | 0 | 10 | 16 | 0 | 3 | 17 | 313 |

 ## Technologies Used

* Python 3.x
* TensorFlow & Keras API (including `tf.keras.applications.VGG16`)
* Scikit-learn (for classification reports, confusion matrices, class weight calculation if used)
* NumPy
* Matplotlib (for plotting)
* OS, json (for Kaggle API setup and environment management, as seen in the notebook)

## Setup

## Setup for Google Colab

This project is designed to be run in a Google Colab environment.

1.  **Open in Colab:**
    * Upload the `Tensorflow.ipynb` notebook to your Google Drive.
    * Open it with Google Colaboratory.
    * Alternatively, if your project is on GitHub, you can open it directly in Colab by replacing `github.com` with `colab.research.google.com/github/` in your repository's notebook URL.
        * Example: `https://colab.research.google.com/github/[YourUsername]/[Your-Repository-Name]/blob/main/Tensorflow.ipynb`

2.  **Enable GPU Acceleration:**
    * In Colab, go to **Runtime** -> **Change runtime type**.
    * Select **GPU** from the "Hardware accelerator" dropdown menu and click **Save**. This will significantly speed up model training.
    * It will work in free tier **T4 GPU**
      
3.  **Kaggle API Setup (for Dataset Download):**
    * The `Tensorflow.ipynb` notebook (Cells 1 & 2) includes steps to download the "Animals-10" dataset directly from Kaggle using the Kaggle API.
    * **To make this work in Colab:**
        1.  Go to your Kaggle account page (`https://www.kaggle.com/[YourKaggleUsername]/account`).
        2.  Click on "Create New API Token." This will download a `kaggle.json` file.
        3.  In the Colab notebook, when you run the first code cell (Cell 1, which sets up Kaggle credentials), it will prompt you to upload this `kaggle.json` file. Alternatively, the cell has lines to manually input your username and key.  **You must replace stars "*" in `KAGGLE_USERNAME = "*****"` and `KAGGLE_KEY = "***************"` with your own Kaggle API credentials.**
    * After setting up the credentials, the subsequent cells will download and unzip the dataset into your Colab environment.

4.  **Dependencies:**
    * Google Colab comes with TensorFlow, Scikit-learn, Matplotlib, and NumPy pre-installed.

5.  **Dataset Path:**
    * The notebook is configured to download and unzip the dataset to a specific path (usually `./animals10/raw-img/` relative to the Colab environment's root after download). The `DATA_DIR` variable in the notebook should already point to this.
    * If you manually upload the dataset to your Google Drive and mount your Drive, you would need to adjust the `DATA_DIR` path accordingly. However, the notebook is set up for direct Kaggle download.
    * The `Tensorflow.ipynb` notebook (Cells 1 & 2) handles Kaggle API setup for downloading the "Animals-10" dataset.
  
## Usage

The entire project is contained within the **`Tensorflow.ipynb`** Jupyter Notebook.
1.  Ensure your Kaggle API credentials are correctly set in Cell 1 of the notebook.
2.  Run the notebook cells sequentially.
    * **Cells 1-2:** Kaggle API and dataset download/setup.
    * **Cell 3 Custom CNN Section (following data prep):** Contains the development and training of the custom CNN (best run achieved ~81% accuracy with AdamW).
    * **Cell 4 Transfer Learning with VGG16:** Implements VGG16 feature extraction and fine-tuning.

