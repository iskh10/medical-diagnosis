# Healthcare Diagnosis with Medical Imaging

An AI-based brain MRI analysis project that combines **deep-learning classification** and **image segmentation** to identify MRI scans that may contain a brain tumour and highlight the suspected tumour region.

> **Important:** This project is for educational and research purposes only. It is not a medical device and must not be used to diagnose, treat, or rule out a medical condition. Predictions require review by qualified healthcare professionals.

## Overview

Medical imaging can support clinicians by helping analyse scans consistently and identify regions that may need closer examination. This project explores two related tasks using brain MRI images:

1. **Tumour presence classification** — a ResNet50-based transfer-learning model classifies an MRI image into one of two categories based on the dataset labels: tumour mask absent (`0`) or present (`1`).
2. **Tumour-region segmentation** — a ResUNet-style model predicts a pixel-level mask for the suspected tumour region.

The notebook also visualises MRI scans, ground-truth masks, predicted masks, and classification evaluation results.

## Features

- Brain MRI dataset inspection and visualisation.
- Image classification using a pretrained ResNet50 backbone.
- Transfer learning with a custom dense classification head.
- ResUNet-style encoder-decoder architecture with residual blocks for segmentation.
- Custom Tversky-based loss functions for segmentation.
- Predicted-versus-ground-truth mask visualisation.
- Classification metrics including accuracy, confusion matrix, and classification report.

## Project structure

```text
Healthcare-Diagnosis-with-Medical-Imaging/
├── Healthcare.ipynb          # Main notebook: data exploration, training, evaluation and prediction
├── utilities.py               # Data generator, custom losses/metrics and prediction helpers
├── Project-Architecture.png   # Project workflow diagram
├── CNN-Architecture.png       # CNN illustration
├── RESUNET.png                # ResUNet architecture illustration
├── Brain-Tumour-detected.png  # Example visualisation
├── Total-Output.png           # Example output visualisation
└── README.md
```

The dataset, pretrained weights, and generated model files may need to be obtained or created separately; they are not included in this repository archive.

## Models and workflow

### 1. Data preparation

The notebook expects a CSV file named `data_mask.csv` containing image and mask paths. The workflow uses the `image_path`, `mask_path`, and `mask` fields; the original notebook also references `patient_id`.

- `image_path`: path to the MRI image.
- `mask_path`: path to its corresponding segmentation mask.
- `mask`: binary label indicating whether a tumour mask is present (`0` or `1`).
- `patient_id`: patient identifier used in the original dataset metadata.

Images are resized to **256 × 256** for model input. The classification pipeline scales pixel values to the range 0–1. The notebook splits data into training and testing subsets and creates a validation subset from the training data.

**Dataset note:** The original notebook uses Google Drive paths and assumes the dataset and CSV are already available. Update paths and verify the CSV column names before running it. Keep patient-level separation between training and evaluation sets where possible to reduce data leakage.

### 2. Classification with ResNet50

The classification model uses ResNet50 pretrained on ImageNet as a feature extractor, followed by average pooling, fully connected layers, dropout, and a two-class softmax output. It is compiled with categorical cross-entropy and the Adam optimizer.

The notebook evaluates predictions using:
- Accuracy
- Confusion matrix
- Precision, recall, and F1-score via a classification report

### 3. Segmentation with ResUNet

The segmentation model combines a U-Net-style encoder-decoder with residual blocks and skip connections. It produces a pixel-level mask using a sigmoid output layer. The notebook uses custom Tversky-related loss/metric functions from `utilities.py` to address foreground/background imbalance.

## Requirements

The notebook was written for a Google Colab-style environment. A compatible Python environment with the following packages is needed:

- Python
- TensorFlow / Keras
- NumPy
- pandas
- Matplotlib
- Seaborn
- OpenCV (`opencv-python`)
- scikit-image
- scikit-learn
- Plotly
- `keras-preprocessing` (for the notebook's `ImageDataGenerator` import)
- Jupyter Notebook or Google Colab

Package compatibility can depend on the TensorFlow/Keras version. The notebook uses some older APIs and may require updates for newer versions. It also imports `google.colab`, which is available in Google Colab but not in a standard local Python session.

## How to run

### Option A — Google Colab

1. Upload `Healthcare.ipynb` and `utilities.py` to Google Colab, or open the notebook from your repository.
2. Upload or mount the dataset in Google Drive.
3. Ensure `data_mask.csv` and all referenced MRI/mask files are available.
4. In the notebook, update the Google Drive mount and working-directory cells to match your own folder structure.
5. Install any missing dependencies in the notebook environment.
6. Run the cells in order to explore the dataset, train the classification model, train the segmentation model, and view the outputs.

### Option B — Local Jupyter environment

1. Extract or clone the project files.
2. Create and activate a virtual environment.
3. Install the required packages.
4. Remove or replace the `google.colab` imports and Drive-mount commands.
5. Update all dataset paths to valid local paths.
6. Run `Healthcare.ipynb` from top to bottom.

Example setup commands:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install numpy pandas matplotlib seaborn opencv-python scikit-image scikit-learn plotly tensorflow keras-preprocessing jupyter
```

If TensorFlow installation fails, select a TensorFlow version compatible with your Python version and hardware.

## Model files and checkpoints

The notebook saves model architecture and weights during training. Filenames used in the notebook include:

- `classifier-resnet-model.json`
- `classifier-resnet-weights.hdf5`
- `ResUNet-model.json`
- `ResUNet-weights.hdf5`

Later notebook cells also attempt to load files such as `resnet-50-MRI.json`, `weights.hdf5`, `ResUNet-MRI.json`, and `weights_seg.hdf5`. These files must exist at the specified paths or those cells will fail. Update the filenames to match the checkpoints you actually trained or downloaded. Do not assume pretrained project-specific weights are included.

## Results and interpretation

The notebook generates classification evaluation outputs and visual comparisons of:

- Original MRI images
- Ground-truth tumour masks
- AI-predicted masks
- MRI images with ground-truth or predicted regions overlaid

Actual accuracy and segmentation quality depend on dataset quality, train/test separation, preprocessing, model configuration, and training duration. The notebook's sample training cells use `epochs=1`; this is a demonstration setting and is not sufficient by itself to establish clinical performance. Report measured results from a properly held-out test set rather than assuming a particular accuracy.

## Limitations

- The project is a prototype for learning and experimentation, not a clinically validated diagnostic system.
- Model performance may be affected by scanner differences, image quality, dataset imbalance, and differences between hospitals or populations.
- The original notebook includes environment-specific paths and expects data/checkpoint files that may not be bundled.
- The training and evaluation code may need debugging or adaptation for current TensorFlow/Keras versions.
- A robust evaluation should include patient-level splits, suitable segmentation metrics (for example, Dice score and IoU), and external validation.

## Future improvements

- Make dataset paths configurable instead of hard-coding Google Drive directories.
- Pin tested package versions in a `requirements.txt` file.
- Add reproducible random seeds and patient-level train/validation/test splits.
- Evaluate classification with ROC-AUC, sensitivity, specificity, precision, recall, and F1-score.
- Evaluate segmentation with Dice score and Intersection over Union (IoU).
- Add model explainability and uncertainty estimation.
- Validate on independent datasets and obtain clinical expert review before considering any real-world use.

## Acknowledgements

This project is based on the code and architecture concepts contained in the provided repository, including ResNet50 transfer learning and ResUNet-style segmentation. Please retain and verify the original dataset and model attribution/licensing before redistributing or using them.

## License

No explicit license was identified in the supplied project files. Check the original repository and dataset terms before reuse or redistribution.
