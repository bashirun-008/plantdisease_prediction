https://plantdiseaseprediction-u4my6a83d6tvmkz4ua4ale.streamlit.app/
# Plant Disease Classifier

An image classifier that identifies plant diseases from leaf photos using transfer learning.

## Overview

This project trains a deep learning model to classify plant leaf images into 15 categories (healthy vs. various diseases across multiple plant species) using the PlantVillage dataset.

## Dataset

- **Source:** [PlantVillage dataset](https://www.kaggle.com/datasets/emmarex/plantdisease) (via Kaggle API)
- **Size:** 20,638 images across 15 classes
- **Split:** 80% training (16,511 images) / 20% validation (4,127 images)

## Approach

- **Base model:** MobileNetV2 pre-trained on ImageNet, used as a frozen feature extractor
- **Custom head:** Global average pooling → dropout (0.2) → dense softmax layer (15 classes)
- **Data augmentation:** random horizontal/vertical flips, rotation, and zoom to improve generalization
- **Input size:** 224×224 RGB images
- **Optimizer:** Adam, sparse categorical crossentropy loss

## Results

Trained for 3 epochs:

| Epoch | Train Accuracy | Val Accuracy | Val Loss |
|-------|----------------|--------------|----------|
| 1 | 70.9% | 81.6% | 0.584 |
| 2 | 83.5% | 85.5% | 0.468 |
| 3 | 85.3% | **86.1%** | 0.444 |

**Final validation accuracy: 86.1%**

Validation accuracy and loss were both still improving at epoch 3, suggesting the model would likely benefit from additional training epochs.

## Tech Stack

- Python
- TensorFlow / Keras
- MobileNetV2 (transfer learning)
- Kaggle API (dataset access)

## Next Steps

- Train for more epochs (10-15) to see if validation accuracy continues improving
- Fine-tune the last few layers of MobileNetV2 (unfreeze partially) rather than keeping the whole base frozen
- Add a confusion matrix to see which classes are most often confused
- Test on external images not from the PlantVillage dataset to check real-world generalization

## How to Run

1. Get a Kaggle API token (`kaggle.json`) from your Kaggle account settings
2. Open the notebook in Google Colab
3. Upload your `kaggle.json` when prompted
4. Run all cells in order

## Author

Md Bashirun Sultana
