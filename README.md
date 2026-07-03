# Handwriting Style Classification: Cursive, Printed and Mixed

A computer vision project that classifies handwritten text lines into three categories:
- **Cursive** (*corsivo*): connected letters with continuous strokes
- **Printed** (*stampatello*): isolated letters with separated strokes
- **Mixed** (*misto*): hybrid styles combining features of both

## Project Overview

The pipeline extracts handcrafted features from binary images of handwritten lines and trains a Random Forest classifier. A transfer learning experiment with MobileNetV2 is also included for comparison.

**Final results:**
- Random Forest: 64.4% accuracy (5-fold cross-validation)
- MobileNetV2 (transfer learning): 46% accuracy

## Pipeline

```
Raw image → Grayscale → Otsu binarization → Morphological opening
    → Feature extraction (component density, loop density,
       average blob width, distance variance, HOG)
    → StandardScaler normalization
    → Random Forest classifier
    → Confidence thresholding (post-processing)
```

## Repository Structure

```
handwriting-classification/
├── Graph__2_.ipynb
├── Technical_Analysis_Document.pdf
├── requirements.txt
├── README.md
├── cursive_files.txt
├── mixed_files.txt
└── printed_files.txt

```

## Dataset

The dataset consists of 441 handwritten line images (147 per class), manually
selected and labelled from the
[IAM Handwriting Database](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database).

The dataset cannot be redistributed due to licensing restrictions.

To reproduce this project:
1. Register and download the dataset from the link above
2. Create the following folder structure:

dataset_graph/
├── corsivo/
├── misto/
└── stampatello/

3. Copy the files listed in `cursive_files.txt`, `mixed_files.txt` 
   and `printed_files.txt` into the corresponding folders
4. Update `DATASET_PATH` in the first cell of the notebook

## Setup and Running Instructions

This project runs on **Google Colab**.

**Step 1** — Open the notebook:
> https://colab.research.google.com/drive/1mRFRhfZqEYav0hAZc48GXGZiZNFMZW31?usp=sharing

**Step 2** — Set the dataset path in the first cell

**Step 3** — Run all cells in order (Runtime → Run all).
For the MobileNetV2 section, enable GPU first:
Runtime → Change runtime type → T4 GPU.

## Dependencies

See `requirements.txt` for the full list. Main libraries:
- `opencv-python`
- `scikit-learn`
- `scikit-image`
- `tensorflow`
- `matplotlib`
- `seaborn`
- `numpy`

## Results Summary

| Model | Accuracy (CV) | Accuracy (test set) |
|-------|--------------|---------------------|
| Random Forest (5-fold CV) | 64.4% ± 5.3% | 69% |
| MobileNetV2 (transfer learning) | — | 46% |



## Ethical Considerations

Handwriting can function as a biometric identifier. The IAM dataset was
collected from a limited demographic group — models trained on it may not
generalize well to underrepresented writing styles.
See Section 6 of the Technical Analysis Document for details.
