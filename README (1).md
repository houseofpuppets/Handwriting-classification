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
├── Graph(2).ipynb          # Main notebook (full pipeline)
├── Technical_Analysis_Document.pdf    # Technical analysis document
├── requirements.txt         # Python dependencies
└── README.md
```

## Dataset

The dataset consists of 441 handwritten line images (147 per class), manually
selected and labelled from the
[IAM Handwriting Database](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database).

The pre-labelled subset used in this project is available here:
https://drive.google.com/drive/folders/1461p90NsMJ3rQdsAu2tJWSIMV8NIM72k?usp=drive_link

To use it in Colab without downloading:
1. Open the link above and click "Add shortcut to Drive"
2. Mount your Drive in the notebook (first cell)
3. Update `DATASET_PATH` with the shortcut location
4. Run all cells

## Setup and Running Instructions

This project runs on **Google Colab**.

**Step 1** — Open the notebook:
> https://colab.research.google.com/drive/1mRFRhfZqEYav0hAZc48GXGZiZNFMZW31?usp=sharing

**Step 2** — Set the dataset path in the first cell:
```python
DATASET_PATH = "/content/drive/MyDrive/your_path/dataset_graph"
```

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
