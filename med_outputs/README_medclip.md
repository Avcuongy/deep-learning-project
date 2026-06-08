# MedCLIP Retrieval Model

This folder contains the exported results for the **MedCLIP / Medical CLIP** part of the Deep Learning final project.

## Task Overview

The MedCLIP model is used as a **contrastive image-text retrieval model**, not as a generative captioning model.

Pipeline:

```text
Chest X-ray image -> MedCLIP image encoder
Radiology report  -> MedCLIP text encoder
=> Contrastive learning / image-text matching
=> Evaluation with Recall@K
```

The generated captions/reports are handled by the CNN+LSTM and CNN+Transformer models.  
This MedCLIP module provides image-text matching and exported encoders for downstream use.

## Dataset

Dataset used:

```text
simhadrisadaram/mimic-cxr-dataset
```

Split CSV used:

```text
avcuongy/mimic-cxr-split
```

Expected CSV columns:

```text
subject_id | view | best_image | path | text
```

The image path is loaded from:

```text
ROOT_DIR / best_image
```

where:

```text
ROOT_DIR = DATA_DIR / official_data_iccv_final
```

## Evaluation Metrics

The MedCLIP model is evaluated using retrieval metrics:

```text
Image-to-Text:
- I2T_R@1
- I2T_R@5
- I2T_R@10

Text-to-Image:
- T2I_R@1
- T2I_R@5
- T2I_R@10
```

BLEU, ROUGE, and CIDEr are not used here because MedCLIP does not generate text directly.

## Exported Files

After running the notebook, the following files are exported:

```text
medclip_full_final.pt
medclip_vision_encoder_final.pt
medclip_text_encoder_final.pt
test_retrieval_metrics.csv
retrieval_examples.csv
medclip_export_metadata.json
train_history.csv
```

Recommended file for downstream transfer learning:

```text
medclip_vision_encoder_final.pt
```

## Model Checkpoint

Large model checkpoints are not pushed directly to GitHub.

Download link:

```text
PASTE_GOOGLE_DRIVE_OR_KAGGLE_OUTPUT_LINK_HERE
```

Main checkpoint for other members:

```text
medclip_vision_encoder_final.pt
```

## Notebook

Main notebook:

```text
colab/medclip/medclip_retrieval_export_kaggle_style_paths.ipynb
```

The notebook performs:

```text
1. Install dependencies
2. Download MIMIC-CXR image dataset
3. Download split CSV dataset
4. Load pretrained MedCLIP-ResNet
5. Evaluate pretrained retrieval performance
6. Fine-tune contrastive model for 10 epochs if needed
7. Evaluate Recall@1, Recall@5, Recall@10
8. Export model checkpoints and metrics
```

## How to Run

Run the notebook on Kaggle or Google Colab with GPU enabled.

Recommended runtime:

```text
GPU: T4 or higher
```

For quick testing, use subsets:

```python
TRAIN_SUBSET = 10000
VAL_SUBSET = 1000
TEST_SUBSET = 1000
```

For full training, set:

```python
TRAIN_SUBSET = None
VAL_SUBSET = None
TEST_SUBSET = None
```

## Notes

MedCLIP is a contrastive model with two encoders:

```text
image encoder + text encoder
```

It does not directly generate medical reports.  
For report generation, use the image encoder as a transferred visual encoder or use MedCLIP to evaluate/rerank generated reports from other models.
