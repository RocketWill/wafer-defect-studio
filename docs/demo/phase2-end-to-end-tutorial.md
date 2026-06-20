# End-to-End Generated Demo

This demo runs the connected Wafer Defect Studio workflow from generated Wafer
Images through Dataset Snapshot creation, ResNet18 training, Grid Evaluation,
Defect Confidence Map generation, and result export. It produces a matched CAM
v2 and Patch Classification v3 comparison from one immutable image-level split.

## Prerequisites

- Windows with Python 3.11
- An editable installation of this repository
- PySide6, NumPy, PyTorch, and torchvision
- A CUDA-enabled PyTorch runtime for the realistic-source matched comparison

Run commands from the repository root.

## Run the matched demo

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
$env:QT_QPA_FONTDIR = "C:/Windows/Fonts"
python docs/demo/run_phase2_demo.py `
  --source-image docs/demo/assets/realistic-wafer-20mp.png `
  --output docs/demo/screenshots/realistic-20mp
```

The 4,472 × 4,472 input is generated size-reference data. The runner uses a
1,536 × 1,536 processing proxy to create 20 deterministic Wafer Images and an
immutable 16/2/2 train, validation, and test split. CAM v2 and Patch
Classification v3 share the same `split_id`. Each Defect Class has two
independent asserted-image examples in validation and two in test.

Thresholds come from validation data. Grid Evaluation reads the test images
after threshold selection.

## Run the compact UI path

Omit `--source-image` to exercise the CAM v2 interface flow with a smaller
generated source:

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
python docs/demo/run_phase2_demo.py `
  --output artifacts/phase2-demo-synthetic
```

This command checks the connected UI path. It does not run the 20-image matched
comparison.

## Workflow

1. Import generated Wafer Images and apply a 512 px Grid Profile.
2. Confirm the Effective Wafer Area and create `scratch` and `particle` Grid
   Annotations.
3. Mark the images Reviewed, then freeze a Dataset Snapshot and image-level
   Dataset Split.
4. Train and evaluate CAM v2 from one classifier input per Annotation Grid.
5. Train Patch Classification v3 from ordered Model Patches grouped into Patch
   Bags. Positive truth remains attached to the bag rather than an individual
   Model Patch.
6. Score v3 patches at 128 px size and 64 px stride, then use per-class max
   pooling for Grid logits.
7. Apply validation-derived thresholds and run dense detection.
8. Render the Defect Confidence Map and export the native-size PNG result.

Both model paths reuse the same Snapshot, evaluation, approval, Detection
Profile, Proposal, review, conversion, and export contracts.

## Outputs

Screenshots `01`–`09` are Ticket 29 evidence for the connected CAM v2 and Patch
Classification v3 workflow.

| Stage | Output |
| --- | --- |
| Import | [`01-import.png`](screenshots/realistic-20mp/01-import.png) |
| Two-class annotation | [`02-annotation-two-classes.png`](screenshots/realistic-20mp/02-annotation-two-classes.png) |
| Dataset Snapshot | [`03-dataset-snapshot.png`](screenshots/realistic-20mp/03-dataset-snapshot.png) |
| CAM v2 training | [`04-cam-v2-training.png`](screenshots/realistic-20mp/04-cam-v2-training.png) |
| CAM v2 Grid Evaluation | [`05-cam-v2-grid-evaluation.png`](screenshots/realistic-20mp/05-cam-v2-grid-evaluation.png) |
| Patch v3 training | [`06-patch-v3-training.png`](screenshots/realistic-20mp/06-patch-v3-training.png) |
| Patch v3 Grid Evaluation | [`07-patch-v3-grid-evaluation.png`](screenshots/realistic-20mp/07-patch-v3-grid-evaluation.png) |
| Patch v3 Defect Confidence Map | [`08-patch-v3-confidence-map.png`](screenshots/realistic-20mp/08-patch-v3-confidence-map.png) |
| Patch v3 PNG export | [`09-patch-v3-heatmap-export.png`](screenshots/realistic-20mp/09-patch-v3-heatmap-export.png) |
| Matched measurement record | [`demo-summary.json`](screenshots/realistic-20mp/demo-summary.json) |

The screenshots remain application-workflow evidence. Spatial MIL model gates
are indexed separately in [`docs/demo/README.md`](README.md).

## Measured comparison

The tracked RTX 3090 summary uses the same immutable split for both models:

| Model | Scratch F1 | Particle F1 | Exact Grid match |
| --- | ---: | ---: | ---: |
| CAM v2 | 0.4000 | 1.0000 | 15/18 |
| Patch Classification v3 | 0.2000 | 0.8571 | 10/18 |

Coarse localization reports asserted-Grid intersection rate followed by Normal
Grid leak rate:

| Model and class | Intersection | Leakage |
| --- | ---: | ---: |
| CAM v2 scratch | 1.0000 | 1.0000 |
| CAM v2 particle | 1.0000 | 1.0000 |
| Patch v3 scratch | 0.0000 | 0.3125 |
| Patch v3 particle | 0.2500 | 0.0000 |

CAM v2 remains the default because the matched comparison does not support
promoting Patch Classification v3. Patch v3 remains available as an optional
path through the same application workflow.

Grid Annotation is weak multi-label truth. Defect Confidence Maps and retained
regions provide approximate source-coordinate localization rather than pixel
segmentation. The generated corpus measures the connected workflow and the
stated comparison; production acceptance requires representative Wafer Images.

## Cleanup and rerun

The runner creates its SQLite project in a temporary directory. The temporary
project is removed after execution, while the selected output directory keeps
the screenshots and `demo-summary.json`. Rerunning the matched command replaces
that output set with a new measurement record.
