# Wafer Defect Studio

Wafer Defect Studio is a Windows desktop platform for grid-supervised wafer
defect classification and approximate localization. It preserves native
grayscale source pixels while connecting data review, reproducible training,
model evaluation, full-image detection, proposal review, and result export in
one PySide6 application.

![Wafer Defect Studio annotation workspace showing a synthetic wafer, Annotation Grids, Effective Wafer Area, and review controls](docs/images/wafer-defect-studio-overview.png)

The interface screenshots use deterministic generated wafer images rendered by
the application. Source-image coordinates remain authoritative throughout the
workflow.

## Workflow

1. Import native 8-bit or 16-bit grayscale Wafer Images.
2. Define a pixel-sized Grid Profile and confirm the Effective Wafer Area.
3. Apply multi-label Grid Annotations and mark each completed image Reviewed.
4. Freeze a Dataset Snapshot with an image-level train, validation, and test
   split.
5. Train and evaluate a checkpoint-backed ResNet18 model, select per-class
   thresholds, and approve the evaluation.
6. Run full-image overlapping-window inference to produce class-specific Defect
   Confidence Maps and Defect Proposals in source coordinates.
7. Review Proposals, explicitly convert selected results to Grid Annotations,
   and export CSV, JSON, or native-size PNG results.

A Grid Annotation states that a grid contains visible evidence of one or more
Defect Classes. It does not assign a pixel boundary. The generated Defect
Confidence Maps and Proposals preserve the same approximate-localization
contract.

## Model paths

| Path | Status | Role |
| --- | --- | --- |
| CAM v2 | Default | ResNet18 layer4 activation projected through the classifier weights |
| Patch Classification v3 | Optional | Dense 128 px Model Patches evaluated at 64 px stride |
| Spatial MIL | Experimental | Development path for finer proposal localization from grid supervision |

CAM v2 is the default detection path. Patch Classification v3 uses the same
Dataset Snapshot, approval, Detection Profile, and proposal-review workflow.
Spatial MIL remains experimental.

The latest Spatial MIL development result evaluates a core-union objective on
four generated cases spanning sparse and dense scratch and particle patterns.
All four cases passed the frozen development contract with exact truth,
proposal, and match counts, proposal precision and recall of `1.0`, no Normal
Grid leakage, and asserted-grid occupancy P95 below `0.016`. The report records
the recommendation as `continue_development`; it does not change the default
model path.

- [Ticket 38 core-union development gate](docs/demo/ticket38-core-union-gate.json)
- [Ticket 38 frozen contract](docs/demo/ticket38-core-union-gate-contract.json)

## Capabilities

- Native-resolution grayscale import with source fingerprint checks.
- Versioned Grid Profiles, per-image grid origins, and ellipse or polygon
  Effective Wafer Areas.
- Multi-label Grid Annotations with explicit Reviewed state and derived Normal
  Grids.
- Immutable Training Scopes, Dataset Snapshots, image-level splits, and
  normalization bounds.
- Separate worker processes for training, evaluation, and detection, with the
  GUI-side project service as the only SQLite writer.
- Snapshot-backed ResNet18 training, checkpoint evaluation, per-class threshold
  selection, and explicit approval.
- Overlapping-window CAM v2 and dense Patch Classification v3 detection.
- Append-only Proposal acceptance, rejection, and correction history.
- Explicit Proposal-to-Grid Annotation conversion with run, profile, and
  proposal provenance.
- CSV, JSON, and native-size PNG export in source-image coordinates.

## Requirements

- Windows 10 or newer
- Python 3.11
- PySide6 6.6 or newer
- NumPy
- PyTorch and torchvision

CUDA is optional. GPU execution requires a compatible PyTorch build, NVIDIA
driver, and available CUDA device.

## Install

```powershell
conda create -n wafer-defect-studio python=3.11 -y
conda activate wafer-defect-studio
python -m pip install -e .
```

For GPU use, install the CUDA-enabled PyTorch build appropriate for the target
machine before the editable install, then verify the runtime:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

## Run

```powershell
wafer-defect-studio
```

The application opens at the Project Hub. Create or open a project, import a
Wafer Image, then move through the Data, Annotate, Dataset, Train, Evaluate,
Detect, and Review workspaces.

## Demo

Run the deterministic end-to-end demo headlessly:

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
$env:QT_QPA_FONTDIR = "C:/Windows/Fonts"
python docs/demo/run_phase2_demo.py `
  --source-image docs/demo/assets/realistic-wafer-20mp.png `
  --output docs/demo/screenshots/realistic-20mp
```

The 4,472 × 4,472 source image is generated size-reference data. The demo uses
a 1,536 × 1,536 processing proxy to construct 20 deterministic Wafer Images
with an immutable 16/2/2 image-level split. CAM v2 and Patch Classification v3
share the split, and class thresholds are selected from validation data before
Grid Evaluation uses the test images.

The tracked `01`–`09` screenshots record the connected CAM v2 and Patch
Classification v3 workflow:

- [End-to-end demo guide](docs/demo/phase2-end-to-end-tutorial.md)
- [Measured demo summary](docs/demo/screenshots/realistic-20mp/demo-summary.json)

## Evidence

The repository keeps machine-readable reports beside the runners that produced
them. Three reports define the current localization evidence boundary:

| Evidence | Result | Interpretation |
| --- | --- | --- |
| [CAM v2 and Patch Classification v3 matched demo](docs/demo/screenshots/realistic-20mp/demo-summary.json) | Measured generated-data comparison | Connected training, evaluation, detection, and export workflow |
| [Spatial MIL v6 frozen final gate](docs/demo/ticket34-final-gate.json) | FAIL | The frozen held-out extent and grid-quality targets were not met |
| [Core-union development gate](docs/demo/ticket38-core-union-gate.json) | PASS | The four-case development contract passed; recommendation is `continue_development` |

Detailed commands, screenshot provenance, and earlier experimental results are
kept in the [end-to-end demo guide](docs/demo/phase2-end-to-end-tutorial.md).

## Validate

Run the repository suite headlessly:

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
python -m unittest discover -s tests -p "test*.py" -v
```

Focused performance checks are also available:

```powershell
python benchmarks/first_useful_display.py
python benchmarks/interaction_feedback.py --samples 3
```

## Project layout

```text
src/wafer_defect_studio/    Application, domain services, workers, and Widgets
tests/                      Unit, integration, UI-smoke, and validation tests
benchmarks/                 First-display and interaction measurements
docs/adr/                   Architectural decisions
docs/demo/                  Reproducible demos, reports, and screenshots
docs/images/                Public interface screenshots
```

## Current boundary

The application targets a single offline Windows workstation and keeps SQLite
as the local metadata source of truth. Model outputs are Defect Confidence Maps
and Defect Proposals derived from grid supervision. Production inspection
accuracy requires evaluation on representative acquisition data and acceptance
criteria defined for the intended deployment.
