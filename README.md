# Wafer Defect Studio

A Windows desktop MVP for source-pixel-aligned wafer grid classification. The
original grayscale pixels remain authoritative while the application supports
multi-label review, reproducible training snapshots, checkpoint-backed ResNet18
training and evaluation, CAM v2 or grid-supervised Patch Classification v3,
approximate localization, proposal review, conversion, and result export.

![Wafer Defect Studio annotation workspace showing a synthetic wafer, grid, effective area, and review controls](docs/images/wafer-defect-studio-overview.png)

> The screenshots in this README use deterministic synthetic wafer data and
> real PySide6 Widgets renders. They are UI evidence, not production inspection
> data or model-accuracy claims.

## Current status

Phase 2 MVP is complete (Tickets 01–28), and Ticket 29 adds an optional
grid-supervised Patch Classification path. The active-project desktop happy
path is connected:

1. Import a native 8-bit or 16-bit grayscale wafer image.
2. Configure a pixel-sized Grid Profile and confirmed Effective Wafer Area.
3. Apply multi-label Grid Annotations and mark the image Reviewed.
4. Freeze a Dataset Snapshot, choose CAM v2 or Patch Classification v3, train
   ResNet18 from the frozen input bundle, evaluate the published checkpoint,
   choose thresholds, and approve the evaluation.
5. Create a Detection Profile from the approved evaluation and generate a
   source-coordinate Defect Confidence Map and Proposals. CAM v2 projects
   layer4+FC; v3 scores dense 128 px Model Patches at 64 px stride and combines
   Patch Bags with per-class max pooling.
6. Review proposals, preview and explicitly confirm conversion to Grid
   Annotations, then prepare and publish CSV, JSON, and native-size PNG results.
7. Reopen the project with the Review filters and Result Export context intact.

CAM v2 remains the default; Patch Classification v3 is optional. Both paths
produce approximate weak localization, not pixel-accurate boundaries.

### Phase 2 UI snapshots

<p>
  <img alt="Phase 2 Proposal Review queue" src="docs/images/phase2-review.png" width="32%">
  <img alt="Phase 2 Proposal Conversion preview" src="docs/images/phase2-conversion.png" width="32%">
  <img alt="Phase 2 Result Export controls" src="docs/images/phase2-result-export.png" width="32%">
</p>

### End-to-end demo

推薦用 repo 內的 generated 20MP wafer source，在 CUDA 上重跑同一個 immutable
image-level Dataset Split 的 CAM v2／Patch Classification v3 matched comparison：

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
$env:QT_QPA_FONTDIR = "C:/Windows/Fonts"
& 'E:\miniconda3\envs\wafer-defect-studio\python.exe' docs/demo/run_phase2_demo.py `
  --source-image docs/demo/assets/realistic-wafer-20mp.png `
  --output .\docs\demo\screenshots\realistic-20mp
```

這張 4,472×4,472（19,998,784 pixels）圖片是生成並放大的尺寸參考，不是真實
量測資料。runner 以 1,536×1,536 processing proxy 建立 20 張 deterministic、
彼此不同的 generated Wafer Images。20 images; image-level split 16/2/2；兩類在
validation/test 各有 2 張 independent asserted-image support，兩個模型使用同一
split，threshold 只由 validation 產生，Grid Evaluation 數字只來自 test。

2026-05-04 最新重跑的 RTX 3090 summary 是 `measured`，結果混合：

- v2: scratch F1 0.4000, particle F1 1.0000, exact Grid match 15/18
- v3: scratch F1 0.2000, particle F1 0.8571, exact Grid match 10/18
- v2: scratch intersection/leak 1.0000/1.0000; particle intersection/leak 1.0000/1.0000
- v3: scratch intersection/leak 0.0000/0.3125; particle intersection/leak 0.2500/0.0000
- Patch geometry: 128 px size, 64 px stride, max pooling.

因此 CAM v2 remains the default; Patch Classification v3 is optional. 這次
generated-data comparison does not establish segmentation, Neurocle equivalence,
or production accuracy.

若只要快速驗證 UI 串接，可省略 `--source-image` 使用 synthetic source。

完整步驟、UI 截圖的證據範圍、兩個 checkpoint 版本與 matched comparison 邊界請參考
[Phase 2 端到端 Demo 教學](docs/demo/phase2-end-to-end-tutorial.md)。

## Capabilities

- Import and retain native 8-bit or 16-bit grayscale source pixels.
- Define versioned pixel-sized Grid Profiles and per-image grid origins.
- Confirm ellipse or polygon Effective Wafer Areas with strict center-point
  participation.
- Create multi-label Grid Annotations and track Reviewed state separately.
- Freeze Training Scopes, Dataset Snapshots, deterministic image-level splits,
  and normalization bounds.
- Run isolated PyTorch ResNet18 training from a Snapshot-backed input bundle,
  checkpoint-backed evaluation, threshold selection, and approval workers with
  persisted job state and recovery.
- Train v3 Patch Bags from existing multi-label Grid Annotations without
  assigning invented positive truth to individual Model Patches.
- Create approved-evaluation Detection Profiles, run overlapping-window
  CAM v2 or dense patch v3 detection, retain thresholded connected regions, and
  persist the same source-coordinate Detection Runs and Proposals.
- Review Proposals with append-only Accept/Reject/Correct revisions.
- Preview and explicitly confirm Proposal-to-Grid Annotation conversion with
  run/profile/proposal provenance.
- Export reviewed results as CSV, JSON, and native-size PNG with source-image
  coordinates, selected-class CAM/retained-region context, and grid/proposal
  overlays. CSV/JSON Proposal schemas do not contain mask fields.

SQLite is the local source of truth. Source-image pixel coordinates are
authoritative throughout annotation, localization, conversion, and export.

## Requirements

- Windows desktop (validated locally on Windows 10.0.22631)
- Python 3.11
- PySide6 6.6 or newer
- NumPy
- PyTorch and torchvision
- CUDA is optional; GPU execution requires a compatible PyTorch build, NVIDIA
  driver, and an available device.

The source tree currently declares only PySide6 in `pyproject.toml`; install the
ML dependencies explicitly until package metadata is completed.

## Setup

```powershell
conda create -n wafer-defect-studio python=3.11 -y
conda activate wafer-defect-studio
python -m pip install numpy torch torchvision
python -m pip install -e .
```

For GPU use, install the CUDA-enabled PyTorch build appropriate for the machine,
then verify it:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

The environment used for the latest local checks reported Python 3.11.15,
PySide6 6.11.2, NumPy 2.4.6, PyTorch 2.13.0+cu126, torchvision 0.28.0+cu126,
and CUDA available.

## Run

```powershell
wafer-defect-studio
```

This launches the editor shell and Project Hub. Use the File menu to create or
open a project, import a referenced Wafer Image, and move through the Data,
Dataset, Train, Evaluate, Detect, and Review workspaces.

## Validate

Run the repository suite headlessly:

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
& 'E:\miniconda3\envs\wafer-defect-studio\python.exe' -m unittest discover -s tests -p "test*.py" -v
```

Earlier Phase 2 validation evidence is recorded in
[Ticket 28](.scratch/wafer-defect-classification/issues/28-model-backed-confidence-regions.md)
and the [MVP validation report](.scratch/wafer-defect-classification/mvp-validation-report.md).
The latest matched GPU evidence is recorded in
[Ticket 29](.scratch/wafer-defect-classification/issues/29-grid-supervised-patch-classification.md)
and [`demo-summary.json`](docs/demo/screenshots/realistic-20mp/demo-summary.json):

- Ticket 28 focused suite: **53/53 passed** in the E conda environment.
- Full discovery: **155 tests**, with 4 known pre-existing Qt/offscreen
  canvas-geometry failures in `test_grid_controls`, `test_grid_overlay`,
  `test_image_grid_origin`, and `test_wafer_interaction`.
- The remaining tests passed; `py_compile` and `git diff --check` passed.

The matched run completes Snapshot-backed Training, checkpoint-scored Grid
Evaluation, and dense Detection for both v2 and v3. Its screenshot directory
contains the shared import/Snapshot setup, each model's Training and Grid
Evaluation state, and the v3 Defect Confidence Map/export. `demo-summary.json`
is the machine-readable latest matched measurement record.

Those four failures are outside the active Phase 2 project path and are kept
visible rather than reported as a green full suite.

Run the focused performance checks when needed:

```powershell
python benchmarks/first_useful_display.py
python benchmarks/interaction_feedback.py --samples 3
```

## Project layout

```text
src/wafer_defect_studio/                 Application, domain services, workers, Widgets
tests/                                   Unit, integration, UI-smoke, validation tests
benchmarks/                              First-display and interaction measurements
docs/images/                             README and Phase 2 UI screenshots
docs/adr/                                Architectural decisions
docs/agents/                             Domain and delivery rules
.scratch/wafer-defect-classification/    Product spec, tickets, and validation evidence
```

## Known limits and deferred roadmap

- `pyproject.toml` does not yet declare NumPy, PyTorch, or torchvision.
- CAM and patch confidence regions remain approximate weak localization; pixel
  masks and segmentation are intentionally out of scope.
- Comprehensive Wafer Evaluation, ResNet50,
  EfficientNet-B0, ONNX import/parity, active learning, backup/collect UI,
  advanced job control, polished dark theme, layout profiles, and image
  pyramids remain deferred.

SQLite remains the single local source of truth, and no deferred item is needed
to exercise the Phase 2 happy path shown above.
