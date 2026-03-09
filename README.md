# Wafer Defect Studio

A Windows desktop MVP for source-pixel-aligned wafer grid classification. It keeps the original grayscale pixels authoritative while supporting multi-label review, reproducible training snapshots, model evaluation, approximate CAM localization, and result export.

![Wafer Defect Studio showing a synthetic wafer, annotation grid, and review controls](docs/images/wafer-defect-studio-overview.png)

> The screenshot is a real PySide6 Widgets render using a generated wafer image; it is not production inspection data.

## Current status

The domain services and feature controls for tickets 01–13 are implemented. The desktop shell now exposes Create Project, Open Project, referenced Wafer Image import, a persistent Project Hub with recent-project source health, and GUI smoke coverage for the connected project/grid/annotation/review workflow. Training, evaluation, and detection controls still rely on prepared service configuration rather than one fully connected end-user workflow.

## Capabilities

- Import and retain native 8-bit or 16-bit grayscale source pixels.
- Define versioned pixel-sized Grid Profiles and per-image grid origins.
- Confirm ellipse or polygon Effective Wafer Areas with strict center-point participation.
- Create multi-label Grid Annotations and track Reviewed state separately.
- Freeze Training Scopes, Dataset Snapshots, deterministic image-level splits, and normalization bounds.
- Run isolated PyTorch training, evaluation, and detection workers with persisted job state and recovery.
- Validate and approve Training Runs, review Defect Proposals, and convert accepted proposals into annotations.
- Export CSV, JSON, and PNG results in source-image pixel coordinates.

CAM and proposal regions are approximate weak localization, not pixel-accurate segmentation masks.

## Requirements

- Windows desktop (validated locally on Windows 10.0.22631)
- Python 3.11
- PySide6 6.6 or newer
- NumPy
- PyTorch and torchvision
- CUDA is optional; GPU execution requires a compatible PyTorch build, NVIDIA driver, and available device.

The source tree currently declares only PySide6 in `pyproject.toml`; install the ML dependencies explicitly until the package metadata is completed.

## Setup

```powershell
conda create -n wafer-defect-studio python=3.11 -y
conda activate wafer-defect-studio
python -m pip install numpy torch torchvision
python -m pip install -e .
```

For GPU use, install the CUDA-enabled PyTorch build appropriate for the machine, then verify it:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

The maintained local environment used for the latest validation reported Python 3.11.15, PySide6 6.11.2, NumPy 2.4.6, PyTorch 2.13.0+cu126, torchvision 0.28.0+cu126, and CUDA available.

## Run

```powershell
wafer-defect-studio
```

This launches the editor shell and Project Hub. Use the File menu to create or open a project and import a referenced Wafer Image.

## Validate

Run the full automated suite headlessly:

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
python -m unittest discover -s tests
```

Run the focused performance checks:

```powershell
python benchmarks/first_useful_display.py
python benchmarks/interaction_feedback.py --samples 3
```

The latest local validation ran 95 tests successfully. Three focused GUI smoke tests cover create/import/display, grid/origin/area confirmation, and multi-label annotation/review; training through export remains service-pipeline evidence. A synthetic 20 MP `uint16` image reached first useful display in 0.218 seconds, and the measured view interactions remained below the 100 ms target. These are local/offscreen checks, not multi-machine production certification; see [the MVP validation report](.scratch/wafer-defect-classification/mvp-validation-report.md) for the exact evidence and limits.

## Project layout

```text
src/wafer_defect_studio/   Application, domain services, workers, and Widgets
tests/                     Unit, integration, UI-smoke, and validation tests
benchmarks/                First-display and interaction measurements
docs/adr/                  Architectural decisions
docs/agents/               Domain and delivery rules
.scratch/wafer-defect-classification/  Product spec, tickets, and validation report
```

## Known gaps found in the repository review

- The service-pipeline smoke validates domain orchestration; the three GUI smoke tests separately cover the connected desktop workflow through annotation/review.
- completed Training Runs may persist an empty environment record instead of automatically capturing Python, PyTorch, torchvision, CUDA/driver, OS, and package versions.
- Runtime imports include NumPy, PyTorch, and torchvision, but `pyproject.toml` does not declare them.
- Some UI copy says “Labeled” where the domain vocabulary calls for Grid Annotations / Defect Classes.
- `MainWindow` and the three worker modules contain repeated wiring that is a maintainability concern, though not an immediate correctness failure.

SQLite remains the single local source of truth. Source-image pixel coordinates are authoritative throughout annotation, localization, and export.
