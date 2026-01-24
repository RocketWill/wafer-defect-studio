# Slice 01.6c — 20 MP first-useful-display measurement

Measured at: `2026-01-24T15:29:40+08:00`

## Local environment

- Python: `3.11.9` (CPython)
- PySide6: `6.11.2`; Qt: `6.11.2`
- Platform: `Windows-10-10.0.22631-SP0`; machine: `AMD64`
- `QT_QPA_PLATFORM`: `offscreen`

## Measurement

- Fixture: deterministic `5000 × 4000` `Grayscale16` TIFF, filled with `0x1234`.
- Setup (project creation, TIFF write, registration, and reopen) completed before timing.
- Timed interval: immediately before `MainWindow.load_wafer_image(...)` through source-health hashing, background decode, native array copy, and the first `Ready` state with a non-null scene pixmap.
- First useful display: `0.433 s`.
- Target: `≤ 3.000 s`; result: **MET**.
- Final status: `Ready`; native payload: `5000x4000` `uint16`; pixmap ready: `True`; decode threads cleaned: `True`.

## Limitations

- This is one local offscreen run on one deterministic fixture; it is not a cold-start, multi-run distribution, or production-display benchmark.
- The result includes the current source-health SHA-256 pass and native copy, but excludes project/TIFF setup by design.
- No optimization, caching, image pyramid, or benchmark framework was added for this measurement.
