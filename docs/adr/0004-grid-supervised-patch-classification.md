---
status: accepted
---

# Add grid-supervised Patch Classification without replacing CAM checkpoints

Small defects occupy too little of an Annotation Grid for global-average-pooled CAM to localize reliably. Patch Classification Training Runs therefore treat the Model Patches inside each Annotation Grid as a multi-instance Patch Bag and pool their class scores against the existing Grid Annotation; they do not invent per-patch truth or require pixel masks. Existing ResNet18 v1/v2 CAM checkpoints remain loadable with their original semantics, while the v3 checkpoint contract is versioned separately.

The accepted design adds v3 as an optional workflow; it does not promote v3 to the default. The latest matched 20-image Demo did not establish a quality improvement: v3 reduced particle Normal Grid leakage to 0.0000, but scratch retained 0.3125 leakage, both classes lost asserted-Grid intersection, and scratch Grid F1, particle Grid F1, and exact Grid match were lower than v2. CAM v2 remains the default until broader evidence supports a different decision.

## Consequences

- Grid Annotation, Dataset Snapshot, image-level Dataset Split, approval, Detection Run, Defect Confidence Map, Proposal, Review, conversion, and export contracts remain authoritative.
- The initial pooling rule, patch geometry, and quality gate must be explicit and reproducible; none may be inferred from one Demo screenshot.
- The accepted initial v3 geometry is 128 px Model Patches, 64 px stride, and per-class max pooling.
- This does not claim segmentation or pixel-accurate boundaries.
- Neurocle remains a behavioral reference only; this decision does not claim architectural or quality equivalence.
