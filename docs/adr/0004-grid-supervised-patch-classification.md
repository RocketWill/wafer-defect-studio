---
status: proposed
---

# Add grid-supervised Patch Classification without replacing CAM checkpoints

Small defects occupy too little of an Annotation Grid for global-average-pooled CAM to localize reliably. New Patch Classification Training Runs will therefore treat the Model Patches inside each Annotation Grid as a multi-instance Patch Bag and pool their class scores against the existing Grid Annotation; they will not invent per-patch truth or require pixel masks. Existing ResNet18 v1/v2 CAM checkpoints remain loadable with their original semantics, while the new checkpoint contract is versioned separately and becomes the default only after held-out Grid Evaluation and localization evidence pass.

## Consequences

- Grid Annotation, Dataset Snapshot, image-level Dataset Split, approval, Detection Run, Defect Confidence Map, Proposal, Review, conversion, and export contracts remain authoritative.
- The initial pooling rule, patch geometry, and quality gate must be explicit and reproducible; none may be inferred from one Demo screenshot.
- This does not claim segmentation or pixel-accurate boundaries.
