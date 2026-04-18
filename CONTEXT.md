# Wafer Defect Classification

This context defines the language of a desktop training platform for locating and classifying small defects in large wafer images without reducing the source image resolution.

## Language

**Algorithm Engineer**:
The primary user, who understands deep-learning training, evaluation, and model parameters.
_Avoid_: Operator, inspector

**Wafer Image**:
A source image of a wafer that may be too large to process as a single model input and whose native detail must be preserved.
_Avoid_: Whole-slide image, source photo

**Changed Source**:
A Wafer Image whose referenced source file no longer matches the content that was annotated. Its annotations, review state, and derived results are not trusted until an Algorithm Engineer resolves the change.
_Avoid_: Missing image, updated image

**Annotation Grid**:
A non-overlapping rectangular region in a regular partition over a Wafer Image, used as the unit presented for human annotation.
_Avoid_: Grid, cell, inference window

**Defect Class**:
A named category of wafer defect, such as scratch or particle. A Defect Class already used by Grid Annotations may be deactivated or merged, but is not discarded.
_Avoid_: Label, tag

**Grid Annotation**:
The assertion that an Annotation Grid contains one or more Defect Classes; it does not assert that every pixel in the Annotation Grid is defective. A defect crossing multiple Annotation Grids is asserted on every grid that visibly contains it.
_Avoid_: Segmentation mask, pixel annotation

**Unreviewed Grid**:
An Annotation Grid whose contents have not yet been confirmed by an Algorithm Engineer. It must not be treated as normal by omission.
_Avoid_: Negative grid, normal grid

**Reviewed Wafer Image**:
A Wafer Image whose Grid Annotations have been completed and explicitly confirmed by an Algorithm Engineer. Its remaining unannotated Annotation Grids within the Effective Wafer Area are considered normal.
_Avoid_: Finished image, labelled image

**Wafer Image Under Review**:
A Wafer Image whose annotations are incomplete or being edited. It has no Normal Grids derived from missing annotations until it is explicitly confirmed again.
_Avoid_: Draft image, reopened image

**Normal Grid**:
An Annotation Grid in a Reviewed Wafer Image that contains none of the project's Defect Classes.
_Avoid_: Unlabelled grid, negative grid

**Grid Profile**:
The regular Annotation Grid geometry shared by a project. Individual Wafer Images may adjust where that geometry begins without changing its grid size.
_Avoid_: Grid configuration, tile settings

**Inference Window**:
A model input region sampled across a Wafer Image during detection. Inference Windows may overlap and are independent of Annotation Grid boundaries.
_Avoid_: Annotation grid, cell

**Model Patch**:
A source-coordinate model input sampled inside an Annotation Grid or across a Wafer Image. It is smaller than an Annotation Grid when Patch Classification is used and does not inherit a Defect Class by itself.
_Avoid_: Annotation Grid, tile label, defect crop

**Patch Bag**:
The ordered Model Patches derived from one Annotation Grid and trained against that grid's multi-label truth. A positive Patch Bag asserts that at least one contained Model Patch supports each asserted Defect Class; it does not identify which patch.
_Avoid_: Patch label, segmentation region, bounding box

**Data Group**:
A set of Wafer Images that share acquisition conditions such as product, magnification, camera, and lighting. A project may contain multiple Data Groups.
_Avoid_: Dataset, batch, category

**Training Scope**:
The explicitly selected Data Groups whose Wafer Images are eligible for a particular model training and evaluation cycle.
_Avoid_: Dataset filter, selected folders

**Dataset Split**:
An immutable assignment of Wafer Images to training, validation, and test sets for a Training Run. Each Wafer Image represents a distinct physical wafer and is the smallest split unit.
_Avoid_: Patch split, random batches

**Dataset Snapshot**:
An immutable capture of the Wafer Images, review states, Grid Annotations, and Dataset Split used to create a Training Run. Later annotation edits do not change it.
_Avoid_: Current dataset, annotation backup

**Training Run**:
An immutable record of one model-training attempt, including its Training Scope, Dataset Split, model inputs, learned checkpoint, evaluation results, and Class Thresholds.
_Avoid_: Model, experiment folder, checkpoint

**Resumed Training Run**:
A new Training Run initialized from a checkpoint of a parent Training Run. It preserves the parent rather than continuing or rewriting it in place.
_Avoid_: Continued run, overwritten run

**Candidate Run**:
A completed Training Run that has not yet satisfied or been accepted against the project's evaluation criteria.
_Avoid_: Draft model

**Validated Run**:
A Training Run whose evaluation results satisfy the project's defined criteria.
_Avoid_: Good model, passing checkpoint

**Approved Run**:
A Validated Run explicitly accepted by an Algorithm Engineer for detection use. Approval may later be withdrawn without deleting the Training Run.
_Avoid_: Production model, deployed model

**Model Bundle**:
A portable, integrity-checked package containing a trained model and the complete contract required to reproduce its inputs, class outputs, confidence maps, thresholds, and provenance.
_Avoid_: Checkpoint, ONNX file, model archive

**Detection Profile**:
An immutable set of Class Thresholds, aggregation, and proposal postprocessing settings applied with a specific Training Run. Changing these settings creates a new profile rather than changing the Training Run.
_Avoid_: Inference settings, modified model

**Effective Wafer Area**:
The engineer-confirmed ellipse or polygon within a Wafer Image that contains inspectable wafer material. An Annotation Grid participates when its center lies inside this area; other grids are excluded rather than treated as normal.
_Avoid_: Background, crop

**Class Threshold**:
The confidence boundary above which a model proposes a specific Defect Class. Each Defect Class may have its own threshold.
_Avoid_: Global threshold, cutoff

**Grid Evaluation**:
Evaluation of model classification at the Annotation Grid level.
_Avoid_: Patch test, wafer evaluation

**Wafer Evaluation**:
Evaluation of the complete detection process on Wafer Images, including overlapping Inference Windows, aggregation, and Class Thresholds.
_Avoid_: Grid evaluation, image accuracy

**Defect Confidence Map**:
A class-specific spatial map assembled from class activation or Model Patch confidence within overlapping Inference Windows. It provides weakly supervised approximate localization, not a pixel-accurate defect boundary.
_Avoid_: Segmentation mask, defect outline, heatmap

**Defect Proposal**:
A model-suggested connected region derived from one class of a Defect Confidence Map, with an approximate extent and confidence values. Proposals of different Defect Classes may overlap; none is a confirmed defect.
_Avoid_: Detection, annotation, defect instance

**Detection Review**:
An Algorithm Engineer's acceptance, rejection, or correction of Defect Proposals for a Wafer Image. It remains separate from training truth until explicitly converted to Grid Annotations.
_Avoid_: Annotation, ground truth, validation

**Detection Run**:
An immutable record of applying an Approved Run and explicit detection settings to one or more Wafer Images. Repeating detection creates a new Detection Run rather than replacing an earlier result.
_Avoid_: Inference job, result folder

**Review Queue**:
A prioritized collection of Defect Proposals selected for human attention because of low confidence, class conflict, or disagreement between Training Runs.
_Avoid_: Annotation queue, active training set

**Display Mapping**:
A reversible mapping of source grayscale values for human viewing. It does not alter model inputs or learned behavior.
_Avoid_: Preprocessing, normalization

**Model Preprocessing**:
The reproducible transformation from source pixel values to model input values, retained as part of a trained model's definition.
_Avoid_: Display mapping, viewer contrast
