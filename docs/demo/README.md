# Demo and Evaluation Evidence

This directory contains reproducible demo runners, generated interface
screenshots, and machine-readable evaluation reports for Wafer Defect Studio.
The reports preserve the configuration and decision attached to each experiment;
the public product status is summarized here.

## Current decision

CAM v2 remains the default detection path. Patch Classification v3 is optional,
and Spatial MIL remains experimental.

The latest core-union development gate passed four generated sparse and dense
cases with exact truth-to-proposal matching, no Normal Grid leakage, and
asserted-grid occupancy P95 below `0.016`. Its recorded recommendation is
`continue_development`, so it advances the experimental path without changing
the application default.

## Evidence index

| Evidence | Scope | Result | Artifact |
| --- | --- | --- | --- |
| CAM v2 and Patch Classification v3 | Generated 20-image matched demo | Measured | [`demo-summary.json`](screenshots/realistic-20mp/demo-summary.json) |
| Spatial MIL v4 | Real generated held-out RTX 3090 gate | FAIL | [`ticket30-quality-gate.json`](ticket30-quality-gate.json) |
| Spatial MIL v5 | Position-diverse development gate | FAIL | [`ticket31-development-gate.json`](ticket31-development-gate.json) |
| Spatial MIL v5 repair | Replayed development gate | FAIL | [`ticket32-development-gate.json`](ticket32-development-gate.json) |
| Grid-contrastive Spatial MIL v6 | Five-seed development gate | PASS | [`ticket33-development-gate.json`](ticket33-development-gate.json) |
| Grid-contrastive Spatial MIL v6 | Frozen final held-out gate | FAIL | [`ticket34-final-gate.json`](ticket34-final-gate.json) |
| Core-union Spatial MIL | Four-case development gate | PASS | [`ticket38-core-union-gate.json`](ticket38-core-union-gate.json) |

The Ticket 34 gate is the latest frozen held-out decision for Spatial MIL. The
Ticket 38 result is a later development result on a different four-case
contract. These scopes remain separate: the development PASS does not replace
the frozen held-out FAIL.

## Demo screenshots

Screenshots `01`–`09` under
[`screenshots/realistic-20mp`](screenshots/realistic-20mp) are generated Ticket
29 workflow evidence. They show the shared import and Dataset Snapshot, CAM v2
and Patch Classification v3 training and Grid Evaluation, the v3 Defect
Confidence Map, and native-size PNG export.

No later Spatial MIL result is represented by those screenshots. Spatial MIL
decisions are published as JSON reports because their gates evaluate model
artifacts and proposal metrics rather than a new application workflow.

## Reproduce the connected demo

Follow the [end-to-end demo guide](phase2-end-to-end-tutorial.md) to generate a
new screenshot directory and measured summary. The runner constructs its
project in a temporary directory and only writes to the selected output path.

The generated corpus demonstrates the connected application and bounded model
comparisons. Deployment acceptance requires representative acquisition data
and criteria defined for the intended inspection process.
