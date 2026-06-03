# Phase 2 MVP 端到端 Demo

這份教學走過既有 happy path，並重跑 Ticket 29 的 matched comparison：Grid
Annotation → Patch Bag training → Grid Evaluation → dense patch Defect Confidence
Map → Review／Export。CAM v2 remains the default; Patch Classification v3 is
optional。

Ticket 30 的 real generated held-out RTX 3090 evidence 已完成發布，但 Spatial
MIL v4 gate **FAIL**。因此 CAM v2 remains the default；v4 is experimental，
不升級為預設或 production path。完整品質決策在
[`ticket30-quality-gate.json`](ticket30-quality-gate.json)（repo path：
`docs/demo/ticket30-quality-gate.json`）。這是 generated-data evidence，不是
segmentation、Neurocle 等價或 production accuracy 聲明。

Ticket 31 另以五個 development seeds 在 RTX 3090 實跑 Spatial MIL v5；strict
development gate 仍為 **FAIL**，10 個 seed/class rows 中 7 個失敗，主要問題是
whole-wafer leakage。決策報告在
[`ticket31-development-gate.json`](ticket31-development-gate.json)。依 frozen
stop condition，不執行 final held-out gate；CAM v2 維持預設，v5 維持
experimental。這也不是 Neurocle 等價、segmentation 或 production accuracy
聲明。

Ticket 32 修正 Ticket 31 已知的 checkpoint selection、selected-epoch metadata
與 development corpus diversity 問題，並在同一張 RTX 3090 重跑五個 seeds。
repaired development gate 仍為 **FAIL**：10 個 seed/class rows 僅 2 個通過，
其餘 8 個出現 whole-wafer activation；五個 seeds 都由 validation spatial
selector 選到 epoch 1。完整報告在
[`ticket32-development-gate.json`](ticket32-development-gate.json)。依 stop
condition，仍不執行 final held-out gate；CAM v2 維持預設，v5 維持
experimental。這不是 Neurocle 等價、segmentation 或 production accuracy
聲明。

Ticket 33 的 Grid-contrastive Spatial MIL v6 改用 top-1% localized positive
evidence、dense absent-class suppression 與同圖 asserted-vs-Normal Grid
ranking。RTX 3090 五個 development seeds 的 10/10 rows 全部 **PASS**：Grid
precision/recall 皆為 1.0，Normal Grid leakage 皆為 0。完整報告在
[`ticket33-development-gate.json`](ticket33-development-gate.json)。這只解封
後續獨立的 final held-out gate；CAM v2 仍是預設，且不構成 Neurocle 等價、
segmentation 或 production accuracy 聲明。

Ticket 34 已完成 Grid-contrastive Spatial MIL v6 的 frozen final held-out
gate。一次 sealed RTX 3090 執行只使用 Ticket 30 的三個 final members
(`ticket30-evidence-17.png`、`ticket30-evidence-42.png`、
`ticket30-evidence-91.png`)，涵蓋 `scratch`／`particle` 兩個 Defect Class，
共 3 × 2 seed/class rows；每列至少有 150 defect instances。Final calibration
禁止，threshold 只來自 sealed validation artifact。canonical report 是
[`ticket34-final-gate.json`](ticket34-final-gate.json)，pre-final seal 是
[`ticket34-final-seal.json`](ticket34-final-seal.json)。

v6 final gate 結果為 **FAIL**。Scratch 三個 seeds 的 defect-coverage recall
都是 `0.9`，occupancy P95 為
`0.5312423706054688–0.5492210388183594`（高於 `0.53`）。Particle Grid
precision 是 `0.5`、Normal Grid leakage 是 `0.125`，occupancy P95 為
`0.8409576416015625–0.8425254821777344`（高於 `0.84`）。六列都有 positive
score-separation margins，但 frozen final targets 仍未達成。CAM v2 remains
the default；v6 remains experimental。No 10–13 screenshots were added；01–09
remain Ticket 29 evidence。這個結果不宣稱 Neurocle equivalence、segmentation
或 production accuracy。

## 執行 Demo

在 repository root 執行 realistic-source GPU Demo：

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
$env:QT_QPA_FONTDIR = "C:/Windows/Fonts"
& 'E:\miniconda3\envs\wafer-defect-studio\python.exe' docs/demo/run_phase2_demo.py `
  --source-image docs/demo/assets/realistic-wafer-20mp.png `
  --output .\docs\demo\screenshots\realistic-20mp
```

程式在暫存目錄建立 project，不修改既有專案。realistic source 要求 CUDA；
若 CUDA 不可用會明確失敗。4,472×4,472（19,998,784 pixels）輸入是生成並放大
的尺寸參考，不是真實量測資料；runner 以 1,536×1,536 processing geometry
建立 20 張彼此不同的 deterministic generated Wafer Images。

20 images; image-level split 16/2/2。train、validation、test 都以 Wafer Image
隔離，CAM v2 與 Patch Classification v3 使用相同 `split_id`。scratch 與
particle 在 validation/test 各有 2 張 independent asserted-image support。
threshold 由 validation 產生；以下 Grid Evaluation 只讀 test，沒有用 test
調參。

若只要快速驗證 UI 串接，可省略 `--source-image` 使用小型 synthetic source：

```powershell
& 'E:\miniconda3\envs\wafer-defect-studio\python.exe' docs/demo/run_phase2_demo.py `
  --output .\artifacts\phase2-demo-synthetic
```

這個快速分支只走 CAM v2 UI flow，不是 20-image matched quality comparison。

## 工作流程

1. 匯入 Wafer Images，套用 512 px Grid Profile 與 Effective Wafer Area。
2. 以既有多標籤 Grid Annotation 寫入 `scratch`／`particle` truth，完成 Reviewed。
3. 建立 immutable Dataset Snapshot 與 image-level Dataset Split。
4. CAM v2 維持原本一個 Annotation Grid 對一個 classifier input 的契約。
5. 選擇 Patch Classification v3 時，每個 Annotation Grid 建立一個 Patch Bag；
   bag 內是 ordered source-coordinate Model Patches，positive truth 只存在 bag，
   不會虛構 individual patch label。
6. v3 scorer 對 Model Patches 產生 logits，以 per-class max pooling 得到 Grid
   logits，再沿用 Grid Evaluation、threshold 與 approval workflow。
7. Approved v3 run 以 dense patch scores 經既有 overlap stitcher 產生同一種
   source-coordinate Defect Confidence Map。
8. Proposal、Review、Proposal-to-Grid conversion 及 CSV／JSON／PNG Export 沿用
   既有契約，沒有 patch-specific schema。

Patch geometry: 128 px size, 64 px stride, max pooling. Checkpoint v1／v2 仍以
原 CAM semantics 載入；v3 checkpoint 是
`wafer_defect_studio.resnet18.v3`，並明載上述 geometry 與 pooling rule。

## 截圖與 machine-readable evidence

下表的 `01`–`09` 全部是 Ticket 29 demo evidence；它們展示 v2/v3 的既有
workflow，不代表 Spatial MIL v4 結果。Ticket 30 gate FAIL，所以本次不新增或
更新官方 v4 截圖。

| 階段 | 輸出 | 證據範圍 |
| --- | --- | --- |
| 1. 匯入 | [`01-import.png`](screenshots/realistic-20mp/01-import.png) | final run 的 1,536×1,536 processing proxy；summary 記錄 20-image corpus。 |
| 2. 兩類標註 | [`02-annotation-two-classes.png`](screenshots/realistic-20mp/02-annotation-two-classes.png) | `scratch`、`particle` Grid Annotation 與 Reviewed truth。 |
| 3. Dataset Snapshot | [`03-dataset-snapshot.png`](screenshots/realistic-20mp/03-dataset-snapshot.png) | final run 的 Snapshot／split UI。 |
| 4. CAM v2 訓練 | [`04-cam-v2-training.png`](screenshots/realistic-20mp/04-cam-v2-training.png) | 同一 matched run 的 CAM v2 completed Training Run。 |
| 5. CAM v2 Grid Evaluation | [`05-cam-v2-grid-evaluation.png`](screenshots/realistic-20mp/05-cam-v2-grid-evaluation.png) | test-only v2 per-class metrics。 |
| 6. Patch v3 訓練 | [`06-patch-v3-training.png`](screenshots/realistic-20mp/06-patch-v3-training.png) | v3、128/64/max immutable request summary。 |
| 7. Patch v3 Grid Evaluation | [`07-patch-v3-grid-evaluation.png`](screenshots/realistic-20mp/07-patch-v3-grid-evaluation.png) | test-only v3 per-class metrics。 |
| 8. Patch v3 map | [`08-patch-v3-confidence-map.png`](screenshots/realistic-20mp/08-patch-v3-confidence-map.png) | direct Detection worker 的單張 `evidence-patch_v3-test-1` artifact 經 value-only confidence viewer 顯示；不是 persisted Profile/Run 畫面。 |
| 9. Patch v3 export | [`09-patch-v3-heatmap-export.png`](screenshots/realistic-20mp/09-patch-v3-heatmap-export.png) | v3 同一 source-coordinate map 的 native-size PNG export。 |
| Machine-readable matched comparison | [`demo-summary.json`](screenshots/realistic-20mp/demo-summary.json) | 2026-05-04 最新重跑的完整 v2/v3 measured record。 |

### Ticket 30 quality gate — real generated held-out RTX 3090 evidence

品質門檻報告為
[`docs/demo/ticket30-quality-gate.json`](ticket30-quality-gate.json)。它要求每個
seed/class 同時達到 150 個 defect instances、1.0 defect coverage recall、至少
0.95 Grid precision/recall、最多 0.05 Normal Grid leakage，以及最多 0.25
asserted-Grid occupancy P95。實際結果是 **Spatial MIL v4 gate FAIL**；因此
CAM v2 remains the default，v4 is experimental。

下列六組數字直接對應 JSON 的 `per_seed` rows；`coverage` =
`defect_coverage_recall`，`precision`/`recall` = Grid metrics，`leakage` =
`normal_grid_leak_rate`，`occupancy` = `asserted_grid_occupancy_p95`。

| Seed | Class | Coverage | Precision | Recall | Leakage | Occupancy P95 | Instances |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 17 | scratch | 0.10666666666666667 | 0.3333333333333333 | 1.0 | 0.25 | 0.08049774169921875 | 150 |
| 17 | particle | 0.47333333333333333 | 0.1111111111111111 | 1.0 | 1.0 | 0.5470352172851562 | 150 |
| 42 | scratch | 0.24666666666666667 | 0.3333333333333333 | 1.0 | 0.25 | 0.16243743896484375 | 150 |
| 42 | particle | 1.0 | 0.1111111111111111 | 1.0 | 1.0 | 0.9989433288574219 | 150 |
| 91 | scratch | 0.44 | 0.3333333333333333 | 1.0 | 0.25 | 0.2965354919433594 | 150 |
| 91 | particle | 1.0 | 0.1111111111111111 | 1.0 | 1.0 | 0.9944877624511719 | 150 |

## Matched GPU 結果

硬體為 RTX 3090，兩個模型使用相同 immutable 16/2/2 split；每個 class 的
validation/test asserted-image support 都是 2/2，`evidence_status` 為
`measured`。

Test-only Grid Evaluation：

- v2: scratch F1 0.4000, particle F1 1.0000, exact Grid match 15/18
- v3: scratch F1 0.2000, particle F1 0.8571, exact Grid match 10/18

Coarse localization 的數字順序是 asserted Grid intersection rate／Normal Grid
leak rate；它不是 pixel IoU：

- v2: scratch intersection/leak 1.0000/1.0000; particle intersection/leak 1.0000/1.0000
- v3: scratch intersection/leak 0.0000/0.3125; particle intersection/leak 0.2500/0.0000

本次 measured evidence 沒有建立品質改善：v3 把 particle Normal Grid leakage
降到 0.0000，但 scratch 仍有 0.3125 leakage；兩類 asserted-Grid intersection、
scratch／particle Grid F1 與 exact Grid match 都比 v2 低。因此 CAM v2 remains
the default; Patch Classification v3 is optional. This generated-data comparison
does not establish segmentation, Neurocle equivalence, or production accuracy.

`demo-summary.json` 也保留兩個 checkpoint version、validation-derived
thresholds，以及 train／evaluation／detection runtime。這些值是單次本機執行
證據；重跑會產生新的 run IDs、checksum、thresholds 與 runtime。

Matched GPU runner 不自動建立 approval decision、Detection Profile 或 persisted
Detection Run；它直接執行 validation/test worker requests 後彙總 evidence。
Approved v3 的正式 UI routing 由 Slice 29.14 的
`tests.test_project_detection_profile` 與 `tests.test_project_detection_launch`
覆蓋。第 08 張只顯示最後一個 test image artifact；跨四張 held-out images 的
aggregate 數值仍以 summary 為準。

## 解讀邊界

- Grid Annotation 是 weak multi-label truth，不是 pixel mask、polygon 或 box。
- Patch Bag 只表示每個 asserted Defect Class 至少由一個 Model Patch 支持，
  不指出是哪一個 patch。
- Defect Confidence Map 與 retained regions 是 source-coordinate approximate
  localization，不是 pixel-accurate boundary。
- Neurocle 公開材料只提供行為參考；本 repo 沒有其專有 architecture、loss、
  patch policy、weights 或相依套件，也不主張等價。
- generated corpus 與 processing proxy 只證明 workflow 和 bounded comparison
  可執行；上線品質仍需獨立真實 wafer corpus 驗證。

要用真實資料驗證，請匯入真實 8/16-bit grayscale Wafer Images，完成 Grid
Profile、Effective Wafer Area、Grid Annotation、Reviewed 與 Dataset Snapshot，
再按 Train → Evaluate → Detect → Review → Export 操作。只有滿足專案 criteria
並由 Algorithm Engineer 明確核准的 run，才可成為 Approved Run。

## 清理與重跑

Demo 使用 `TemporaryDirectory`，執行結束後暫存 SQLite project 自動清除；只有
指定 output 目錄的 `01`–`09` screenshots 與 `demo-summary.json` 保留。重跑
realistic command 會覆寫這組 matched evidence。
