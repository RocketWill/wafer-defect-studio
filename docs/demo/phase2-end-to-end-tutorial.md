# Phase 2 MVP 端到端 Demo

這份教學走過 Phase 2 MVP 的主要 happy path：匯入圖片、建立兩類標註、建立
Dataset Snapshot、以真實 frozen patch 訓練 ResNet18、用同一個 checkpoint
評估並核准、執行 checkpoint-backed Detection，最後顯示 Heatmap、Regions
與 Both。推薦使用本 repo 附帶的 generated 20MP wafer source；它是合理的
demo 視覺素材，不是真實量測資料，也不構成模型準確率證據。

## 執行 Demo

在 repository root 執行 GPU realistic-source Demo：

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
$env:QT_QPA_FONTDIR = "C:/Windows/Fonts"
& 'E:\miniconda3\envs\wafer-defect-studio\python.exe' docs/demo/run_phase2_demo.py `
  --source-image docs/demo/assets/realistic-wafer-20mp.png `
  --output .\docs\demo\screenshots\realistic-20mp
```

程式會在暫存目錄建立 project，不會修改既有專案；截圖、heatmap 與摘要會寫
入 [`docs/demo/screenshots/realistic-20mp/`](screenshots/realistic-20mp/)。
20MP 輸入是 4,472×4,472（19,998,784 pixels）；為避免現有 JSON map artifact
在 Demo 中產生數百 MB 序列化負擔，runner 會使用 1,536×1,536 processing
geometry 建立十張 deterministic generated wafer bases，並在 summary 保留輸入
尺寸。這些 bases 不是輸入圖的重複註冊；train、validation、test 按 Wafer Image
隔離。realistic source 會要求 CUDA。

若只要快速驗證 UI 串接，也可以省略 `--source-image` 使用小型 synthetic source：

```powershell
& 'E:\miniconda3\envs\wafer-defect-studio\python.exe' docs/demo/run_phase2_demo.py `
  --output .\artifacts\phase2-demo-synthetic
```

## 每一階段看什麼

| 階段 | 輸出 | 驗收重點 |
| --- | --- | --- |
| 1. 匯入 | [`01-import.png`](screenshots/realistic-20mp/01-import.png) | 1,536×1,536 processing proxy、512px Grid Profile 與 Effective Area；summary 另記 4,472×4,472 原圖。 |
| 2. 兩類標註 | [`02-annotation-two-classes.png`](screenshots/realistic-20mp/02-annotation-two-classes.png) | Annotate workspace 顯示 `scratch`、`particle` 兩個 class，並以 Grid Annotation 寫入 Reviewed truth。 |
| 3. Dataset Snapshot | [`03-dataset-snapshot.png`](screenshots/realistic-20mp/03-dataset-snapshot.png) | Snapshot 與 deterministic split 已建立，Training scope 顯示 `demo-line`。 |
| 4. 訓練 | [`04-training-complete.png`](screenshots/realistic-20mp/04-training-complete.png) | Train workspace 顯示 ResNet18、CUDA、20 epochs、`Status: Completed`。 |
| 5. 評估與核准 | [`05-evaluation-approved.png`](screenshots/realistic-20mp/05-evaluation-approved.png) | Evaluate workspace 顯示 Macro F1、thresholds，以及 Candidate → Validated → Approved decision history。 |
| 6. Detection | [`06-detection-controls.png`](screenshots/realistic-20mp/06-detection-controls.png) | Approved Evaluation、Detection Profile、Detection Run 與 source-pixel map context 已接上。 |
| 7. Heatmap | [`06-heatmap.png`](screenshots/realistic-20mp/06-heatmap.png) | Detect canvas 的 1,536×1,536 class confidence map；另有 [`06-heatmap-export.png`](screenshots/realistic-20mp/06-heatmap-export.png) PNG export。 |
| 8. Regions | [`07-regions.png`](screenshots/realistic-20mp/07-regions.png) | 使用 0.5 high-confidence display floor 後，顯示局部 retained region；仍是 approximate localization，不是 segmentation mask。 |
| 9. Both | [`08-both.png`](screenshots/realistic-20mp/08-both.png) | Heatmap 與 retained regions 疊加，使用相同 profile、threshold 與 opacity。 |
| 10. 標註比對 | [`demo-summary.json`](screenshots/realistic-20mp/demo-summary.json) | 把 Detection Proposals 映射回 participating Annotation Grid，計算 per-class TP/FP/FN、precision/recall/F1 與 exact-cell match。 |

完整的 machine-readable 結果在
[`realistic-20mp/demo-summary.json`](screenshots/realistic-20mp/demo-summary.json)。
本次實測摘要為 ResNet18 / CUDA / 20 epochs、checkpoint format
`wafer_defect_studio.resnet18.v2`（feature stride 16）、兩類順序
`scratch → particle`、Detection
map method `all_convolutional_sigmoid`、map shape `[1536, 1536, 2]`。Evaluation
Macro F1 為 `0.5`；annotation comparison Macro F1 為 `0.5152`、exact-cell
match 為 `2/9`。Scratch 的 Grid comparison 為 TP `1`、FP `1`、FN `0`，但
held-out Evaluation scratch F1 仍為 `0.0`。這些數字是本次 pipeline evidence，
重新執行會因新 run/checksum 而更新，不能當成 production accuracy。

## 目前 MVP 的解讀邊界

這個 runner 的目的，是驗證專案資料、服務、worker 和 GUI 的串接，而不是
宣稱模型已具備真實 wafer 的準確率：

- realistic input 是生成並放大的 20MP grayscale 尺寸參考，不是真實 wafer
  量測；runner 依其 processing geometry 建立獨立 generated corpus。
- Demo 會建立 10 個 source images，讓 deterministic split 同時有 train、
  validation、test。八張 train bases 中七張包含不同位置／角度的 scratch，
  一張是 scratch hard negative；validation/test 使用未重複的 generated bases。
  每個 image 的 Grid Annotation 都會真的寫入 SQLite，類別是 `scratch` 與
  `particle`，並標記 Reviewed。
- Training worker 讀取 Snapshot-backed `training_input_bundle.json`，以真實
  ResNet18 訓練並發佈 `model.pt`；runner 會拒絕缺 checkpoint 或 synthetic
  artifact 的結果。
- Evaluation worker 從已發佈 checkpoint score test split，再計算實際
  `y_true`/`y_score` metrics；它不接受 runner 內嵌的受控矩陣。
- Detection 由 Approved Evaluation 綁定的 checkpoint 產生 layer4+FC
  1×1 convolution、sigmoid local maps，再用既有 overlap stitcher。runner
  會拒絕 `map_method` 不是 `all_convolutional_sigmoid` 的 artifact。
- `annotation_validation` 是實際把 Proposal rectangle 與 participating
  Annotation Grid 做交集後的比對；本次 `2/9` exact-cell match 與 Macro F1
  `0.5152` 表示 scratch localization 已縮小，但模型仍未形成可靠的兩類泛化。
- realistic Demo Profile 使用 Evaluation threshold 的 positive 值，並把
  Regions/Both 的顯示下限提高到 `0.5`，以避免零 threshold 造成整片 overlay；
  這是 Demo 可視化門檻，不是 production threshold tuning。
- Detection/CAM 與 retained regions 是 approximate weak localization。Heatmap
  與 Regions 都不是 segmentation mask；目前不應把這個 synthetic demo 當成
  可部署模型品質。

要用真實資料驗證，請在應用程式中建立或開啟 project，匯入真實 8/16-bit
grayscale wafer，完成 Grid/Effective Area/Reviewed 與 Dataset Snapshot，然後
以相同的 Train → Evaluate → Detect → Review → Export 順序操作。

## 清理與重跑

Demo 使用 `TemporaryDirectory`，執行結束後暫存 SQLite project 會自動清除；
只有指定的 screenshots 與 summary 會保留。重新執行會覆寫同名輸出，方便在
程式或 UI 有變更時更新文件證據。
