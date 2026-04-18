# Phase 2 MVP 端到端 Demo

這份教學用固定規格的 synthetic wafer images，走過 Phase 2 MVP 的主要
happy path：匯入圖片、建立兩類標註、建立 Dataset Snapshot、以真實 frozen
patch 訓練 ResNet18、用同一個 checkpoint 評估並核准、執行 checkpoint-backed
Detection，最後顯示 Heatmap、Regions 與 Both。synthetic 只描述資料來源，
不是 Training/Evaluation/Detection 的替代流程。

## 執行 Demo

在 repository root 執行：

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
$env:QT_QPA_FONTDIR = "C:/Windows/Fonts"
& 'E:\miniconda3\envs\wafer-defect-studio\python.exe' docs/demo/run_phase2_demo.py
```

程式會在暫存目錄建立 project，不會修改既有專案；截圖、heatmap 與摘要會寫
入 [`docs/demo/screenshots/`](screenshots/)。也可以指定輸出目錄：

```powershell
& 'E:\miniconda3\envs\wafer-defect-studio\python.exe' docs/demo/run_phase2_demo.py `
  --output .\artifacts\phase2-demo
```

## 每一階段看什麼

| 階段 | 輸出 | 驗收重點 |
| --- | --- | --- |
| 1. 匯入 | [`01-import.png`](screenshots/01-import.png) | Data workspace 顯示 128×96 的 uint8 PNG、Grid Profile 與 Effective Area。 |
| 2. 兩類標註 | [`02-annotation-two-classes.png`](screenshots/02-annotation-two-classes.png) | Annotate workspace 顯示 `scratch`、`particle` 兩個 class，Review 顯示 `Labeled: 2`。 |
| 3. Dataset Snapshot | [`03-dataset-snapshot.png`](screenshots/03-dataset-snapshot.png) | Snapshot 與 deterministic split 已建立，Training scope 顯示 `demo-line`。 |
| 4. 訓練 | [`04-training-complete.png`](screenshots/04-training-complete.png) | Train workspace 顯示 ResNet18、CPU、1 epoch、`Status: Completed`。 |
| 5. 評估與核准 | [`05-evaluation-approved.png`](screenshots/05-evaluation-approved.png) | Evaluate workspace 顯示 Macro F1、thresholds，以及 Candidate → Validated → Approved decision history。 |
| 6. Detection | [`06-detection-controls.png`](screenshots/06-detection-controls.png) | Approved Evaluation、Detection Profile、Detection Run 與 source-pixel map context 已接上。 |
| 7. Heatmap | [`06-heatmap.png`](screenshots/06-heatmap.png) | Detect canvas 的原圖尺寸 class confidence map；另有 [`06-heatmap-export.png`](screenshots/06-heatmap-export.png) PNG export。 |
| 8. Regions | [`07-regions.png`](screenshots/07-regions.png) | 顯示同一個 selected-class retained region map；source pixel 對齊，不是 segmentation mask。 |
| 9. Both | [`08-both.png`](screenshots/08-both.png) | Heatmap 與 retained regions 疊加，使用相同 profile、threshold 與 opacity。 |
| 10. 標註比對 | [`demo-summary.json`](screenshots/demo-summary.json) | 把 Detection Proposals 映射回 participating Annotation Grid，計算 per-class TP/FP/FN、precision/recall/F1 與 exact-cell match。 |

完整的 machine-readable 結果在 [`demo-summary.json`](screenshots/demo-summary.json)。
最近一次本機執行的摘要包含：ResNet18 / CPU / 1 epoch、checkpoint format
`wafer_defect_studio.resnet18.v1`、兩類順序 `scratch → particle`、Detection
map method `all_convolutional_sigmoid`、map shape `[96, 128, 2]`，以及每次
執行產生的 `model.pt` SHA-256。摘要中的 Evaluation 與 annotation comparison
數字是實測值；重新執行會因新 run/checksum 而更新，不能當成 production accuracy。

## 目前 MVP 的解讀邊界

這個 runner 的目的，是驗證專案資料、服務、worker 和 GUI 的串接，而不是
宣稱模型已具備真實 wafer 的準確率：

- 匯入的是程式產生的 8-bit synthetic wafer。
- Demo 會建立 10 個 source images，讓 deterministic split 同時有 train、
  validation、test；每個 image 的兩個 Grid Annotation 都會真的寫入 SQLite，
  類別是 `scratch` 與 `particle`，並標記 Reviewed。
- Training worker 讀取 Snapshot-backed `training_input_bundle.json`，以真實
  ResNet18 訓練並發佈 `model.pt`；runner 會拒絕缺 checkpoint 或 synthetic
  artifact 的結果。
- Evaluation worker 從已發佈 checkpoint score test split，再計算實際
  `y_true`/`y_score` metrics；它不接受 runner 內嵌的受控矩陣。
- Detection 由 Approved Evaluation 綁定的 checkpoint 產生 layer4+FC
  1×1 convolution、sigmoid local maps，再用既有 overlap stitcher。runner
  會拒絕 `map_method` 不是 `all_convolutional_sigmoid` 的 artifact。
- `annotation_validation` 是實際把 Proposal rectangle 與 participating
  Annotation Grid 做交集後的比對；這次低 precision 表示目前 Detection 結果
  尚未符合兩個人工標註的範圍。
- Demo Detection Profile 的 threshold 設為 `0.0`，只為讓 Regions/Both 的
  source-coordinate overlay 在截圖中可見；這不是 threshold tuning 或品質結論。
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
