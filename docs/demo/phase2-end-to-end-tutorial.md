# Phase 2 MVP 端到端 Demo

這份教學用一個可重跑的 synthetic wafer，走過 Phase 2 MVP 的主要 happy
path：匯入圖片、建立兩類標註、建立 Dataset Snapshot、訓練 ResNet18、評估
並核准、執行 Detection，最後輸出 native-coordinate heatmap。

## 執行 Demo

在 repository root 執行：

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
$env:QT_QPA_FONTDIR = "C:/Windows/Fonts"
python docs/demo/run_phase2_demo.py
```

程式會在暫存目錄建立 project，不會修改既有專案；截圖、heatmap 與摘要會寫
入 [`docs/demo/screenshots/`](screenshots/)。也可以指定輸出目錄：

```powershell
python docs/demo/run_phase2_demo.py --output .\artifacts\phase2-demo
```

## 每一階段看什麼

| 階段 | 輸出 | 驗收重點 |
| --- | --- | --- |
| 1. 匯入 | [`01-import.png`](screenshots/01-import.png) | Data workspace 顯示 512×384 的 uint8 PNG、Grid Profile 與 Effective Area。 |
| 2. 兩類標註 | [`02-annotation-two-classes.png`](screenshots/02-annotation-two-classes.png) | Annotate workspace 顯示 `scratch`、`particle` 兩個 class，Review 顯示 `Labeled: 2`。 |
| 3. Dataset Snapshot | [`03-dataset-snapshot.png`](screenshots/03-dataset-snapshot.png) | Snapshot 與 deterministic split 已建立，Training scope 顯示 `demo-line`。 |
| 4. 訓練 | [`04-training-complete.png`](screenshots/04-training-complete.png) | Train workspace 顯示 ResNet18、CPU、1 epoch、`Status: Completed`。 |
| 5. 評估與核准 | [`05-evaluation-approved.png`](screenshots/05-evaluation-approved.png) | Evaluate workspace 顯示 Macro F1、thresholds，以及 Candidate → Validated → Approved decision history。 |
| 6. Detection | [`06-detection-controls.png`](screenshots/06-detection-controls.png) | Approved Evaluation、Detection Profile、Detection Run 與 source-pixel map context 已接上。 |
| 7. Heatmap | [`06-heatmap.png`](screenshots/06-heatmap.png) | 輸出為原圖尺寸 512×384 的 class confidence map，含 grid 與 wafer-area overlay。 |

完整的 machine-readable 結果在 [`demo-summary.json`](screenshots/demo-summary.json)。
最近一次本機執行的摘要包含：ResNet18 / CPU / 1 epoch、Evaluation Macro F1
`1.0` 且 target satisfied、Detection map shape `[384, 512, 2]`。

## 目前 MVP 的解讀邊界

這個 runner 的目的，是驗證專案資料、服務、worker 和 GUI 的串接，而不是
宣稱模型已具備真實 wafer 的準確率：

- 匯入的是程式產生的 8-bit synthetic wafer。
- 兩個 Grid Annotation 會真的寫入 SQLite，類別是 `scratch` 與 `particle`，
  並標記 Reviewed。
- 現有 `training_worker` 是 deterministic synthetic training worker；本
  demo 的 1 epoch 用來確認 Training Run、checkpoint 與 UI 狀態流，不代表已
  用真實標註 patch 完成模型品質驗證。
- Evaluation worker 使用 demo 的受控 `y_true` / `y_score`，所以 Macro F1
  `1.0` 只代表流程與 approval gate 通過，不是泛化能力報告。
- Detection/CAM 是目前 MVP 的 approximate weak localization。Heatmap 是
  native-coordinate confidence visualization，不是 segmentation mask。

要用真實資料驗證，請在應用程式中建立或開啟 project，匯入真實 8/16-bit
grayscale wafer，完成 Grid/Effective Area/Reviewed 與 Dataset Snapshot，然後
以相同的 Train → Evaluate → Detect → Review → Export 順序操作。

## 清理與重跑

Demo 使用 `TemporaryDirectory`，執行結束後暫存 SQLite project 會自動清除；
只有指定的 screenshots 與 summary 會保留。重新執行會覆寫同名輸出，方便在
程式或 UI 有變更時更新文件證據。
