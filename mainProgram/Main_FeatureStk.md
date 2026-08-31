# Main_FeatureStk.py 執行步驟總覽

路徑基準：腳本在 `mainProgram/` 下執行，所以 `../data/...` 對應到專案根目錄的 `data/`。`dataName = 'NeuroP_1'`，`normalizeMethod` 會跑 `standard`、`robust` 兩輪。

## 0. 初始化（不產生檔案）
- 設定 matplotlib 為 `Agg`（不彈窗）、把上層目錄加入 `sys.path`。
- 定義 `featureDict`（iFeature/pFeature/ampFeature/ovpFeature/centerGDPFeature 五大類特徵開關）。

## 1. 讀取原始 fasta 序列（不產生檔案）
讀入 `MainDatasetNeg/Pos.fasta`、`DS_IndpNeg/Pos.fasta` 四個檔案為序列字典。

## 2. 儲存 fasta 序列數量統計 — 受 `b_saveFastaSeqCountStat` 控制
- **產生**：`data/fasta_seq_count.csv`
- 內容：先寫入 MainDatasetNeg/Pos、DS_IndpNeg/Pos 四筆數量（`'w'`），之後在切分完 DS_Train/DS_Val 後再各補 4 筆（`'a'` 續寫，同一份檔案）。

## 3. 切分 MainDataset → DS_Train / DS_Val（9:1，不產生檔案）
用 `train_test_split` 對 Neg/Pos 各自切分，維持類別比例。

## 4. 儲存三個資料集的原始序列 csv（一定執行，不受開關控制）
- **產生**：
  - `data/featureStat/DS_Train.csv`
  - `data/featureStat/DS_Indp.csv`
  - `data/featureStat/DS_Val.csv`
- 欄位：`name, sequence, label`

## 5. 統計啟用的 feature type 數量（三個階段，不產生檔案，只 print）
1. 原始 `featureDict`（count）
2. `buildOVPC_GAAC_formulaFeatureDict()`：把 OVPC(ovpFeature)、GAAC(iFeature)、formula(ampFeature) 三個開關關掉，合併成 `mergedFeature.OVPC,GAAC,formula` 一個開關（count2）
3. `buildSingleValueCombineFeatureDict()`：再把 27 個單一數值理化性質特徵關掉，合併成 `mergedFeature.<27個名稱>` 一個開關（count3）

> 之後所有下游步驟用的都是「合併過兩次」後的最終 `featureDict`。

## 6. 用少量樣本探索每個 feature type 的欄位名稱（不產生檔案）
- 取 5 筆序列（`columnDiscoverySampleDataDict`），逐一單獨開啟每個真正啟用的 feature type 做 encode，建立：
  - `featureTypeColumnMap`：feature type → 欄位名稱清單
  - `columnToFeatureType`：欄位名稱 → 所屬 feature type（供之後過濾結果回查用）
- 因為 `mergedFeature` 這個 key encode 不出東西，改用寫死的 `OVPC_GAAC_FORMULA_COLUMN_LIST` / `SINGLE_VALUE_COMBINE_COLUMN_LIST` 取代。

## 7. 輸出 feature type 明細表（橫向）
- **產生**：`data/featureStat/featureType_featureName_table_NeuroP_1.csv`
- 內容：第一列是各 feature type 名稱（如 AAC、CKSAAGP…，mergedFeature 排最後），第二列是各自的欄位數（feature size）。

## 8. 輸出三階段 feature type 數量統計
- **產生**：`data/featureStat/featureTypeStatistics.csv`
- 內容：`origin featureDict` / `after merged OVPC GAAC formula` / `after merged all single value` 三列的 count 數量。

## 9. 儲存最終 `featureDict` 設定
- **產生**：`data/param/NeuroP_1_featureTypeDict.json`
- 之後 `Main_MLStkLv1.py` 會讀這份 json 還原完整特徵設定。

## 10. 對 DS_Train / DS_Indp / DS_Val 做完整特徵編碼
- **產生**：
  - `data/featureStat/encode_NeuroP_1_DS_Train.csv`
  - `data/featureStat/encode_NeuroP_1_DS_Indp.csv`
  - `data/featureStat/encode_NeuroP_1_DS_Val.csv`
- 這是編碼完、**尚未 normalize** 的完整特徵資料。

## 11. Normalization + Feature 過濾（`for normalizeMethod in ['standard', 'robust']`，以下每輪都各產生一份）

| 步驟 | 做的事 | 產生檔案 |
|---|---|---|
| 11.1 正規化 | train 算 scaler 並存檔；indp/val 套用同一 scaler | `data/param/NeuroP_1_{method}Scaler.pkl` |
| 11.2 儲存正規化後資料 | — | `data/featureStat/train_NeuroP_1_{method}.csv`<br>`data/featureStat/indp_NeuroP_1_{method}.csv`<br>`data/featureStat/val_NeuroP_1_{method}.csv` |
| 11.3 標準差分析 | `sdAnalysis`：畫圖分析各 feature 標準差區間分布 | `data/featureStat/sd_analysis_NeuroP_1_{method}.jpg` |
| 11.4 數值集中度分析 | `featureValuePct_analysis`：算每個欄位 top1/top2/top1percent 等指標 | `data/featureStat/featureAnalysis_NeuroP_1_{method}.xlsx` |
| 11.5 過濾特徵 | `processData`：`top1percent > 0.98` 的欄位視為數值過度集中而移除（`MotifBitVec` 系列受保護不濾） | 產生 `removeList`（記憶體） |
| 11.6 過濾 log | `processDataLog` | `data/mlData/NeuroP_1_{method}_processLog_.txt` |
| 11.7 儲存過濾後資料 | train 過濾；indp/val 依同一 `removeList` 同步 drop | `data/featureStat/filtered_train_NeuroP_1_{method}.csv`<br>`data/featureStat/filtered_indp_NeuroP_1_{method}.csv`<br>`data/featureStat/filtered_val_NeuroP_1_{method}.csv` |
| 11.8 被濾掉的欄位清單 | — | `data/featureStat/remove_feature_list_NeuroP_1_{method}.json` |
| 11.9 被濾掉欄位依 feature type 分組 | 用 `columnToFeatureType` 回查所屬 feature type | `data/featureStat/remove_featureType_list_NeuroP_1_{method}.json` |
| 11.10 過濾前後數量摘要 | 受 `b_saveFeatureTypeFilterSummary` 控制：每個 feature type 過濾前/後欄位數，`type` 欄合併置中 | `data/featureStat/featureType_filterSummary_NeuroP_1_{method}.xlsx` |

---

## 四個 bool 開關的影響範圍
- `disablePlotPopup`／`useVscodeParentPath`：環境設定，不影響輸出檔案。
- `b_saveFastaSeqCountStat = False` → 不會產生/寫入 `fasta_seq_count.csv`。
- `b_saveFeatureTypeFilterSummary = False` → 不會產生兩份 `featureType_filterSummary_..._{method}.xlsx`。

其餘所有檔案（DS_Train/Indp/Val csv、featureTypeDict.json、encode csv、normalize csv、scaler pkl、featureAnalysis xlsx、filtered csv、remove_feature(Type)_list json 等）目前都沒有開關保護，每次執行都會產生。
