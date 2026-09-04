# ======================================================================================================================
# 所有功能開關 (bool)，統一放在檔案最前面方便控制
disablePlotPopup = True  # 是否禁止彈出圖表視窗
useVscodeParentPath = True  # 使用pycharm, vscode進行編譯請開啟
# ======================================================================================================================

import matplotlib

def setMatplotlibBackend(disablePopup):
    """disablePopup=True: 使用Agg backend，禁止彈出圖表視窗；disablePopup=False: 維持預設backend，正常跳出視窗"""
    if disablePopup:
        matplotlib.use('Agg')

setMatplotlibBackend(disablePlotPopup)

import sys, os

def setupParentPath(enableVscodeParentPath):
    """
    enableVscodeParentPath=True:
    使用Pycharm, Vscode進行編譯時，把上一層目錄加入sys.path，方便import套件
    """
    if enableVscodeParentPath:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent = os.path.join(current_dir, "..")
        sys.path.append(parent)

setupParentPath(useVscodeParentPath)

import pandas as pd
from userPackage.Package_Encode import EncodeAllFeatures

# ======================================================================================================================
# 基本路徑：直接讀取 Main_MLStkLv1_debug_test.py 存的 Meta-Feature-Matrix（DS_Val 預測機率表）當作 Lv2 的訓練資料，
# 對每一欄 (feature type × model) 的機率做 Boruta 特徵排序，篩選出真正有用的 meta-feature 組合。
# 參考 main_Feature_v2.py 的 Boruta 流程（encodeObj.dataBoruta(...)），只做排序，不做 dataEvalFeatureNum 那段
# feature 數量掃描（那段需要同時有 train/indp 兩份資料，Lv1 debug_test 目前只產生 DS_Val 的 Meta-Feature-Matrix）。
mlDataPath = "../data/mlData/"  # Boruta 排序結果 Boruta-featureRank-{method}.csv 存這裡
mlScorePath = "../data/mlScore/"  # 內含 Main_MLStkLv1_debug_test.py 存的 Meta-Feature-Matrix_{dataName}_test_{normalizeMethod}.csv
dataName = 'NeuroP_1'

# 要跟 Main_FeatureStk_debug.py / Main_MLStkLv1_debug_test.py 的 normalizeMethodList 保持一致，
# 否則會去讀一份不存在的 Meta-Feature-Matrix_{dataName}_test_{normalizeMethod}.csv
normalizeMethodList = ['standard']

borutaMethod = 'XGB'  # Boruta 底層估計器：'XGB' / 'RF' / 'LGB'，跟 main_Feature_v2.py 一致用 XGB

os.makedirs(mlDataPath, exist_ok=True)

encodeObj = EncodeAllFeatures()

for normalizeMethod in normalizeMethodList:
    metaFeatureMatrixPath = mlScorePath + f'Meta-Feature-Matrix_{dataName}_test_{normalizeMethod}.csv'
    metaFeatureMatrixDf = pd.read_csv(metaFeatureMatrixPath, index_col=[0])
    print(f"[{normalizeMethod}] 讀取 Lv1 Meta-Feature-Matrix：{metaFeatureMatrixDf.shape}")

    # 每一欄都是某個 (feature type × model) 的預測機率，全部都要參與 Boruta 排序，沒有需要保護、跳過的欄位
    skipFeatureList = []
    featRankPrefix = mlDataPath + f'Lv2_{dataName}_{normalizeMethod}_'
    brtObj = encodeObj.dataBoruta(borutaMethod=borutaMethod, runBoruta=True, featRankPath=featRankPrefix,
                                  trainDf=metaFeatureMatrixDf, skipFeatureList=skipFeatureList)

    featRankCsvPath = featRankPrefix + f'Boruta-featureRank-{borutaMethod}.csv'
    print(f"[{normalizeMethod}] Boruta 排序結果（{len(brtObj.feature_sort)} 個 meta-feature）已儲存到 {featRankCsvPath}")
    print(brtObj.feature_sort)

    # ==================================================================================================================
    # dataEvalFeatureNum：依 Boruta 排序，掃描不同 meta-feature 數量（startNum~endNum，每次間隔 step）分別做一次
    # CV 比較，把每個數量下最好的分數存到 bestScore_mcc.csv / bestScore_auc.csv，方便挑選最終要用幾個 meta-feature。
    # 需要同時有 train（metaFeatureMatrixDf）跟 indp 兩份 Meta-Feature-Matrix：indp 版本目前 Main_MLStkLv1_debug_test.py
    # 還沒有產生，要先讓 Lv1 也對 DS_Indp 做一次預測、存成 Meta-Feature-Matrix_{dataName}_test_indp_{normalizeMethod}.csv，
    # 這裡才讀得到，否則下面這行會丟 FileNotFoundError。
    indpMetaFeatureMatrixPath = mlScorePath + f'Meta-Feature-Matrix_{dataName}_test_indp_{normalizeMethod}.csv'
    indpMetaFeatureMatrixDf = pd.read_csv(indpMetaFeatureMatrixPath, index_col=[0])

    encodeObj.dataEvalFeatureNum(startNum=5, endNum=len(brtObj.feature_sort), step=5,
                                 featNumScorePath=featRankPrefix, saveCsvPath=featRankPrefix,
                                 trainDf=metaFeatureMatrixDf, indpDf=indpMetaFeatureMatrixDf,
                                 brtObj=brtObj, foldNum=5, session=None)  # sessionID可修改成任意整數，ex:1,4,10,15...

    # encodeObj.dataDecidedFeatureNum(featureNum=20, saveCsvPath=featRankPrefix,
    #                                 trainDf=metaFeatureMatrixDf, indpDf=indpMetaFeatureMatrixDf,
    #                                 brtObj=brtObj)  # 決定好 feature 數字請開這個
