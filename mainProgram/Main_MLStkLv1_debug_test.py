# ======================================================================================================================
# 所有功能開關 (bool)，統一放在檔案最前面方便控制
disablePlotPopup = True  # 是否禁止彈出圖表視窗
useVscodeParentPath = True  # 使用pycharm, vscode進行編譯請開啟
tune_model = True  # True: 針對每個 normalizeMethod × feature type × model 做一對一 tune 並存檔；False: 跳過訓練，直接讀取已存的 finalized model 做 predict
# ======================================================================================================================
# 測試用開關：跟 Main_MLStkLv1_debug.py 功能完全相同，只差在把 feature type 與 model 數量都縮小方便快速測試
testFeatureTypeCount = 5  # 只取 5 個 feature type（兩個 mergedFeature 合併項目一定會被包含在內）
testModelCount = 5  # 只取 5 個 model（lightgbm、xgboost 一定會被包含在內）
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
from MLProcess.PycaretWrapper import PycaretWrapper
from MLProcess.Predict import Predict

# ======================================================================================================================
# 基本路徑：直接接 Main_FeatureStk_debug.py 產生的結果，不重新推導 feature type/欄位
# Main_FeatureStk_debug.py 對每個 normalizeMethod 都會把 feature filter 完的資料依 feature type 拆成
# featureStatPath/featureType_csv_{dataName}_{normalizeMethod}_filtered/DS_Train(或 DS_Val)/{typeName}.csv，
# 檔名就是 feature type 名稱（mergedFeature 底下的合併項目已經是簡短名稱，如 mergedFeature.SingleValueCombine），
# 這裡直接列出資料夾裡的檔案逐一讀取、訓練，不用重新讀 featureTypeDict.json 或重算欄位對照表。
featureStatPath = '../data/featureStat/'
mlScorePath = "../data/mlScore/"  # 內含 ml model 預測完並算好分的檔案（測試結果會加上 _test 後綴，跟正式結果分開）
tuneModelPath = "../data/tuneModel_test/"  # 測試專用資料夾，避免覆蓋 Main_MLStkLv1_debug.py 存的正式 finalized model
dataName = 'NeuroP_1'

# Main_FeatureStk_debug.py 目前 normalizeMethodList 只有 'standard'，這裡要跟它保持一致，
# 否則會去讀一個 Main_FeatureStk_debug.py 根本沒產生的 featureType_csv_{dataName}_{normalizeMethod}_filtered 資料夾
normalizeMethodList = ['standard']

modelNameList = ['lightgbm', 'catboost', 'rbfsvm', 'gbc', 'ridge', 'lr', 'lda', 'ada', 'knn', 'nb', 'et', 'rf',
                 'xgboost', 'mlp', 'dt', 'svm', 'qda']

# 測試用：只取 testModelCount 個 model，lightgbm 跟 xgboost 一定要包含在內
requiredModelNameList = ['lightgbm', 'xgboost']
otherModelNameList = [modelName for modelName in modelNameList if modelName not in requiredModelNameList]
modelNameList = (requiredModelNameList + otherModelNameList)[:testModelCount]
print(f"[test] 只使用 {len(modelNameList)} 個 model 進行測試: {modelNameList}")

os.makedirs(mlScorePath, exist_ok=True)
resultRows = []

# ======================================================================================================================
# 每一個 normalize 方式 × 每一個 feature type × 每一個 model 做一對一訓練（Optuna TPE tune，5-fold CV，用 MCC 當優化目標）
for normalizeMethod in normalizeMethodList:
    filteredSplitDir = featureStatPath + f'featureType_csv_{dataName}_{normalizeMethod}_filtered/'
    trainSplitDir = filteredSplitDir + 'DS_Train/'
    valSplitDir = filteredSplitDir + 'DS_Val/'

    # 檔名（去掉 .csv）就是 feature type 名稱，train / val 兩邊都要有才拿來訓練
    trainTypeFileSet = {f for f in os.listdir(trainSplitDir) if f.endswith('.csv')}
    valTypeFileSet = {f for f in os.listdir(valSplitDir) if f.endswith('.csv')}
    typeFileNameList = sorted(trainTypeFileSet & valTypeFileSet)
    missingInVal = trainTypeFileSet - valTypeFileSet
    if missingInVal:
        print(f"[提醒] {normalizeMethod}：{len(missingInVal)} 個 feature type 只有 DS_Train 有檔案、DS_Val 沒有，已跳過: {sorted(missingInVal)}")
    print(f"[{normalizeMethod}] 讀到 {len(typeFileNameList)} 個 feature type 的過濾後資料：{trainSplitDir}")

    # 測試用：只取 testFeatureTypeCount 個 feature type，兩個 mergedFeature 合併項目一定要包含在內
    mergedTypeFileNameList = [f for f in typeFileNameList if f.startswith('mergedFeature')]
    otherTypeFileNameList = [f for f in typeFileNameList if not f.startswith('mergedFeature')]
    typeFileNameList = (mergedTypeFileNameList + otherTypeFileNameList)[:testFeatureTypeCount]
    print(f"[test][{normalizeMethod}] 只使用 {len(typeFileNameList)} 個 feature type 進行測試: {typeFileNameList}")

    # 用任一份 DS_Val 檔案先確定 index/y，組出 Meta-Feature-Matrix 骨架
    firstValDf = pd.read_csv(valSplitDir + typeFileNameList[0], index_col=[0])
    metaFeatureMatrix = pd.DataFrame(index=firstValDf.index)
    metaFeatureMatrix['y'] = firstValDf['y']

    for typeFileName in typeFileNameList:  # 一次挑一種 Feature Type
        typeName = typeFileName[:-len('.csv')]

        subTrainDf = pd.read_csv(trainSplitDir + typeFileName, index_col=[0])
        subValDf = pd.read_csv(valSplitDir + typeFileName, index_col=[0])
        subVal_X = subValDf.drop(columns=['y'])  # DS_Val 同一組欄位，只留 X 給 predict 用
        print(f"[{normalizeMethod}] {typeName}: train={subTrainDf.shape}, val_X={subVal_X.shape}")

        # 每個 normalizeMethod + featureType 的 model 各自存在獨立資料夾，避免互相覆蓋
        comboSavePath = os.path.join(tuneModelPath, normalizeMethod, typeName.replace('.', '_'))
        os.makedirs(comboSavePath, exist_ok=True)

        pycObj = PycaretWrapper()
        if tune_model:
            pycObj.doSetup(trainData=subTrainDf, sessionID=42)

        for modelName in modelNameList:  # 每個 model 逐一嘗試（feature type × model 一對一訓練）
            try:
                if tune_model:
                    # 進行一對一訓練
                    pycObj.doTuneModel(searchLibrary='optuna', searchAlg='tpe',
                                       includeModelList=[modelName], foldNum=5,
                                       n_iter=10, early_stopping=False, customGridDict=None)
                    _, scoreRank = pycObj.doCompareModel(fold=5, includeModelList=pycObj.tunedModelList)  # 對 tune 好的 model 重新做一次 CV 拿分數表
                    mccScore = scoreRank['MCC'].iloc[0]
                    print(f"[完成] {normalizeMethod} + {typeName} + {modelName}: MCC = {mccScore}")
                    resultRows.append({'normalizeMethod': normalizeMethod, 'featureType': typeName,
                                       'model': modelName, 'mcc': mccScore})

                    pycObj.doFinalizeModel()  # train + self test 合併重新 fit，存起來給之後直接讀取用
                    pycObj.doSaveModel(comboSavePath, b_isFinalizedModel=True)  # 每個 model 各自存檔
                    predictModelList = pycObj.finalModelList
                else:
                    # tune_model=False：跳過訓練，直接讀取已存的 finalized model
                    predictModelList = pycObj.doLoadModel(comboSavePath, fileNameList=[modelName],
                                                           b_isFinalizedModel=True)

                # 用這個 model 對 DS_Val 做 predict，寫進 Meta-Feature-Matrix（機率表）
                predObj = Predict(dataX=subVal_X, modelList=predictModelList)
                _, probVectorList = predObj.doPredict()
                metaFeatureMatrix[f'{typeName}.{modelName}'] = probVectorList[0]
                print(f"[predict] {normalizeMethod} + {typeName} + {modelName} 已寫入 Meta-Feature-Matrix")
            except Exception as e:
                actionName = '訓練' if tune_model else '讀取/predict'
                print(f"[跳過] {normalizeMethod} + {typeName} + {modelName} {actionName}失敗: {e}")
                if tune_model:
                    resultRows.append({'normalizeMethod': normalizeMethod, 'featureType': typeName,
                                       'model': modelName, 'mcc': None, 'error': str(e)})

    metaFeatureMatrixPath = mlScorePath + f'Meta-Feature-Matrix_{dataName}_test_{normalizeMethod}.csv'
    metaFeatureMatrix.to_csv(metaFeatureMatrixPath)
    print(f"[{normalizeMethod}] DS_Val 機率表 (Meta-Feature-Matrix) 已儲存到 {metaFeatureMatrixPath}")

if tune_model:
    resultDf = pd.DataFrame(resultRows)
    resultCsvPath = mlScorePath + f'featureType_model_mccScore_{dataName}_test.csv'
    resultDf.to_csv(resultCsvPath, index=False)
    print(f"每個 normalizeMethod × feature type × model 的 MCC 分數已儲存到 {resultCsvPath}")
