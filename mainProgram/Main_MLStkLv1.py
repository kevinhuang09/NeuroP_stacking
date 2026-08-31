import matplotlib

def setMatplotlibBackend(disablePopup):
    """disablePopup=True: 使用Agg backend，禁止彈出圖表視窗；disablePopup=False: 維持預設backend，正常跳出視窗"""
    if disablePopup:
        matplotlib.use('Agg')

disablePlotPopup = True  # 是否禁止彈出圖表視窗
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

useVscodeParentPath = True  # 使用pycharm, vscode進行編譯請開啟
setupParentPath(useVscodeParentPath)

import json
import copy
import pandas as pd
from userPackage.Package_Encode import EncodeAllFeatures
from MLProcess.PycaretWrapper import PycaretWrapper
from MLProcess.Predict import Predict

# ======================================================================================================================
# 基本路徑
paramPath = "../data/param/"  # 內含檔案: featureTypeDict.pkl, normalize.pkl
featureStatPath = '../data/featureStat/'
mlScorePath = "../data/mlScore/"  # 內含 ml model 預測完並算好分的檔案
tuneModelPath = "../data/tuneModel/"  # 內含每個 normalizeMethod/featureType 底下 finalize 好的 model
dataName = 'NeuroP_1'

normalizeMethod = ['standard', 'robust']  # normalization 要跑的兩種方式

tune_model = True  # True: 針對每個 normalizeMethod × feature type × model 做一對一 tune 並存檔；False: 跳過訓練，直接讀取已存的 finalized model 做 predict

# ======================================================================================================================
# 讀取 Main_FeatureStk.py 存的 featureTypeDict.json（實際 encode 時用的完整 featureDict，含各參數）
featureTypeDictJsonPath = paramPath + f'{dataName}_featureTypeDict.json'
with open(featureTypeDictJsonPath, 'r', encoding='utf-8') as f:
    usedFeatureDict = json.load(f)

def buildAllOffFeatureDict(baseDict):
    """把 iFeature/pFeature/ampFeature/ovpFeature/mergedFeature/motifBitVecFeature/centerGDPFeature 全部開關關閉，保留其餘參數結構"""
    offDict = copy.deepcopy(baseDict)
    for groupName in ['iFeature', 'pFeature', 'ampFeature', 'ovpFeature', 'mergedFeature']:
        if groupName not in offDict:
            continue
        for key, value in offDict[groupName].items():
            if isinstance(value, list):
                value[0] = False
            else:
                offDict[groupName][key] = False
    # offDict['motifBitVecFeature']['Usage'] = False
    offDict['centerGDPFeature']['Usage'] = False
    return offDict

def getRealEnabledFeatureTypeList(featureDict):
    """列出真正會產生欄位的 feature type（(groupName, key) tuple 清單）"""
    enabledList = []
    for groupName in ['iFeature', 'pFeature', 'ampFeature', 'ovpFeature', 'mergedFeature']:
        if groupName not in featureDict:
            continue
        for key, value in featureDict[groupName].items():
            if value is True or (isinstance(value, list) and value[0] is True):
                enabledList.append((groupName, key))
    # if featureDict['motifBitVecFeature'].get('Usage') is True:
    #     enabledList.append(('motifBitVecFeature', 'Usage'))
    if featureDict['centerGDPFeature'].get('Usage') is True:
        enabledList.append(('centerGDPFeature', 'Usage'))
    return enabledList

# mergedFeature.OVPC_GAAC_formula / mergedFeature.SingleValueCombine 這兩個 type，Package_Encode.py
# 並不認得 'mergedFeature' 這個 key，直接對它們 encode 只會拿到 0 欄，所以跟 Main_FeatureStk.py 一樣，
# 用寫死的欄位名稱清單取代掉原本 encode 出來的空清單
SINGLE_VALUE_FEATURE_NAME_LIST = [
    "length", "calculate_mw", "calculate_charge", "isoelectric_point",
    "instability_index", "aromaticity", "aliphatic_index", "hydrophobic",
    "aasi", "argos", "bulkiness", "charge_phys", "charge_acid",
    "flexibility", "gravy", "levitt_alpha", "mss", "polarity",
    "refractivity", "tm_tend", "boman_index", "eisenberg",
    "hopp_woods", "janin", "kytedoolittle", "SE", "charge_density"
]

OVPC_GAAC_FORMULA_COLUMN_LIST = [
    'OVPC_Aromatic', 'OVPC_Negative', 'OVPC_Positive', 'OVPC_Polar', 'OVPC_Hydrophobic',
    'OVPC_Aliphatic', 'OVPC_Tiny', 'OVPC_Charged', 'OVPC_Small', 'OVPC_Imino_acid',
    'GAAC_alphatic', 'GAAC_aromatic', 'GAAC_postivecharge', 'GAAC_negativecharge', 'GAAC_uncharge',
    'formula_C', 'formula_H', 'formula_N', 'formula_O', 'formula_S'
]

SINGLE_VALUE_COMBINE_COLUMN_LIST = [
    'Length', 'Calculate_mw', 'Calculate_charge', 'Isoelectric_point',
    'Instability_index', 'Aromaticity', 'Aliphatic_Index', 'Hydrophobic',
    'AASI', 'Argos', 'Bulkiness', 'Charge_phys', 'Charge_acid',
    'Flexibility', 'Gravy', 'Levitt_alpha', 'MSS', 'Polarity',
    'Refractivity', 'TM_tend', 'Boman_Index', 'Eisenberg',
    'Hopp_woods', 'Janin', 'Kytedoolittle', 'Shannon-Entropy', 'Charge_density'
]

mergedFeatureColumnMap = {
    'mergedFeature.OVPC,GAAC,formula': OVPC_GAAC_FORMULA_COLUMN_LIST,
    f'mergedFeature.{",".join(map(str, SINGLE_VALUE_FEATURE_NAME_LIST))}': SINGLE_VALUE_COMBINE_COLUMN_LIST,
}

def discoverFeatureTypeColumnMap(featureDict, sampleDataDict):
    """
    對每個真正開啟的 feature type，各自單獨開啟、encode 一次（只為了拿欄位名稱，不重算真正的特徵值），
    建立 feature type 名稱 -> 欄位名稱清單 的對照表，之後直接從已經 encode 好的完整 DataFrame 切欄位。
    """
    realEnabledList = getRealEnabledFeatureTypeList(featureDict)
    allOffTemplate = buildAllOffFeatureDict(featureDict)

    featureTypeColumnMap = {}
    for groupName, key in realEnabledList:
        singleFeatureDict = copy.deepcopy(allOffTemplate)
        if isinstance(singleFeatureDict[groupName][key], list):
            singleFeatureDict[groupName][key][0] = True
        else:
            singleFeatureDict[groupName][key] = True

        typeName = f'{groupName}.{key}'
        if typeName in mergedFeatureColumnMap:
            columnList = mergedFeatureColumnMap[typeName]
        else:
            singleEncodeObj = EncodeAllFeatures()
            singleEncodeObj.featureDict = singleFeatureDict
            singleDf = singleEncodeObj.dataEncodeOutPut(dataDict=sampleDataDict)
            columnList = [c for c in singleDf.columns if c != 'y']

        featureTypeColumnMap[typeName] = columnList
        print(f"feature type {typeName}: {len(columnList)} 欄")

    return featureTypeColumnMap


# 只需要少量序列來辨識每個 feature type 產生的欄位名稱，不需要整個 DS_Train
DS_TrainSeqDf = pd.read_csv(featureStatPath + 'DS_Train.csv')
sampleSeqDict = dict(zip(DS_TrainSeqDf['name'].head(5), DS_TrainSeqDf['sequence'].head(5)))
sampleDataDict = {0: sampleSeqDict, 1: None, -1: None}

featureTypeColumnMap = discoverFeatureTypeColumnMap(usedFeatureDict, sampleDataDict)
print(f"use {len(featureTypeColumnMap)} feature types")

# ======================================================================================================================
# 每一個 normalize 方式 × 每一個 feature type × 每一個 model 做一對一訓練（Optuna TPE tune，5-fold CV，用 MCC 當優化目標）
# normalize 後的資料是 Main_FeatureStk.py 存好的，這裡只需要讀取
modelNameList = ['lightgbm', 'catboost', 'rbfsvm', 'gbc', 'ridge', 'lr', 'lda', 'ada', 'knn', 'nb', 'et', 'rf',
                 'xgboost', 'mlp', 'dt', 'svm', 'qda']

print(f"use {len(modelNameList)} models")

os.makedirs(mlScorePath, exist_ok=True)
resultRows = []

for normalizeMethod in normalizeMethod:
    filterTrainNmlzCsvPath = featureStatPath + f'filtered_train_{dataName}_{normalizeMethod}.csv'
    dataTrainDf = pd.read_csv(filterTrainNmlzCsvPath, index_col=[0])
    print(f"[{normalizeMethod}] 讀取 filtered 後的資料：dataTrainDf={dataTrainDf.shape}")

    # DS_Val：拿來給每個 tune/load 好的 model 做 predict，組成 Meta-Feature-Matrix
    filterValNmlzCsvPath = featureStatPath + f'filtered_val_{dataName}_{normalizeMethod}.csv'
    dataValDf = pd.read_csv(filterValNmlzCsvPath, index_col=[0])
    print(f"[{normalizeMethod}] 讀取 DS_Val filtered 後的資料：dataValDf={dataValDf.shape}")

    metaFeatureMatrix = pd.DataFrame(index=dataValDf.index)
    metaFeatureMatrix['y'] = dataValDf['y']

    for typeName, columnList in featureTypeColumnMap.items(): # 一次挑一種 Feature Type
        # filtered_train 已被 FeatureStat 過濾掉部分欄位，columnList 是用原始未過濾的 featureDict 反查出來的，
        # 兩者可能對不上，所以只取仍存在於 dataTrainDf 的欄位，避免 KeyError
        survivedColumnList = [c for c in columnList if c in dataTrainDf.columns]
        if not survivedColumnList:
            print(f"[跳過] {normalizeMethod} + {typeName}：欄位全部被 FeatureStat 過濾掉，無法訓練")
            if tune_model:
                resultRows.append({'normalizeMethod': normalizeMethod, 'featureType': typeName,
                                   'model': None, 'mcc': None, 'error': 'all columns filtered out'})
            continue
        if len(survivedColumnList) < len(columnList):
            print(f"[提醒] {normalizeMethod} + {typeName}：{len(columnList) - len(survivedColumnList)} 欄"
                  f"被 FeatureStat 過濾掉，剩餘 {len(survivedColumnList)} 欄")
        subTrainDf = dataTrainDf[survivedColumnList + ['y']].copy() # 篩選該feature type 之 feature
        subVal_X = dataValDf[survivedColumnList].copy()  # DS_Val 同一組欄位，只留 X 給 predict 用

        # 每個 normalizeMethod + featureType 的 model 各自存在獨立資料夾，避免互相覆蓋
        comboSavePath = os.path.join(tuneModelPath, normalizeMethod, typeName.replace('.', '_'))
        os.makedirs(comboSavePath, exist_ok=True)

        pycObj = PycaretWrapper()
        if tune_model:
            pycObj.doSetup(trainData=subTrainDf, sessionID=42)

        for modelName in modelNameList: # 每個 model 逐一嘗試
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
                    pycObj.doSaveModel(comboSavePath, b_isFinalizedModel=True)
                    predictModelList = pycObj.finalModelList
                else:
                    # tune_model=False：跳過訓練，直接讀取已存的 finalized model
                    predictModelList = pycObj.doLoadModel(comboSavePath, fileNameList=[modelName],
                                                           b_isFinalizedModel=True)

                # 用這個 model 對 DS_Val 做 predict，寫進 Meta-Feature-Matrix
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

    metaFeatureMatrixPath = mlScorePath + f'Meta-Feature-Matrix_{dataName}_{normalizeMethod}.csv'
    metaFeatureMatrix.to_csv(metaFeatureMatrixPath)
    print(f"[{normalizeMethod}] Meta-Feature-Matrix 已儲存到 {metaFeatureMatrixPath}")

if tune_model:
    resultDf = pd.DataFrame(resultRows)
    resultCsvPath = mlScorePath + f'featureType_model_mccScore_{dataName}.csv'
    resultDf.to_csv(resultCsvPath, index=False)
    print(f"每個 normalizeMethod × feature type × model 的 MCC 分數已儲存到 {resultCsvPath}")
