# 禁止彈出圖表視窗
import matplotlib
matplotlib.use('Agg')

# 使用vscode進行編譯請開啟
import sys, os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent = os.path.join(current_dir, "..")
sys.path.append(parent)

import json
import copy
import pandas as pd
from userPackage.Package_Encode import EncodeAllFeatures
from MLProcess.PycaretWrapper import PycaretWrapper

# ======================================================================================================================
# 基本路徑
mlDataPath = "../data/mlData/"  # 內含 data 檔案 ex : train_F390.csv, boruta 檔案 ex :Boruta-featRank-RF.csv
paramPath = "../data/param/"  # 內含檔案: featureTypeDict.pkl, normalize.pkl
featureStatPath = '../data/featureStat/'
mlScorePath = "../data/mlScore/"  # 內含 ml model 預測完並算好分的檔案
dataName = 'NeuroP_1'

normalizeMethodList = ['standard', 'robust']  # normalization 要跑的兩種方式

# ======================================================================================================================
# 載入 Main_FeatureStk.py 已經 encode 好並存檔的資料
encodeDS_TrainCsvPath = featureStatPath + f'encode_{dataName}_DS_Train.csv'
encodeDS_IndpCsvPath = featureStatPath + f'encode_{dataName}_DS_Indp.csv'
encodeDS_TrainDf = pd.read_csv(encodeDS_TrainCsvPath, index_col=[0])
encodeDS_IndpDf = pd.read_csv(encodeDS_IndpCsvPath, index_col=[0])
print(f"讀取 encode 後的資料：DS_Train={encodeDS_TrainDf.shape}, DS_Indp={encodeDS_IndpDf.shape}")

# 讀取 Main_FeatureStk.py 存的 featureTypeDict.json（實際 encode 時用的完整 featureDict，含各參數）
featureTypeDictJsonPath = paramPath + f'{dataName}_featureTypeDict.json'
with open(featureTypeDictJsonPath, 'r', encoding='utf-8') as f:
    usedFeatureDict = json.load(f)

def buildAllOffFeatureDict(baseDict):
    """把 iFeature/pFeature/ampFeature/ovpFeature/motifBitVecFeature/centerGDPFeature 全部開關關閉，保留其餘參數結構"""
    offDict = copy.deepcopy(baseDict)
    for groupName in ['iFeature', 'pFeature', 'ampFeature', 'ovpFeature']:
        for key, value in offDict[groupName].items():
            if isinstance(value, list):
                value[0] = False
            else:
                offDict[groupName][key] = False
    offDict['motifBitVecFeature']['Usage'] = False
    offDict['centerGDPFeature']['Usage'] = False
    return offDict

def getRealEnabledFeatureTypeList(featureDict):
    """列出真正會產生欄位的 feature type（(groupName, key) tuple 清單）"""
    enabledList = []
    for groupName in ['iFeature', 'pFeature', 'ampFeature', 'ovpFeature']:
        for key, value in featureDict[groupName].items():
            if value is True or (isinstance(value, list) and value[0] is True):
                enabledList.append((groupName, key))
    if featureDict['motifBitVecFeature'].get('Usage') is True:
        enabledList.append(('motifBitVecFeature', 'Usage'))
    if featureDict['centerGDPFeature'].get('Usage') is True:
        enabledList.append(('centerGDPFeature', 'Usage'))
    return enabledList


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

        singleEncodeObj = EncodeAllFeatures()
        singleEncodeObj.featureDict = singleFeatureDict
        singleDf = singleEncodeObj.dataEncodeOutPut(dataDict=sampleDataDict)
        columnList = [c for c in singleDf.columns if c != 'y']

        typeName = f'{groupName}.{key}'
        featureTypeColumnMap[typeName] = columnList
        print(f"feature type {typeName}: {len(columnList)} 欄")

    return featureTypeColumnMap


# 只需要少量序列來辨識每個 feature type 產生的欄位名稱，不需要整個 DS_Train
DS_TrainSeqDf = pd.read_csv(featureStatPath + 'DS_Train.csv')
sampleSeqDict = dict(zip(DS_TrainSeqDf['name'].head(5), DS_TrainSeqDf['sequence'].head(5)))
sampleDataDict = {0: sampleSeqDict, 1: None, -1: None}

featureTypeColumnMap = discoverFeatureTypeColumnMap(usedFeatureDict, sampleDataDict)

# ------------------------------------------------------------------------------------------------------------------
# 測試版：只取前 2 個 feature type，快速確認整條流程有沒有錯誤
featureTypeColumnMap = dict(list(featureTypeColumnMap.items())[:2])
print(f"[test] 只跑這 2 個 feature type: {list(featureTypeColumnMap.keys())}")

# ======================================================================================================================
# 每一個 normalize 方式 × 每一個 feature type × 每一個 model 做一對一訓練（Optuna TPE tune，5-fold CV，用 MCC 當優化目標）
# normalize 後的資料是 Main_FeatureStk.py 存好的，這裡只需要讀取
# 測試版：只用 lightgbm、rf(random forest) 這兩個 model，1小時內快速確認有無錯誤
modelNameList = ['lightgbm', 'rf']

os.makedirs(mlScorePath, exist_ok=True)
resultRows = []

for normalizeMethod in normalizeMethodList:
    trainNmlzCsvPath = featureStatPath + f'train_{dataName}_{normalizeMethod}.csv'
    indpNmlzCsvPath = featureStatPath + f'indp_{dataName}_{normalizeMethod}.csv'
    trainNmlzDf = pd.read_csv(trainNmlzCsvPath, index_col=[0])
    indpNmlzDf = pd.read_csv(indpNmlzCsvPath, index_col=[0])
    print(f"[{normalizeMethod}] 讀取 normalize 後的資料：DS_Train={trainNmlzDf.shape}, DS_Indp={indpNmlzDf.shape}")

    for typeName, columnList in featureTypeColumnMap.items():
        subTrainDf = trainNmlzDf[columnList + ['y']].copy()

        pycObj = PycaretWrapper()
        pycObj.doSetup(trainData=subTrainDf, sessionID=42)

        for modelName in modelNameList:
            try:
                tunedModelList, tunerList = pycObj.doTuneModel(searchLibrary='optuna', searchAlg='tpe',
                                                                includeModelList=[modelName], foldNum=5,
                                                                n_iter=10, early_stopping=False, customGridDict=None)
                _, scoreRank = pycObj.doCompareModel(fold=5, includeModelList=tunedModelList)  # 對 tune 好的 model 重新做一次 CV 拿分數表
                mccScore = scoreRank['MCC'].iloc[0]
                print(f"[完成] {normalizeMethod} + {typeName} + {modelName}: MCC = {mccScore}")
                resultRows.append({'normalizeMethod': normalizeMethod, 'featureType': typeName,
                                   'model': modelName, 'mcc': mccScore})
            except Exception as e:
                print(f"[跳過] {normalizeMethod} + {typeName} + {modelName} 訓練失敗: {e}")
                resultRows.append({'normalizeMethod': normalizeMethod, 'featureType': typeName,
                                   'model': modelName, 'mcc': None, 'error': str(e)})

resultDf = pd.DataFrame(resultRows)
resultCsvPath = mlScorePath + f'featureType_model_mccScore_{dataName}_test.csv'
resultDf.to_csv(resultCsvPath, index=False)
print(f"[test] 每個 normalizeMethod × feature type × model 的 MCC 分數已儲存到 {resultCsvPath}")
