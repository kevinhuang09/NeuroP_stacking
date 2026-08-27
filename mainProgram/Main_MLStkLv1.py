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

# ======================================================================================================================
# 基本路徑
paramPath = "../data/param/"  # 內含檔案: featureTypeDict.pkl, normalize.pkl
featureStatPath = '../data/featureStat/'
mlScorePath = "../data/mlScore/"  # 內含 ml model 預測完並算好分的檔案
dataName = 'NeuroP_1'

normalizeMethod = ['standard', 'robust']  # normalization 要跑的兩種方式

# ======================================================================================================================
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

    for typeName, columnList in featureTypeColumnMap.items(): # 一次挑一種 Feature Type
        # filtered_train 已被 FeatureStat 過濾掉部分欄位，columnList 是用原始未過濾的 featureDict 反查出來的，
        # 兩者可能對不上，所以只取仍存在於 dataTrainDf 的欄位，避免 KeyError
        survivedColumnList = [c for c in columnList if c in dataTrainDf.columns]
        if not survivedColumnList:
            print(f"[跳過] {normalizeMethod} + {typeName}：欄位全部被 FeatureStat 過濾掉，無法訓練")
            resultRows.append({'normalizeMethod': normalizeMethod, 'featureType': typeName,
                               'model': None, 'mcc': None, 'error': 'all columns filtered out'})
            continue
        if len(survivedColumnList) < len(columnList):
            print(f"[提醒] {normalizeMethod} + {typeName}：{len(columnList) - len(survivedColumnList)} 欄"
                  f"被 FeatureStat 過濾掉，剩餘 {len(survivedColumnList)} 欄")
        subTrainDf = dataTrainDf[survivedColumnList + ['y']].copy() # 篩選該feature type 之 feature

        pycObj = PycaretWrapper()
        pycObj.doSetup(trainData=subTrainDf, sessionID=42)

        for modelName in modelNameList: # 每個 model 逐一嘗試
            try:
                # 進行一對一訓練
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
resultCsvPath = mlScorePath + f'featureType_model_mccScore_{dataName}.csv'
resultDf.to_csv(resultCsvPath, index=False)
print(f"每個 normalizeMethod × feature type × model 的 MCC 分數已儲存到 {resultCsvPath}")
