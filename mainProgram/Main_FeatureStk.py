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

import copy
import csv
import json

from sklearn.model_selection import train_test_split

from userPackage.Package_Encode import EncodeAllFeatures
from userPackage.LoadDataset import LoadDataset
from userPackage.FeatureStat import FeatureStat

#  移到外面再把舊的加進空的dict裡面
ifeatureDict = {"AAC": True,
                "AAINDEX": False,  # 注意!!需等長
                "CKSAAGP": [True, 5],
                "CTDC": True,
                "CTDD": True,
                "CTDT": True,
                "CTriad": True,  # 會產生feature 343個
                "DDE": True,
                "DPC": True,
                "GAAC": True,
                "GDPC": True,
                "GTPC": True,
                "KSCTriad": [True, 0],  # 會產生feature 343個
                "QSOrder": [False, 30, 0.1],
                "TPC": False,  # 會有8000個feature 不建議開啟
                "SOCN": [False, 3],
                "APAAC": [True, 3, 0.05],
                "Geary": [True, 3],
                "Moran": [True, 3],
                "NMBroto": [True, 3],
                "CKSAAP": [True, 3],  # 會產生feature 1600個
                "BINARY": False,  # 注意!!需等長
                "PAAC": [True, 3, 0.05]
                }

PfeatureDict = {"DDOR": True,
                "RRI": True,
                "SER": True,
                "SEP": True,
                "SE": True,
                "QSO": [True, 3, 0.1]
                }  # gap=3 w=0.1

AMPfeatureDict = {"length": True,
                  "calculate_mw": [True, True],  # [0] = on(True)/off [1] = amide
                  "calculate_charge": [True, 7, True],  # [0] = on(True)/off [1] = ph, [2] = amide
                  "charge_density": [True, 7, True],  # [0] = on(True)/off [1] = ph, [2] = amide
                  "isoelectric_point": [True, True],  # [0] = on(True)/off [1] = amide
                  "instability_index": True,
                  "aromaticity": True,
                  "aliphatic_index": True,
                  "hydrophobic": True,
                  "aasi": True,
                  "abhprk": [True, 5],  # [0] = on(True)/off [1] = window
                  "argos": True,
                  "bulkiness": True,
                  "charge_phys": True,
                  "charge_acid": True,
                  "cougar": [True, 5],  # [0] = on(True)/off [1] = window
                  "ez": [True, 5],  # [0] = on(True)/off [1] = window
                  "flexibility": True,
                  "gravy": True,
                  "levitt_alpha": True,
                  "mss": True,
                  "msw": [True, 5],  # [0] = on(True)/off [1] = window
                  "pepArc": True,
                  "polarity": True,
                  "refractivity": True,
                  "tm_tend": True,
                  "z3": [True, 5],  # [0] = on(True)/off [1] = window
                  "z5": [True, 5],  # [0] = on(True)/off [1] = window
                  "formula": True,  # C,H,N,O,S atom composition
                  "boman_index": True,
                  "eisenberg": True,
                  "hopp_woods": True,
                  "janin": True,
                  "kytedoolittle": True
                  }

OVPfeatureDict = {"OVPC": True,
                  "OVP": [True, 4, 4]  # 兩個數為 N, C 端胺基酸數目  N= C=
                  }

MotifBitVecfeatureDict = {"Usage": False,
                          "motifList": ['FKK', 'LKL', 'KKLL', 'KWK', 'VLK',
                                        'CY'
                                        ''
                                        'CR', 'CRR', 'RFC', 'RRR', 'LKKL']
                          }

centerGDPDict = {"Usage": False, "UseGap": False, "gap_size": -1}  # 若是預測每個 amino acid 的 label 在使用

featureDict = {'iFeature': ifeatureDict,
               'pFeature': PfeatureDict,
               'ampFeature': AMPfeatureDict,
               'ovpFeature': OVPfeatureDict,
               'centerGDPFeature': centerGDPDict}


def buildOVPC_GAAC_formulaFeatureDict():
    """
    以 featureDict 為基礎，把原本各自獨立的 OVPC(ovpFeature)、GAAC(iFeature)、formula(ampFeature)
    三個開關關閉，取代成一個新群組 mergedFeature 底下的單一開關 OVPC_GAAC_formula 來代表這三者。
    其餘特徵開關維持不變。統計 feature type 數量時，三個獨立項目變成一個合併項目，理論上會少兩個。
    """
    newFeatureDict = copy.deepcopy(featureDict)

    newFeatureDict['iFeature']['GAAC'] = False
    newFeatureDict['ampFeature']['formula'] = False
    newFeatureDict['ovpFeature']['OVPC'] = False

    newFeatureDict['mergedFeature'] = {'OVPC,GAAC,formula': True}

    return newFeatureDict

# 這裡要用「featureDict 裡實際的 key」，大小寫要對上，不然 if feat in sub_dict 永遠找不到、關不掉
SINGLE_VALUE_FEATURE_NAME_LIST = [
    "length", "calculate_mw", "calculate_charge", "isoelectric_point",
    "instability_index", "aromaticity", "aliphatic_index", "hydrophobic",
    "aasi", "argos", "bulkiness", "charge_phys", "charge_acid",
    "flexibility", "gravy", "levitt_alpha", "mss", "polarity",
    "refractivity", "tm_tend", "boman_index", "eisenberg",
    "hopp_woods", "janin", "kytedoolittle", "SE", "charge_density"
]

def buildSingleValueCombineFeatureDict():
    """
    將單一數值型理化性質特徵關閉，
    改為在 mergedFeature 底下新增單一開關 SingleValueCombine。
    """
    newFeatureDict = copy.deepcopy(featureDict)

    print(f"singlevalue length : {len(SINGLE_VALUE_FEATURE_NAME_LIST)}")
    # 自動巡訪並關閉原始群組中的對應特徵開關
    for group, sub_dict in newFeatureDict.items():
        if isinstance(sub_dict, dict):
            for feat in SINGLE_VALUE_FEATURE_NAME_LIST:
                if feat in sub_dict:
                    if isinstance(sub_dict[feat], list):
                        sub_dict[feat][0] = False  # list 型參數只切換開關位，保留其餘參數
                    else:
                        sub_dict[feat] = False

    # 在 mergedFeature 加入合併開關 (避免覆蓋已存在的 mergedFeature)
    # if 'mergedFeature' not in newFeatureDict:
    #     newFeatureDict['mergedFeature'] = {}
    newFeatureDict['mergedFeature'][f'{",".join(map(str, SINGLE_VALUE_FEATURE_NAME_LIST))}'] = True

    return newFeatureDict

def countEnabledFeatureType(featureDict):
    """統計傳入的 featureDict 中，有多少個 feature type 被開啟(True)，並列出名稱"""
    enabledFeatureList = []

    for groupName, groupDict in featureDict.items():
        if 'Usage' in groupDict:
            if groupDict.get('Usage') is True:
                enabledFeatureList.append(f'{groupName}.Usage')
            continue
        for key, value in groupDict.items():
            if value is True or (isinstance(value, list) and value[0] is True):
                enabledFeatureList.append(f'{groupName}.{key}')

    return len(enabledFeatureList), enabledFeatureList

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

def discoverFeatureTypeColumnMap(featureDict, sampleDataDict):
    """
    對每個真正開啟的 feature type，各自單獨開啟、encode 一次（只為了拿欄位名稱，不重算真正的特徵值），
    建立 feature type 名稱 -> 欄位名稱清單 的對照表，供之後把被過濾掉的 feature 欄位對應回所屬 feature type 使用。
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
        # print(f"feature type {typeName}: {len(columnList)} 欄")

    return featureTypeColumnMap

# ======================================================================================================================
# 基本路徑
mlDataPath = "../data/mlData/"  # 內含 data 檔案 ex : train_F390.csv, boruta 檔案 ex :Boruta-featRank-RF.csv
paramPath = "../data/param/"  # 內含檔案: featureTypeDict.pkl, normalize.pkl
featureStatPath = '../data/featureStat/'
dataName = 'NeuroP_1'

normalizeMethodList = ['standard', 'robust']  # normalization 要跑的兩種方式

# DataSet載入區
# 載入Main Dataset
MainDatasetNegFastaPath = "../data/MainDatasetNeg.fasta"
MainDatasetPosFastaPath = "../data/MainDatasetPos.fasta"

# 載入DS_Indp
DS_IndpNegFastaPath = "../data/DS_IndpNeg.fasta"
DS_IndpPosFastaPath = "../data/DS_IndpPos.fasta"

ldObj = LoadDataset(minSeqLength=5)
MainDatasetNegSeqDict = ldObj.readFasta(MainDatasetNegFastaPath)
MainDatasetPosSeqDict = ldObj.readFasta(MainDatasetPosFastaPath)
DS_IndpNegSeqDict = ldObj.readFasta(DS_IndpNegFastaPath)
DS_IndpPosSeqDict = ldObj.readFasta(DS_IndpPosFastaPath)

# 印出當前序列數量做確認
print(f"[Fasta 統計] MainDatasetNeg: {len(MainDatasetNegSeqDict)}, MainDatasetPos: {len(MainDatasetPosSeqDict)}, "
      f"DS_IndpNeg: {len(DS_IndpNegSeqDict)}, DS_IndpPos: {len(DS_IndpPosSeqDict)}")

# 儲存序列數量統計到 data/ 中
fastaStatPath = "../data/fasta_seq_count.csv"
with open(fastaStatPath, 'w', encoding='utf-8') as f:
    f.write("file,count\n")
    f.write(f"{os.path.basename(MainDatasetNegFastaPath)},{len(MainDatasetNegSeqDict)}\n")
    f.write(f"{os.path.basename(MainDatasetPosFastaPath)},{len(MainDatasetPosSeqDict)}\n")
    f.write(f"{os.path.basename(DS_IndpNegFastaPath)},{len(DS_IndpNegSeqDict)}\n")
    f.write(f"{os.path.basename(DS_IndpPosFastaPath)},{len(DS_IndpPosSeqDict)}\n")
print(f"序列數量統計已儲存到{fastaStatPath}中")

# ======================================================================================================================
# 將 MainDataset 依 9:1 切分為 DS_Train / DS_Val（neg, pos 各自切分以維持類別比例）
splitTestSize = 0.1
splitRandomState = 42

def splitSeqDict(seqDict, test_size, random_state):
    keys = list(seqDict.keys())
    trainKeys, valKeys = train_test_split(keys, test_size=test_size, random_state=random_state)
    return {k: seqDict[k] for k in trainKeys}, {k: seqDict[k] for k in valKeys}

DS_TrainNegSeqDict, DS_ValNegSeqDict = splitSeqDict(MainDatasetNegSeqDict, splitTestSize, splitRandomState)
DS_TrainPosSeqDict, DS_ValPosSeqDict = splitSeqDict(MainDatasetPosSeqDict, splitTestSize, splitRandomState)

print(f"[DS_Train/DS_Val 統計] DS_TrainNeg: {len(DS_TrainNegSeqDict)}, DS_TrainPos: {len(DS_TrainPosSeqDict)}, "
      f"DS_ValNeg: {len(DS_ValNegSeqDict)}, DS_ValPos: {len(DS_ValPosSeqDict)}")

# 儲存 DS_Train/DS_Val 數量統計到同一份 csv 中
with open(fastaStatPath, 'a', encoding='utf-8') as f:
    f.write(f"DS_TrainNeg,{len(DS_TrainNegSeqDict)}\n")
    f.write(f"DS_TrainPos,{len(DS_TrainPosSeqDict)}\n")
    f.write(f"DS_ValNeg,{len(DS_ValNegSeqDict)}\n")
    f.write(f"DS_ValPos,{len(DS_ValPosSeqDict)}\n")
print(f"DS_Train/DS_Val 數量統計已儲存到{fastaStatPath}中")

DS_TrainDataDict = {0 : DS_TrainNegSeqDict, 1 : DS_TrainPosSeqDict, -1 : None}
DS_IndpDataDict = {0 : DS_IndpNegSeqDict, 1 : DS_IndpPosSeqDict, -1 : None}
DS_ValDataDict = {0 : DS_ValNegSeqDict, 1 : DS_ValPosSeqDict, -1 : None}

def saveDataDictToCsv(dataDict, csvPath):
    """把 dataDict（{-1/0/1: {序列名稱: 序列}}）攤平存成 csv，欄位為 name, sequence, label"""
    os.makedirs(os.path.dirname(csvPath), exist_ok=True)
    with open(csvPath, 'w', encoding='utf-8') as f:
        f.write("name,sequence,label\n")
        rowCount = 0
        for label, seqDict in dataDict.items():
            if seqDict is None:
                continue
            for name, sequence in seqDict.items():
                f.write(f"{name},{sequence},{label}\n")
                rowCount += 1
    print(f"{csvPath} 已儲存，共 {rowCount} 筆")


DS_TrainCsvPath = featureStatPath + 'DS_Train.csv'
DS_IndpCsvPath = featureStatPath + 'DS_Indp.csv'
DS_ValCsvPath = featureStatPath + 'DS_Val.csv'
saveDataDictToCsv(DS_TrainDataDict, DS_TrainCsvPath)
saveDataDictToCsv(DS_IndpDataDict, DS_IndpCsvPath)
saveDataDictToCsv(DS_ValDataDict, DS_ValCsvPath)

count, names = countEnabledFeatureType(featureDict)
print(f"featureDict 啟用的 feature type 數量: {count}")

featureDict = buildOVPC_GAAC_formulaFeatureDict()
count2, names2 = countEnabledFeatureType(featureDict)
print(f"OVPC_GAAC_formula 啟用的 feature type 數量: {count2}")

featureDict = buildSingleValueCombineFeatureDict()
count3, names3 = countEnabledFeatureType(featureDict)
print(f"singleValueCombine 啟用的 feature type 數量: {count3}")

# ======================================================================================================================
# 只需要少量序列來辨識每個 feature type 產生的欄位名稱，不需要整個 DS_Train
os.makedirs(featureStatPath, exist_ok=True)
columnDiscoverySampleSize = 5
columnDiscoverySampleDataDict = {
    0: dict(list(DS_TrainNegSeqDict.items())[:columnDiscoverySampleSize]),
    1: dict(list(DS_TrainPosSeqDict.items())[:columnDiscoverySampleSize]),
    -1: None
}

# mergedFeature.OVPC_GAAC_formula / mergedFeature.SingleValueCombine 這兩個 type，Package_Encode.py
# 並不認得 'mergedFeature' 這個 key，直接對它們 encode 只會拿到 0 欄。
# 這兩份清單是對照 devPackage/OVP.py、PackageiFeature.py、PackageModelAmp.py、PackagePFeature.py
# 原始程式碼比對確認過的真實欄位名稱，直接寫死取代掉原本 encode 出來的空清單
OVPC_GAAC_FORMULA_COLUMN_LIST = [
    'OVPC_Aromatic', 'OVPC_Negative', 'OVPC_Positive', 'OVPC_Polar', 'OVPC_Hydrophobic',
    'OVPC_Aliphatic', 'OVPC_Tiny', 'OVPC_Charged', 'OVPC_Small', 'OVPC_Imino_acid',
    'GAAC_alphatic', 'GAAC_aromatic', 'GAAC_postivecharge', 'GAAC_negativecharge', 'GAAC_uncharge',
    'formula_C', 'formula_H', 'formula_N', 'formula_O', 'formula_S'
]

# 順序對應 SINGLE_VALUE_FEATURE_NAME_LIST 的 27 個 key，各自實際的欄位名稱
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

# 建立 feature type 名稱 -> 欄位名稱清單 的對照表（合併後最終使用的 33 類 feature type），
# 之後用來把被 FeatureStat 過濾掉的 feature 欄位對應回所屬 feature type
featureTypeColumnMap = discoverFeatureTypeColumnMap(featureDict, columnDiscoverySampleDataDict)
columnToFeatureType = {column: typeName for typeName, columnList in featureTypeColumnMap.items() for column in columnList}

# 33 類 feature type 明細表：Feature Type, feature size；只有 mergedFeature 底下這兩個合併後的 type，
# 以及 feature size 剛好等於 1 的 type，才列出實際包含的欄位名稱（逗號分隔）
featureTypeTablePath = featureStatPath + f'featureType_featureName_table_{dataName}.csv'

# 33 類 feature type 明細表（轉為橫向輸出）
featureTypeTablePath = featureStatPath + f'featureType_featureName_table_{dataName}.csv'

# 1. 取得原始的所有 Key
all_keys = list(featureTypeColumnMap.keys())

# 2. 拆分成「非 mergedFeature」與「mergedFeature」，再重組（merged 移到最後面）
other_keys = [k for k in all_keys if not k.startswith('mergedFeature')]
merged_keys = [k for k in all_keys if k.startswith('mergedFeature')]
ordered_keys = other_keys + merged_keys

# 3. 依照新順序建立橫向 CSV
featureTypeTablePath = featureStatPath + f'featureType_featureName_table_{dataName}.csv'
featureTypesRow = ['Feature Type']
featureSizesRow = ['feature size']

for typeName in ordered_keys:
    columnList = featureTypeColumnMap[typeName]
    if typeName in mergedFeatureColumnMap:
        columnList = mergedFeatureColumnMap[typeName]
    
    # 移除 mergedFeature. 前綴
    # displayName = typeName.replace('mergedFeature.', '')
    displayName = typeName.split('.')[-1]

    featureTypesRow.append(displayName)
    featureSizesRow.append(len(columnList))

# 4. 寫入 CSV
with open(featureTypeTablePath, 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(featureTypesRow)
    writer.writerow(featureSizesRow)

print(f"{len(featureTypeColumnMap)} 類 feature type 明細表已儲存到 {featureTypeTablePath}")

# 儲存這三步驟的 feature type 數量統計到 csv 中
os.makedirs(featureStatPath, exist_ok=True)
featureTypeStatPath = featureStatPath + 'featureTypeStatistics.csv'
with open(featureTypeStatPath, 'w', encoding='utf-8') as f:
    f.write("featureType Process,count\n")
    f.write(f"origin featureDict,{count}\n")
    f.write(f"after merged OVPC GAAC formula,{count2}\n")
    f.write(f"after merged all single value,{count3}\n")
print(f"feature type 統計已儲存到{featureTypeStatPath}中")

encodeObj = EncodeAllFeatures()

encodeObj.dataEncodeSetup(saveFeatureDict=featureDict,  # normalization 前傳出來
                          saveJsonPath=paramPath + f'{dataName}_featureTypeDict.json',  # 把 featureDict 存至 json 檔
                          loadJsonPath=None,  # 讀取 featureDict 的 pkl 檔
                          b_loadJson=False)  # True: 讀取 featureDict 的 pkl 檔 (loadJsonPath), False: 把 featureDict 存至 pkl 檔 (saveJsonPath)

encodeDS_TrainDf = encodeObj.dataEncodeOutPut(dataDict = DS_TrainDataDict)
encodeDS_IndpDf = encodeObj.dataEncodeOutPut(dataDict = DS_IndpDataDict)
encodeDS_ValDf = encodeObj.dataEncodeOutPut(dataDict = DS_ValDataDict)

# 儲存 encode 之後的檔案
os.makedirs(featureStatPath, exist_ok=True)
encodeDS_TrainCsvPath = featureStatPath + f'encode_{dataName}_DS_Train.csv'
encodeDS_IndpCsvPath = featureStatPath + f'encode_{dataName}_DS_Indp.csv'
encodeDS_ValCsvPath = featureStatPath + f'encode_{dataName}_DS_Val.csv'
encodeDS_TrainDf.to_csv(encodeDS_TrainCsvPath)
encodeDS_IndpDf.to_csv(encodeDS_IndpCsvPath)
encodeDS_ValDf.to_csv(encodeDS_ValCsvPath)
print(f"encode 後的資料已儲存到 {encodeDS_TrainCsvPath}、{encodeDS_IndpCsvPath} 與 {encodeDS_ValCsvPath}")

# 做normalization
# ======================================================================================================================
# normalization：standard 與 robust 都跑，分別存檔給 Main_MLStkLv1.py 讀取
for normalizeMethod in normalizeMethodList:
    nmlzScalerPath = paramPath + f'{dataName}_{normalizeMethod}Scaler.pkl'

    trainNmlzDf = encodeObj.dataNormalization(encodeTrainDf=encodeDS_TrainDf,
                                              encodeIndpDf=None,  # train scaler存起來，indp 另外做
                                              normalization=normalizeMethod,
                                              saveNmlzScalerPklPath=nmlzScalerPath,
                                              loadNmlzScalerPklPath=None,
                                              b_loadPkl=False)
    indpNmlzDf = encodeObj.dataNormalization(encodeTrainDf=None,
                                             encodeIndpDf=encodeDS_IndpDf,
                                             normalization=normalizeMethod,
                                             saveNmlzScalerPklPath=None,
                                             loadNmlzScalerPklPath=nmlzScalerPath,
                                             b_loadPkl=True)  # indp test set 永遠使用 training set 存好的 NmlzScaler.pkl 檔
    valNmlzDf = encodeObj.dataNormalization(encodeTrainDf=None,
                                            encodeIndpDf=encodeDS_ValDf,
                                            normalization=normalizeMethod,
                                            saveNmlzScalerPklPath=None,
                                            loadNmlzScalerPklPath=nmlzScalerPath,
                                            b_loadPkl=True)  # DS_Val 同樣永遠使用 training set 存好的 NmlzScaler.pkl 檔

    trainNmlzCsvPath = featureStatPath + f'train_{dataName}_{normalizeMethod}.csv'
    indpNmlzCsvPath = featureStatPath + f'indp_{dataName}_{normalizeMethod}.csv'
    valNmlzCsvPath = featureStatPath + f'val_{dataName}_{normalizeMethod}.csv'
    trainNmlzDf.to_csv(trainNmlzCsvPath)
    indpNmlzDf.to_csv(indpNmlzCsvPath)
    valNmlzDf.to_csv(valNmlzCsvPath)
    print(f"[{normalizeMethod}] normalize 後的資料已儲存到 {trainNmlzCsvPath}、{indpNmlzCsvPath} 與 {valNmlzCsvPath}")

    # ==================================================================================================================
    # Feature Stat 分析：找出數值過度集中（top1percent 過高）的 feature 並過濾掉，MotifBitVec 系列 feature 受保護不被過濾
    featureStatObj = FeatureStat(dataDf=trainNmlzDf)
    featureStatObj.sdAnalysis(saveFigPath=featureStatPath + f"sd_analysis_{dataName}_{normalizeMethod}.jpg")
    featureAnalysisXlsxPath = featureStatPath + f"featureAnalysis_{dataName}_{normalizeMethod}.xlsx"
    featureStatObj.featureValuePct_analysis(saveFinalExcel=featureAnalysisXlsxPath)

    filteredTrainNmlzDf, removeList = featureStatObj.processData(xlsxPath=featureAnalysisXlsxPath,
                                                                  columnName='top1percent', number='+0.98',
                                                                  protectFeatSubstringList=['MotifBitVec'])
    featureStatObj.processDataLog(logPath=mlDataPath + f'{dataName}_{normalizeMethod}_')

    filterTrainNmlzPath = featureStatPath + f'filtered_train_{dataName}_{normalizeMethod}.csv'  # 過濾完 feature 後的 nmlz 訓練資料
    filterIndpNmlzPath = featureStatPath + f'filtered_indp_{dataName}_{normalizeMethod}.csv'  # DS_Indp 同步 drop 掉跟訓練集一樣的 feature 後的資料
    filterValNmlzPath = featureStatPath + f'filtered_val_{dataName}_{normalizeMethod}.csv'  # DS_Val 同步 drop 掉跟訓練集一樣的 feature 後的資料
    removeFeatureListPath = featureStatPath + f'remove_feature_list_{dataName}_{normalizeMethod}.json'  # 被過濾掉的 feature 名稱清單
    removeFeatureTypeListPath = featureStatPath + f'remove_featureType_list_{dataName}_{normalizeMethod}.json'  # 被過濾掉的 feature，依所屬 feature type 分組的清單
    filteredTrainNmlzDf.to_csv(filterTrainNmlzPath)

    filteredIndpNmlzDf = indpNmlzDf.drop(columns=removeList)  # DS_Indp 用訓練集算出的 removeList 同步過濾，欄位才會跟訓練集一致
    filteredIndpNmlzDf.to_csv(filterIndpNmlzPath)

    filteredValNmlzDf = valNmlzDf.drop(columns=removeList)  # DS_Val 用訓練集算出的 removeList 同步過濾，欄位才會跟訓練集一致
    filteredValNmlzDf.to_csv(filterValNmlzPath)

    with open(removeFeatureListPath, 'w', encoding='utf-8') as f:
        json.dump(removeList, f, ensure_ascii=False, indent=2)

    removeFeatureTypeDict = {}
    for column in removeList:
        typeName = columnToFeatureType.get(column, 'unknown')
        removeFeatureTypeDict.setdefault(typeName, []).append(column)
    with open(removeFeatureTypeListPath, 'w', encoding='utf-8') as f:
        json.dump(removeFeatureTypeDict, f, ensure_ascii=False, indent=2)

    print(f"[{normalizeMethod}] Feature Stat 過濾完成，共過濾掉 {len(removeList)} 個 feature（分屬 {len(removeFeatureTypeDict)} 個 feature type），"
          f"剩餘 {filteredTrainNmlzDf.shape[1]} 欄，結果已儲存到 {filterTrainNmlzPath}、{filterIndpNmlzPath}、{filterValNmlzPath}、"
          f"{removeFeatureListPath} 與 {removeFeatureTypeListPath}")