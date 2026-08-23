
# # ======================================================================================================================
# # normalization：分別跑 standard 與 robust 兩種方式
# for normalizeMethod in normalizeMethodList:
#     nmlzScalerPath = paramPath + f'{dataName}_{normalizeMethod}Scaler.pkl'

#     trainNmlzDf = encodeObj.dataNormalization(encodeTrainDf=encodeTrainDf,
#                                               encodeIndpDf=None,  # train scaler存起來 ，indp 另外做
#                                               normalization=normalizeMethod,
#                                               saveNmlzScalerPklPath=nmlzScalerPath,
#                                               loadNmlzScalerPklPath=None,
#                                               b_loadPkl=False)  # True: 讀取 NmlzScaler 的 pkl 檔 (loadNmlzScalerPklPath)
#     # False: 把 NmlzScaler 存至 pkl 檔 (saveNmlzScalerPklPath)

#     indpNmlzDf = encodeObj.dataNormalization(encodeTrainDf=None,
#                                              encodeIndpDf=encodeIndpDf,
#                                              normalization=normalizeMethod,
#                                              saveNmlzScalerPklPath=None,
#                                              loadNmlzScalerPklPath=nmlzScalerPath,
#                                              b_loadPkl=True)  # indp test set 永遠使用 training set 存好的 NmlzScaler.pkl 檔

#     trainNmlzCsvPath = featureStatPath + f'train_{dataName}_{normalizeMethod}.csv'
#     indpNmlzCsvPath = featureStatPath + f'indp_{dataName}_{normalizeMethod}.csv'

#     trainNmlzDf.to_csv(trainNmlzCsvPath)
#     indpNmlzDf.to_csv(indpNmlzCsvPath)
