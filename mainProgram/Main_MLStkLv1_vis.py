# Prevent chart windows from popping up
import matplotlib
matplotlib.use('Agg')

# Enable this when compiling with vscode
import sys, os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent = os.path.join(current_dir, "..")
sys.path.append(parent)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ======================================================================================================================
# Base paths
mlScorePath = "../data/mlScore/"  # Contains the featureType_model_mccScore csv produced by Main_MLStkLv1.py
visPath = "../data/mlScore/vis/"  # Output path for the charts
dataName = 'NeuroP_1'

os.makedirs(visPath, exist_ok=True)

# ======================================================================================================================
# Load the MCC score table for every normalizeMethod x feature type x model, saved by Main_MLStkLv1.py
resultCsvPath = mlScorePath + f'featureType_model_mccScore_{dataName}.csv'
resultDf = pd.read_csv(resultCsvPath)
print(f"Loaded MCC score table: {resultDf.shape}")

# Drop rows where training failed (mcc is null)
validDf = resultDf.dropna(subset=['mcc']).copy()
print(f"Remaining after dropping failed combinations: {validDf.shape[0]} / {resultDf.shape[0]}")

# ======================================================================================================================
# Chart 1: MCC distribution across all model combinations (normalizeMethod x feature type x model), bin width 0.05
binWidth = 0.05
binStart = np.floor(validDf['mcc'].min() / binWidth) * binWidth
binEnd = np.ceil(validDf['mcc'].max() / binWidth) * binWidth
binCount = int(round((binEnd - binStart) / binWidth)) + 1
binEdges = np.round(np.linspace(binStart, binEnd, binCount), 2)

plt.figure(figsize=(14, 8), dpi=300)
counts, edges, patches = plt.hist(validDf['mcc'], bins=binEdges, color='steelblue', edgecolor='black')
for count, edge in zip(counts, edges[:-1]):
    if count > 0:
        plt.text(edge + binWidth / 2, count, int(count), ha='center', va='bottom', fontsize=9)
plt.xlabel('MCC', fontsize=14)
plt.ylabel('Number of model combinations', fontsize=14)
plt.title(f'MCC Distribution Across All Model Combinations (n={validDf.shape[0]}, bin width={binWidth})', fontsize=14)
plt.xticks(binEdges, rotation=45, fontsize=9)
plt.tight_layout()
mccDistPngPath = visPath + f'mccDistribution_{dataName}.png'
plt.savefig(mccDistPngPath)
plt.close()
print(f"Chart 1 saved: {mccDistPngPath}")

# ======================================================================================================================
# Chart 2: MCC score box plot per feature type (sorted by median, descending)
featureTypeOrder = (validDf.groupby('featureType')['mcc']
                     .median()
                     .sort_values(ascending=False)
                     .index.tolist())

figHeight = max(8, len(featureTypeOrder) * 0.4)
plt.figure(figsize=(12, figHeight), dpi=300)
sns.boxplot(data=validDf, x='mcc', y='featureType', order=featureTypeOrder, color='steelblue')
plt.xlabel('MCC', fontsize=14)
plt.ylabel('Feature Type', fontsize=14)
plt.title('MCC Score Box Plot by Feature Type (Sorted by Median)', fontsize=14)
plt.yticks(fontsize=9)
plt.tight_layout()
featureTypeBoxPngPath = visPath + f'featureType_mccBoxplot_{dataName}.png'
plt.savefig(featureTypeBoxPngPath)
plt.close()
print(f"Chart 2 saved: {featureTypeBoxPngPath}")
