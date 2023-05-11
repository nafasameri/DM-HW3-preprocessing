import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 
           'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 
           'hours-per-week', 'native-country', 'income']
adult_data = pd.read_csv('data/adult.data.csv', names=columns)


# ========================== 1
numeric_features = adult_data.describe().columns.tolist()
print('numeric features:', numeric_features)

df = adult_data[numeric_features]
df.to_csv('numeric features.csv')
# print(df.to_markdown())


# ========================== 2
adult_data.fillna(value=np.nan, inplace=True)
# print(adult_data.to_markdown())
adult_data.to_csv('adult_data nan.csv')


# ========================== 3
numeric_data = adult_data[numeric_features]
for col in numeric_data.columns:
    if numeric_data[col].dtype == 'float64' or numeric_data[col].dtype == 'int64':
        mean_value = numeric_data[col].mean()
        numeric_data[col].fillna(value=mean_value, inplace=True)

adult_data[numeric_features] = numeric_data
adult_data.to_csv('adult_data mean.csv')


# ========================== 4
min_vals = numeric_data.min()
max_vals = numeric_data.max()

scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(numeric_data)

adult_data[numeric_features] = normalized_data
# print(normalized_data)
adult_data.to_csv('adult_data normalized.csv')


# ========================== 5
print(adult_data['age'].describe())
age_group = pd.cut(adult_data['age'], bins=11, labels=False)
age_group.hist()
plt.title('age group')
plt.xlabel('group')
plt.ylabel('count')
plt.savefig('age_group.png')


# ========================== 6
cov_matrix = adult_data.cov()
corr_matrix = adult_data.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(cov_matrix, cmap=sns.cubehelix_palette(as_cmap=True), annot=True, fmt=".1f")
plt.savefig('cov_matrix.png')

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, cmap=sns.cubehelix_palette(as_cmap=True), annot=True, fmt=".1f")
plt.savefig('corr_matrix.png')


print(len(adult_data.columns))
threshold = 0.8
corr_features = set()
for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if abs(corr_matrix.iloc[i, j]) > threshold:
            colname = corr_matrix.columns[i]
            corr_features.add(colname)
            if colname in adult_data.columns:
                print(colname)
                del adult_data[colname]
print(len(adult_data.columns))


corr_matrix = numeric_data.corr()
print(len(numeric_data.columns))
threshold = 0.8
corr_features = set()
for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if abs(corr_matrix.iloc[i, j]) > threshold:
            colname = corr_matrix.columns[i]
            corr_features.add(colname)
            if colname in numeric_data.columns:
                print(colname)
                del numeric_data[colname]
print(len(numeric_data.columns))
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, cmap=sns.cubehelix_palette(as_cmap=True), annot=True, fmt=".1f")
plt.savefig('corr_matrix_num.png')