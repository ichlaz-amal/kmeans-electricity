import pandas as pd
import numpy as np

# INITIAL PLOTTING VALUE
my_household = 'MAC000002'
my_year = 2012
my_month = 10
my_day = 13

# READ CSV FILE
df = pd.read_csv('hourlyid_dataset/MAC000002.csv')
df = df.append(pd.read_csv('hourlyid_dataset/MAC000246.csv'))
df = df.append(pd.read_csv('hourlyid_dataset/MAC000450.csv'))
df = df.append(pd.read_csv('hourlyid_dataset/MAC001074.csv'))
df = df.append(pd.read_csv('hourlyid_dataset/MAC003223.csv'))
df = df.append(pd.read_csv('hourlyid_dataset/MAC003239.csv'))
df = df.append(pd.read_csv('hourlyid_dataset/MAC003252.csv'))
df = df.append(pd.read_csv('hourlyid_dataset/MAC003281.csv'))
df = df.append(pd.read_csv('hourlyid_dataset/MAC003305.csv'))
df = df.append(pd.read_csv('hourlyid_dataset/MAC003348.csv'))
df = df.append(pd.read_csv('hourlyid_dataset/MAC003388.csv'))

df.reset_index(drop=True, inplace=True)
df_copy = df.copy()
df.head(10)

# CONVERT TO NUMBER
d = {'Holiday': np.array([0] * len(df)),
    'Acorn_Group': np.array([0] * len(df))}
data = pd.DataFrame(data=d)
for i in range(len(df)):
    if(df['Holiday'][i] == 'Yes'):
        data.at[i, 'Holiday'] = 1

df['Holiday'] = data['Holiday']
df['Acorn_Group'] = data['Acorn_Group']

df.head(10)

# STANDARDIZATION
from sklearn.preprocessing import StandardScaler

scaled = df.loc[:, ['Temperature', 'kWh']].values
scaled = StandardScaler().fit_transform(scaled)
data = pd.DataFrame(data=scaled, columns=['Temperature', 'kWh'])

df['Temperature'] = data['Temperature']
df['kWh'] = data['kWh']

df.head(10)

# PCA PROJECTION
from sklearn.decomposition import PCA

scaled = df.loc[:, ['Day', 'Month', 'Hour']].values
scaled = PCA(n_components=1).fit_transform(scaled)

data = pd.DataFrame(data=scaled, columns=['Component'])
data.insert(0, 'ID_House', df['ID_House'])
data.insert(2, 'Holiday', df['Holiday'])
data.insert(3, 'Temperature', df['Temperature'])
data.insert(4, 'kWh', df['kWh'])

data.head(10)

# DETERMINE NUMBER OF CLUSTER
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
%matplotlib inline

x = data.loc[:, ['Holiday', 'Temperature', 'kWh']].values
distortions = []
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, random_state=0).fit(x)
    distortions.append(kmeans.inertia_)
    
plt.plot(range(1, 10), distortions, marker='o')
plt.title('Elbow Method For Optimal k')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.show()

# K-MEANS RESULT
kmeans = KMeans(n_clusters=5, random_state=0).fit(x)
df_copy.insert(9, 'Cluster', kmeans.labels_)

plt.scatter(df_copy.index, df_copy['kWh'], c=df_copy['Cluster']);
plt.xlabel('Time')
plt.ylabel('kWh')
plt.show()
plt.scatter(df_copy.index, df_copy['Temperature'], c=df_copy['Cluster']);
plt.xlabel('Time')
plt.ylabel('Temperature')
plt.show()
plt.scatter(df_copy.index, df_copy['Cluster'], c=df_copy['Cluster']);
plt.show()

df_copy = df_copy[df_copy['ID_House'] == my_household]
plt.scatter(df_copy.index, df_copy['kWh'], c=df_copy['Cluster']);
plt.xlabel('Time')
plt.ylabel('kWh')
plt.show()
plt.scatter(df_copy.index, df_copy['Temperature'], c=df_copy['Cluster']);
plt.xlabel('Time')
plt.ylabel('Temperature')
plt.show()
plt.scatter(df_copy.index, df_copy['Cluster'], c=df_copy['Cluster']);
plt.show()

df_copy = df_copy[(df_copy['ID_House'] == my_household) & (df_copy['Year'] == my_year)]
plt.scatter(df_copy.index, df_copy['kWh'], c=df_copy['Cluster']);
plt.xlabel('Time')
plt.ylabel('kWh')
plt.show()
plt.scatter(df_copy.index, df_copy['Temperature'], c=df_copy['Cluster']);
plt.xlabel('Time')
plt.ylabel('Temperature')
plt.show()
plt.scatter(df_copy.index, df_copy['Cluster'], c=df_copy['Cluster']);
plt.show()

df_copy = df_copy[(df_copy['ID_House'] == my_household) & (df_copy['Year'] == my_year) & (df_copy['Month'] == my_month)]
plt.scatter(df_copy.index, df_copy['kWh'], c=df_copy['Cluster']);
plt.xlabel('Time')
plt.ylabel('kWh')
plt.show()
plt.scatter(df_copy.index, df_copy['Temperature'], c=df_copy['Cluster']);
plt.xlabel('Time')
plt.ylabel('Temperature')
plt.show()
plt.scatter(df_copy.index, df_copy['Cluster'], c=df_copy['Cluster']);
plt.show()

df_copy = df_copy[(df_copy['ID_House'] == my_household) & (df_copy['Year'] == my_year) &
                  (df_copy['Month'] == my_month) & (df_copy['Day'] == my_day)]
plt.scatter(df_copy.index, df_copy['kWh'], c=df_copy['Cluster']);
plt.xlabel('Time')
plt.ylabel('kWh')
plt.show()
plt.scatter(df_copy.index, df_copy['Temperature'], c=df_copy['Cluster']);
plt.xlabel('Time')
plt.ylabel('Temperature')
plt.show()
plt.scatter(df_copy.index, df_copy['Cluster'], c=df_copy['Cluster']);
plt.show()