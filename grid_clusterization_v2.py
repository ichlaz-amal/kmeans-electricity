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

# DETERMINE NUMBER OF CLUSTER
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
%matplotlib inline

x = df.loc[:, ['Holiday', 'Temperature', 'kWh']].values
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
df.insert(9, 'Cluster', kmeans.labels_)
df.head(10)

plt.scatter(df.index, df['kWh'], c=df['Cluster'])
plt.ylabel('kWh'); plt.xticks([]); plt.show()

plt.scatter(df.index, df['Temperature'], c=df['Cluster'])
plt.ylabel('Temperature'); plt.xticks([]); plt.show()

df = df[df['ID_House'] == my_household]

plt.scatter(df.index, df['kWh'], c=df['Cluster'])
plt.ylabel('kWh'); plt.xticks([]); plt.show()

plt.scatter(df.index, df['Temperature'], c=df['Cluster'])
plt.ylabel('Temperature'); plt.xticks([]); plt.show()

df = df[df['Year'] == my_year]

plt.scatter(df.index, df['kWh'], c=df['Cluster'])
plt.ylabel('kWh'); plt.xticks([]); plt.show()

plt.scatter(df.index, df['Temperature'], c=df['Cluster'])
plt.ylabel('Temperature'); plt.xticks([]); plt.show()

df = df[df['Month'] == my_month]

plt.scatter(df.index, df['kWh'], c=df['Cluster'])
plt.ylabel('kWh'); plt.xticks([]); plt.show()

plt.scatter(df.index, df['Temperature'], c=df['Cluster'])
plt.ylabel('Temperature'); plt.xticks([]); plt.show()

df = df[df['Day'] == my_day]

plt.scatter(df.index, df['kWh'], c=df['Cluster'])
plt.ylabel('kWh'); plt.xticks([]); plt.show()

plt.scatter(df.index, df['Temperature'], c=df['Cluster'])
plt.ylabel('Temperature'); plt.xticks([]); plt.show()