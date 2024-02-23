# 1 - import packges : 
import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# 2 - reed the file : 
df = pd.read_csv('C:\\Users\\moham\\Desktop\\VScode\\Mall Customer\\Mall_Customers.csv')
df = pd.DataFrame(df)

# 3 - data preprocessing :
df = df.drop(columns=['CustomerID'])
df.reset_index(drop=True, inplace=True)

from sklearn.preprocessing import LabelEncoder
col = ['Gender']
df[col] = df[col].apply(LabelEncoder().fit_transform)

# 4 - spliting the data :
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# 5 - calculate 'within cluster sum of square' :
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)

    wcss.append(kmeans.inertia_)

# 6 - plot the graph :
sns.set()
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Point Graph')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# 7 - fit the model :  
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=0)
Y = kmeans.fit_predict(X)
