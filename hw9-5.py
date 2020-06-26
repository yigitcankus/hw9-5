import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn import datasets, metrics
import numpy as np
import pandas as pd


heartdisease_df = pd.read_csv("cleveland-0_vs_4.csv")

heartdisease_df = heartdisease_df.replace(to_replace='negative', value=0)
heartdisease_df = heartdisease_df.replace(to_replace='positive', value=1)

heartdisease_df["ca"] = heartdisease_df.ca.replace({'<null>':0})
heartdisease_df["ca"] = heartdisease_df["ca"].astype(np.int64)

heartdisease_df["thal"] = heartdisease_df.thal.replace({'<null>':0})
heartdisease_df["thal"] = heartdisease_df["thal"].astype(np.int64)



X = heartdisease_df.iloc[:, :13]
y = heartdisease_df.iloc[:, 13]

scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# #.dbscan yapmadan önce parametre belirlememiz gerekiyor.
# neigh = NearestNeighbors(n_neighbors=2)
# nbrs = neigh.fit(X_std)
# distances, indices = nbrs.kneighbors(X_std)
#
# distances = np.sort(distances, axis=0)
# distances = distances[:,1]
# plt.plot(distances)
# plt.title("verinin distance'ı (std)")
# plt.show()
#
#
# dbscan_cluster = DBSCAN(eps=2, min_samples=4)
#
# clusters = dbscan_cluster.fit_predict(X_std)
#
# pca = PCA(n_components=2).fit_transform(X_std)
#
#
# plt.figure(figsize=(10,5))
# colours = 'rbg'
#
# for i in range(pca.shape[0]):
#     plt.text(pca[i, 0], pca[i, 1], str(clusters[i]),
#              color=colours[y[i]],
#              fontdict={'weight': 'bold', 'size': 50}
#         )
#
# plt.xticks([])
# plt.yticks([])
# plt.axis('off')
# plt.show()



#################################################################################################################

#eps = 1, min_samples = 1, metric =" euclidean " şeklinde parametreleri ayarlayarak DBSCAN uygulayın.
# Ardından, min_samples değerini artırın. Artışın kümelerinin sayısı üzerindeki etkisi nedir?
# min_samples = [1,2,3]
#
# for samples in min_samples:
#     dbscan_cluster = DBSCAN(eps=1, min_samples=samples)
#
#     clusters = dbscan_cluster.fit_predict(X_std)
#
#     pca = PCA(n_components=2).fit_transform(X_std)
#
#     plt.figure(figsize=(10,5))
#     colours = 'rbg'
#     for i in range(pca.shape[0]):
#         plt.text(pca[i, 0], pca[i, 1], str(clusters[i]),
#                  color=colours[y[i]],
#                  fontdict={'weight': 'bold', 'size': 50}
#             )
#     plt.title("Eps=1 , min_samples {0}".format(samples))
#     plt.xticks([])
#     plt.yticks([])
#     plt.axis('off')
#     plt.show()

# min_saples sayısı küçükken daha fazla cluster oluşturuyor çünkü 1 tane ya da 2 tane yan yana bulduğu an yapıyor.

#################################################################################################################

# eps_degeri = [0.5,1,1.5]
#
# for eps in eps_degeri:
#     dbscan_cluster = DBSCAN(eps=eps, min_samples=1)
#
#     clusters = dbscan_cluster.fit_predict(X_std)
#
#     pca = PCA(n_components=2).fit_transform(X_std)
#
#     plt.figure(figsize=(10,5))
#     colours = 'rbg'
#     for i in range(pca.shape[0]):
#         plt.text(pca[i, 0], pca[i, 1], str(clusters[i]),
#                  color=colours[y[i]],
#                  fontdict={'weight': 'bold', 'size': 50}
#             )
#     plt.title("Eps= {} , min_samples 1".format(eps))
#     plt.xticks([])
#     plt.yticks([])
#     plt.axis('off')
#     plt.show()
#
# # min_sample 1 olduğu sürece her bir noktayı cluster olarak alıcak. eps değiştirmemiz bir şey değiştirmez
