# Databricks notebook source
import pandas as pd
import numpy as np
import pyspark.sql.functions as f

dbutils.library.installPyPI("gap-stat", version="1.5.2")
dbutils.library.installPyPI("mlflow")
import mlflow
from mlflow import spark as mlsp

ORG = "intuit"
#INPUT_FOLDER = "s3a://bigpanda-data-lake/mermer/changes/datasets"
#OUTPUT_FOLDER = "s3a://bigpanda-data-lake/mermer/changes/output"

KMEANS_K = 50
INPUT_FOLDER = "s3a://bigpanda-data-lake/mermer/incidents/datasets"
OUTPUT_FOLDER = "s3a://bigpanda-data-lake/mermer/incidents/output"

# COMMAND ----------

#features_path = "%s/%s_changes_features.parquet" % (OUTPUT_FOLDER, ORG)
#features_df = spark.read.parquet(features_path)

features_path = "%s/%s_clustered_incidents_k-%d" % (OUTPUT_FOLDER, ORG, KMEANS_K)
features_df = spark.read.parquet(features_path)


def extract(row):
    return tuple(row.features.toArray().tolist())

features_pd = features_df.rdd.map(extract).toDF().toPandas()

# COMMAND ----------

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.decomposition import PCA

# COMMAND ----------

X = features_pd
n_clusters = 30

# COMMAND ----------

pca = PCA()
pca.fit(X)  
fig = plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');
display(fig)

# COMMAND ----------

pca = PCA(n_components=46)

# COMMAND ----------

kmeans = KMeans(n_clusters=n_clusters, random_state=10)
cluster_labels = kmeans.fit_predict(X)
sample_silhouette_values = silhouette_samples(X, cluster_labels)
silhouette_values = pd.DataFrame(data={'label':cluster_labels, 'sil_val':sample_silhouette_values})
cluster_silhouette_values = silhouette_values.groupby('label').sil_val.mean()
cluster_size = pd.Series(cluster_labels).value_counts().sort_index()

# COMMAND ----------

noisy_cluster_silhouette_values = pd.DataFrame()
noisy_cluster_silhouette_values_shuffled = pd.DataFrame()
  
def calculate(sigma, R=100):
  Xn = X + np.random.normal(0, sigma, X.shape)
  sample_silhouette_values_noise = silhouette_samples(Xn, cluster_labels)
  silhouette_values_noise = pd.DataFrame(data={'label':cluster_labels, 'sil_val':sample_silhouette_values_noise})
  noisy_cluster_silhouette_values[sigma] = silhouette_values_noise.groupby('label').sil_val.mean()
  
  shuffled_silhouette = pd.DataFrame(columns=range(R), index=noisy_cluster_silhouette_values.index)
  for r in range(R):
    shuffled_cluster_labels = pd.Series(cluster_labels).sample(len(cluster_labels))
    sample_silhouette_values_noise_shuffled = silhouette_samples(Xn, shuffled_cluster_labels)
    silhouette_values_noise_shuffled = pd.DataFrame(data={'label':shuffled_cluster_labels, 'sil_val':sample_silhouette_values_noise_shuffled})
    shuffled_silhouette[r] = silhouette_values_noise_shuffled.groupby('label').sil_val.mean()
    
  noisy_cluster_silhouette_values_shuffled[sigma] = shuffled_silhouette.mean(axis = 1)
  significant = shuffled_silhouette.apply(lambda x: noisy_cluster_silhouette_values[sigma] > x).sum(axis=1) / R
  return (sigma, significant, noisy_cluster_silhouette_values, noisy_cluster_silhouette_values_shuffled)


def merge(results):
  cluster_thresh = pd.DataFrame()
  for significant in results:
    cluster_thresh[significant[0]] = significant[1]
  return cluster_thresh

# COMMAND ----------

from multiprocessing.pool import ThreadPool

with ThreadPool(32) as p:
  r = p.map(calculate, np.arange(0.1, 2.1, 0.1))
rpd, real_cluster_scores, mean_shuffled_scores = merge(r)


# COMMAND ----------

cluster_score = rpd.apply(lambda x: (x < 0.95).idxmax(), axis=1)

# COMMAND ----------

cluster_score.sort_values()

# COMMAND ----------

cluster_silhouette_values.sort_values()

# COMMAND ----------

noisy_cluster_silhouette_values = noisy_cluster_silhouette_values[noisy_cluster_silhouette_values.columns.sort_values()]
noisy_cluster_silhouette_values_shuffled = noisy_cluster_silhouette_values_shuffled[noisy_cluster_silhouette_values_shuffled.columns.sort_values()]

# COMMAND ----------

fig = plt.figure(figsize=(14,10))
for c in range(n_clusters):
  ax = fig.add_subplot(6,5,c+1)
  ax.plot(real_cluster_scores.loc[c])
  ax.plot(mean_shuffled_scores.loc[c])
  ax.grid()
display(fig)

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

