import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def read_csv(FILENAME):
	return pd.read_csv(FILENAME)

def one_hot_encode(df, column_name):
	return pd.get_dummies(df, columns=column_name)

def label_encode(df):
	le = LabelEncoder()
	encoded = df.apply(LabelEncoder().fit_transform)
	return encoded

def run_pca(df):
	scaler = StandardScaler()
	scaled_np = scaler.fit_transform(df)
	pca = PCA(n_components='mle')
	pca_results_np = pca.fit_transform(scaled_np)
	pca_df = pd.DataFrame(pca_results_np)
	return pca_df

def plot_3D(df):
	threedee = plt.figure().gca(projection='3d')
	threedee.scatter(df[0], df[1], df[2], c=df['Cluster'])
	threedee.set_xlabel('PC 0')
	threedee.set_ylabel('PC 1')
	threedee.set_zlabel('PC 2')
	plt.show()

def get_cluster_numbers(df, num_clusters):
	kmeans = KMeans(n_clusters=num_clusters)
	cluster_numbers = kmeans.fit_predict(df)
	return cluster_numbers

def add_cluster_column(df, cluster_np):
	df['Cluster'] = cluster_np
	return df

def split_train_test(X_df, Y_df, test_size):
	X_train, X_test, y_train, y_test = train_test_split(
		X_df, Y_df, test_size=test_size)
	return X_train, X_test, y_train, y_test

def get_num_features(df):
	return len(df.columns)



