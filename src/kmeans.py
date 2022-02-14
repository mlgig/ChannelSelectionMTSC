import sys
import os
sys.path.insert(0, os.getcwd())
sys.path.append(".")
sys.path.append("..")

import numpy as np
import pandas as pd
from sktime.utils.data_io import load_from_tsfile_to_dataframe
from src.shrunk_cent import shrunk_centroid
#from scripts.dim_selection import dimension
from src.calc_distance import distance_matrix
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize


class kmeans(TransformerMixin, BaseEstimator):
    def __init__(self, distance = "eu", shrinkage=0, center="mean"):
        self.shrinkage=shrinkage
        self.distancefn=distance
        self.center=center
    
    def fit(self, X, y):
        centroid_obj = shrunk_centroid(0)
        df = centroid_obj.create_centroid(X.copy(),y, center=self.center)
        obj= distance_matrix(distance = self.distancefn)
        self.distance_frame= obj.distance(df)
        # l2 normalisng for kmeans
        self.distance_frame = pd.DataFrame(normalize(self.distance_frame, axis=0),columns= self.distance_frame.columns.tolist())
        
        
        self.kmeans = KMeans(n_clusters=2, random_state=0).fit(self.distance_frame)
        # Find the cluster name with maximum avg distance
        self.cluster = np.argmax(self.kmeans.cluster_centers_.mean(axis=1))
        self.relevant_dims = [id_ for id_, item in enumerate(self.kmeans.labels_) if item==self.cluster]
        
        return self

    def transform(self, X):
        #print("Dimension used: ", X.iloc[:, self.relevant_dims].shape[1]/X.shape[1])
        return X.iloc[:, self.relevant_dims]     

if __name__ == '__main__':

    dataset = ['FullUnnormalized', 'Normalized', 'Unnormalized']
    #dataset = ['DuckDuckGeese', 'FaceDetection', 'MotorImagery', 'PEMS-SF']
    #dataset = ['FullUnnormalized']
    for item in dataset:
        print(item)

        #NOTE: Code to read Jump Dataset
        train_x, train_y = load_from_tsfile_to_dataframe(f"../MP/{item}/TRAIN_X.ts", return_separate_X_and_y=True)
        #train_x = load_from_tsfile_to_dataframe(f"./data/{item}/{item}_TRAIN.ts", return_separate_X_and_y=False)
        print(f"{item} \nShape: {train_x.shape} ")
        
        km = kmeans()
        km.fit(train_x,train_y)
        df = km.transform(train_x)
        print(df.shape)
        print(km.relevant_dims)
        break
