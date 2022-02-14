#ElbowCut
import sys
import os
sys.path.insert(0, os.getcwd())
sys.path.append(".")
sys.path.append("..")

import numpy as np
import pandas as pd
from sktime.utils.data_io import load_from_tsfile_to_dataframe
from .shrunk_cent import shrunk_centroid
from .calc_distance import distance_matrix
from .utils import detect_knee_point
from sklearn.base import TransformerMixin, BaseEstimator
from numpy.linalg import norm

class elbow(TransformerMixin, BaseEstimator):

    def __init__(self, distance = "eu", shrinkage=0, center = "mean",mc =False, fft=False):
        self.shrinkage=shrinkage
        self.distancefn=distance
        self.center = center
        self.mc = mc
        self.fft = fft


    def fit(self, X, y):
        
        centroid_obj = shrunk_centroid(0)
        self.centroid_frame = centroid_obj.create_centroid(X.copy(),y, center=self.center, mean_centering=self.mc, _fft=self.fft)
        obj= distance_matrix(self.distancefn)
        self.distance_frame = obj.distance(self.centroid_frame.copy())
        #self.distance_frame = self.centroid_frame.iloc[:,:-1].applymap(norm) #TODO: L2 Norm

        #self.distance_frame = self.distance_frame.T
        #self.distance_frame.reset_index(drop=True, inplace=True)
        self.relevant_dims = []
        distance = self.distance_frame.sum(axis=1).sort_values(ascending=False).values
        indices = self.distance_frame.sum(axis=1).sort_values(ascending=False).index
        #print(detect_knee_point(distance, indices))
        self.relevant_dims.extend(detect_knee_point(distance, indices)[0])
        self.rank=self.relevant_dims       
        
        return self

    
    def transform(self, X):
        #print(self.relevant_dims)
        #print("Dimension used: ", X.iloc[:, self.relevant_dims].shape[1]/X.shape[1])
        return X.iloc[:, self.relevant_dims]


if __name__ == '__main__':
    
    dataset = ['AtrialFibrillation']
    for item in dataset:
        print(item)

        #NOTE: Code to read Jump Dataset
        #train_x, train_y = load_from_tsfile_to_dataframe(f"./MP/{item}/TRAIN_X.ts", return_separate_X_and_y=True)
        train_x, train_y = load_from_tsfile_to_dataframe("./MP/FullUnnormalized25/TRAIN_default_X.ts", return_separate_X_and_y=True)
        #train_x, train_y = load_from_tsfile_to_dataframe(f"../data/{item}/{item}_TRAIN.ts", return_separate_X_and_y=True)
        #print(f"{item} \nShape: {train_x.shape} ")
        
        obj = elbow()
        obj.fit(train_x, train_y)
        print("RS: ",obj.relevant_dims)
        df = obj.transform(train_x)
        print(obj.distance_frame)
        print(df.shape)
    #pass
