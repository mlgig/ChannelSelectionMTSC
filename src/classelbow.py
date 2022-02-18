#Elbow with class pairwise ECP
from cProfile import label
from math import fabs
import sys 
import os
from tkinter.tix import InputOnly
from traceback import print_tb
sys.path.insert(0, os.getcwd())
sys.path.append(".")
sys.path.append("..")

import numpy as np
import pandas as pd
from sktime.utils.data_io import load_from_tsfile_to_dataframe
from .shrunk_cent import shrunk_centroid
from .calc_distance import distance_matrix
from .utils import detect_knee_point
from collections import Counter
from sklearn.base import TransformerMixin, BaseEstimator
#from dataset import dataset_
import collections
from scipy.stats import rankdata 
from numpy.linalg import norm


class ElbowPair(TransformerMixin, BaseEstimator):
    """
    Class of extract dimension from each class pair
    inp: Shrinkage

    """
    def __init__(self, distance = "eu", shrinkage=0, center = "mad", mc =False, fft=False):
        self.shrinkage=shrinkage
        self.distancefn=distance
        self.center = center
        self.mc = mc
        self.fft = fft
        
    def _countFrequency(self, arr):
        return collections.Counter(arr)

    def _create_rankDictionary(self, arr, dic):
        for element in arr:
            dic[element] = arr.index(element)
        print(dic)

    def _rank(self):
        all_index = self.distance_frame.sum(axis=1).sort_values(ascending=False).index    
        series = self.distance_frame.sum(axis=1)
        series.drop(index=list(set(all_index)-set(self.relevant_dims)), inplace=True)
        return series.sort_values(ascending=False).index
        
    def _rank_frequ(self):    
        
        channel_dist={}
        channel_frequency = Counter(self.relevant_dims)

        
        for key, value in zip(self.relevant_dims, self.relevant_dis):
            if key not in channel_dist:
                channel_dist.update({key: value})
            else:
                channel_dist[key]=channel_dist[key]+value
    

        freq = pd.DataFrame(channel_frequency.items(), columns=['channel', 'frequency'])
        dist = pd.DataFrame(channel_dist.items(), columns=['channel', 'distance'])
        df = pd.merge(freq, dist, on='channel')
        return df.sort_values(by=['frequency','distance'], ascending=False).channel.tolist()

    def fit(self, X, y):
        
        centroid_obj = shrunk_centroid(self.shrinkage)
        self.centroid_frame = centroid_obj.create_centroid(X.copy(),y, center=self.center, mean_centering=self.mc, _fft=self.fft) # Centroid created here
        obj = distance_matrix(distance = self.distancefn)
        self.distance_frame = obj.distance(self.centroid_frame.copy()) # Distance matrix created here

        
        all_chs = np.empty(self.centroid_frame.shape[1] - 1) # -1 for removing class columsn
        all_chs.fill(0)
        chs_freq = np.zeros(self.centroid_frame.shape[1]- 1)
        all_dis = np.zeros(self.centroid_frame.shape[1]- 1)
        self.relevant_dims = []
        self.relevant_dis = []
        

        for pairdistance in self.distance_frame.iteritems():
            dic = {}
            distance = pairdistance[1].sort_values(ascending=False).values
            indices = pairdistance[1].sort_values(ascending=False).index 
            chs_dis = detect_knee_point(distance, indices)           
            chs = chs_dis[0]
            dis = chs_dis[1]
            self.relevant_dims.extend(chs)

        self.rank = self._rank() 
        self.relevant_dims = list(set(self.relevant_dims))
        return self

    
    def transform(self, X):
        return X.iloc[:, self.relevant_dims]


if __name__ == '__main__':


    dataset = ['ArticularyWordRecognition'] #dataset_
    #dataset = ["Epilepsy","EthanolConcentration", "Handwriting", "UWaveGestureLibrary", "AtrialFibrillation", "Libras", "PenDigits"]
    
    for item in dataset:
        #print(item)

        #NOTE: Code to read Jump Dataset
        #train_x, train_y = load_from_tsfile_to_dataframe("./MP/FullUnnormalized25/TRAIN_default_X.ts", return_separate_X_and_y=True)
        train_x, train_y = load_from_tsfile_to_dataframe("./MP/FullUnnormalized25/TRAIN_default_X.ts", return_separate_X_and_y=True)
        #train_x, train_y = load_from_tsfile_to_dataframe(f"./data/{item}/{item}_TRAIN.ts", return_separate_X_and_y=True)
        
        
        print(f"{item} \nShape: {train_x.shape} ")
        
        obj = ElbowPair(distance='eu', shrinkage=0)
        obj.fit(train_x, train_y)
        print("RS:",obj.relevant_dims)
        df = obj.transform(train_x)
        #print("Classes: ", len(np.unique(train_y)))
        #print(df.relevant_dims)
    #pass
