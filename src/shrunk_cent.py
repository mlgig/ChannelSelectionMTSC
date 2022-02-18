#from numpy.core.fromnumeric import shape
import pandas as pd
import sys
import os

sys.path.insert(0, os.getcwd())
sys.path.append(".")
sys.path.append("..")

import numpy as np
from sktime.utils.data_io import load_from_tsfile_to_dataframe
from operator import truediv 
from sktime.utils.data_processing import from_nested_to_3d_numpy, from_3d_numpy_to_nested
from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import LabelEncoder
from scipy.signal import savgol_filter
from sklearn import preprocessing
from scipy.fftpack import fft, ifft
from  scipy.stats import median_abs_deviation as mad


class shrunk_centroid:

    def __init__(self, shrink):
        self.shrink = shrink


    def _mad_median(self, class_X, median=None):
        #print((class_X).shape)

        #if not median:
        #    median = np.median(class_X, axis=0)

        _mad =  mad(class_X, axis=0)

        low_value = median - _mad * 0.50
        high_value = median + _mad * 0.50
        clip = lambda x: np.clip(x, low_value, high_value)
        class_X = np.apply_along_axis(clip, axis=1, arr=class_X) 
        #print("class_X: ", class_X.shape)

        return np.mean(class_X, axis=0), class_X

    def _class_medianc(self, X, y):

        classes_ = np.unique(y)

        #channel_median = np.empty(shape=(len(classes_), X.shape[1])) # classes * length
        channel_median = []
        for class_ in classes_: #for every class
            class_idx = np.where(y==class_) # find the indexes of data point where particular class is located
            dts = X[class_idx]
            class_median = np.median(dts, axis=0)
            class_mean = np.mean(dts, axis=0)
            _mad =  mad(dts, axis=0)
            
            low_value = class_median - _mad * 0.50
            high_value = class_median + _mad * 0.50
            #print("HV: ", high_value)
            print(high_value-class_mean)
            print(class_mean-low_value)

            clip = lambda x: np.clip(x, low_value, high_value)
            #class_mean = np.apply_along_axis(clip, axis=1, arr=class_mean) 

            channel_median.append(class_median)
            #break
        #print(np.array(channel_median).shape)
        return np.array(channel_median)

    def _class_mad_median(self, X, y):

        classes_ = np.unique(y)

        #channel_median = np.empty(shape=(len(classes_), X.shape[1])) # classes * length
        channel_median = []
        for class_ in classes_:
            class_idx = np.where(y==class_) # find the indexes of data point where particular class is located
            
            class_median = np.median(X[class_idx], axis=0)
            class_median = self._mad_median(X[class_idx], class_median)[0]
            channel_median.append(class_median)
            #break
        #print(np.array(channel_median).shape)
            
        return np.array(channel_median)
 
    def _class_median(self, X, y):

        classes_ = np.unique(y)

        #channel_median = np.empty(shape=(len(classes_), X.shape[1])) # classes * length
        channel_median = []
        for class_ in classes_:
            class_idx = np.where(y==class_) # find the indexes of data point where particular class is located
            class_median = np.median(X[class_idx], axis=0)
            channel_median.append(class_median)
        #print(np.array(channel_median).shape)
        return np.array(channel_median)

    def _shrink_median(self, X, y):

        classes_ = np.unique(y)

        #channel_median = np.empty(shape=(len(classes_), X.shape[1])) # classes * length
        channel_data = []
        for class_ in classes_:
            class_idx = np.where(y==class_) # find the indexes of data point where particular class is located
            class_median = np.median(X[class_idx], axis=0)
            class_data = self._mad_median(X[class_idx], class_median)[1] # Get the fixed dataset for a class across channels
            channel_data.append(class_data)
            #print("class_data: ", class_data.shape)
        fixed_data = np.concatenate(channel_data, axis=0)
        return fixed_data #TODO: This should be data of a num_sample * time_length in 2D format


    def _class_std(self, X, y):

        classes_ = np.unique(y)

        #channel_median = np.empty(shape=(len(classes_), X.shape[1])) # classes * length
        channel_std = []
        for class_ in classes_:
            class_idx = np.where(y==class_) # find the indexes of data point where particular class is located
            class_std = np.std(X[class_idx], axis=0)
            channel_std.append(class_std)
            
        return np.array(channel_std)


    def _filter(self, p):
        return pd.Series(savgol_filter(p.values, 7, 1))
    
    def mean_centering(self, centroid):
        return centroid.subtract(centroid.mean())

    def _fft(self, signal):

        freq_domain = fft(signal.values)
        PSD = freq_domain * np.conj(freq_domain)/ len(freq_domain)
        index = PSD>=0.75
        clean = PSD*index
        cleaned_freq = index * freq_domain
    
        return pd.Series(ifft(cleaned_freq).real)
            
    
    def create_centroid(self, X, y, center = "mad",_filter=False, mean_centering=False, _fft=False):
        """
        Creating the centroid for each class
        """
        #y = X.class_vals
        #X.drop('class_vals', axis = 1, inplace = True)
        cols = X.columns.to_list()   
        ts = from_nested_to_3d_numpy(X) # Contains TS in numpy format
        centroids = []
        temp = []

        le = LabelEncoder()
        y_ind = le.fit_transform(y)

        #print(dict(zip(le.classes_, le.transform(le.classes_))))

        print(f"Centroid type: {center}")
        for dim in range(ts.shape[1]): # iterating over channels
            train  = ts[:, dim, :]
            if center == "mean":
                clf = NearestCentroid(shrink_threshold = self.shrink)
                clf.fit(train, y_ind)
                centroids.append(clf.centroids_)

            elif center == "median":
                ch_median = self._class_median(train, y_ind)
                centroids.append(ch_median)

            elif center == "madc":
                ch_median = self._class_medianc(train, y_ind)
                centroids.append(ch_median)
            
            elif center == "mad":
                ch_mad = self._class_mad_median(train, y_ind)
                centroids.append(ch_mad)

            elif center == "madshrink":
                train = self._shrink_median(train, y_ind)
                clf = MC(shrink_threshold=self.shrink)
                clf.fit(train, y_ind)
                centroids.append(clf.centroids_)
                #print(centroids)
                #centroids.append(ch_mad)
                
            elif center == "std":
                ch_std = self._class_std(train, y_ind)
                centroids.append(ch_std)
            #break
            #std = self._class_std(train, y_ind)
            #stds.append(std)

        centroid_frame = from_3d_numpy_to_nested(np.stack(centroids, axis=1), column_names=cols)
        #centroid_frame = centroid_frame.applymap(self._filter)
        
        if mean_centering == True:
            print("Mean Centering done...")
            centroid_frame = centroid_frame.applymap(self.mean_centering)
        
        if _filter == True:
            print("Filter applied")
            centroid_frame = centroid_frame.applymap(self._filter)
        
        if _fft == True:
            print("FFT transformation done...")
            centroid_frame = centroid_frame.applymap(self._fft)

        centroid_frame['class_vals'] = le.classes_ 
        
        return centroid_frame.reset_index(drop =True)

if __name__ == "__main__":
    
    train = "./MP/FullUnnormalized25/TRAIN_default_X.ts"
    dataset = ['Cricket']
    for item in dataset:
        print(item)
        train_x, y = load_from_tsfile_to_dataframe(train, return_separate_X_and_y=True)
        obj = shrunk_centroid(0)
        df_s = obj.create_centroid(train_x.copy(), y, center="mad")
        break
