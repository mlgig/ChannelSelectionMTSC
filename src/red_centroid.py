"""
Not required
"""
import pandas as pd
import sys
import os
sys.path.insert(0, os.getcwd())
sys.path.append("..")
import numpy as np
from sktime.utils.data_io import load_from_tsfile_to_dataframe
from scripts.calculate_centroid import centroid
from scripts.utils import *
from operator import truediv 
from sktime.utils.data_processing import from_3d_numpy_to_nested

class standard_centroid(centroid):
    def __init__(self, X):
        self.centroid_class = super().calculate_centroid(X)
        self.centroid_dataset = X.drop(['class_vals'], 1).sum()/X.shape[0]


    def calculate_std(self, X, class_):
        """
        Calculate the standard deviation for class provide 
        """
        #class_centroid = self.centroid_class(X)
        # Keeping the centroid of relevant clas 
        #class_centroid = class_centroid[class_centroid['class_vals']==class_] 
        
        class_df = X.groupby('class_vals').get_group(class_)
        #print(class_df.shape)
        #print(class_centroid.shape)
        #print(class_df.shape)
        
        all_dim = []
        for ts in  class_df.drop(['class_vals'], axis=1).iterrows(): # Iterate through every row in dataframes(each row a MTS)
            dim = []
            for dim_ts in  ts[1].T.iteritems():
                dim.append(dim_ts[1].values)
            all_dim.append(dim)
        
        class_data = np.array(all_dim) # data in numpy format
        #print(f"class_data:", class_data.median(axis=0))
        mts_std = class_data.std(axis=0) # each row corresponds to respective dimensions

        #print(mts_std.shape)
       

        #ts_std = np.reshape(mts_std, (1, mts_std.shape[0], mts_std.shape[1]))
        
        #print(ts_std.shape)
        return mts_std

    def std(self, X): 
        """
        std = s_i + s_o
        """
        
        #all_stds = np.zeros(shape= (X.shape[0], X.shape[1]-1, X.iloc[0,0].shape[0])) # Shape = (number of TS, number of dim, len of TS)
        #all_stds = np.empty((X.shape[0], X.shape[1]-1, X.iloc[0,0].shape[0]))
        
        s_i = []
        labels = []
        for class_name, _ in X.groupby('class_vals'):
            labels.append(class_name)
            std = self.calculate_std(X, class_name)
            s_i.append(std)
        
        s_i = np.array(s_i)
        
        # s_0 median values among s_i
        s_0 = []        
        for i in range(s_i.shape[1]):
            s_0.append(np.median(s_i[:, i], axis=0))
        
        s_0 = np.array(s_0)
        
        st_dev= [] # s_i + s_o
        for item in s_i:
            st_dev.append(np.add(item, s_0))

        print(np.array(st_dev).shape)
        s = from_3d_numpy_to_nested(np.array(st_dev))
        s.columns = X.columns.to_list()[:-1]
        s['class_vals'] = labels
        return  s # s from the paper

    def reduced_centroid(self, X):
        """
        Numerator part to shrinkage
        class_centroid - dataset_centroid
        """
        
        red_df = pd.DataFrame(columns=X.columns) # Reduced centroid
        labels=[]

        class_ = self.centroid_class # Centroid of each class
        dataset_ = self.centroid_dataset # Centroid of full dataset

        for (idx, item), class_name in zip(class_.drop(['class_vals'], 1).iterrows(), class_.class_vals):
            labels.append(class_name)
            red_df = red_df.append(item-dataset_, ignore_index = True) # Here centroid is getting reduced

        red_df.class_vals = labels

        #print(red_df.head())
        
        return red_df

    def normalised_dim(self, dis, std, class_, cols):
        #print((std.shape))
        
        df = pd.DataFrame()
        ts = []

        for std_dim, dist in zip(std.drop(['class_vals'], 1).iteritems(), dis.drop(['class_vals'], 1).iteritems()):
            seq = dist[1].values[0].values # Yuck too much nested
            dim = std_dim[1].values[0].values
            res = pd.Series(list(map(truediv, seq, dim))) # std_dim is s_i
            #print(std_dim.shape)  
            ts.append(res)
            
        df = pd.DataFrame(pd.Series(ts)).transpose() # too complicate phew
        
        df.columns = cols
        df['class_vals'] = class_
        return df

    def shrinkage(self, X):
        
        numerator = self.reduced_centroid(X) # get the reduced centroid
        labels =  set(X.class_vals)
        denominator = self.std(X) 

        std_df = pd.DataFrame(columns=X.columns)
        cols = X.columns.to_list()
        cols.remove('class_vals')

        st_df = pd.DataFrame() # standard centroid dataframe
        for item in labels:
            #print(item)
            std = denominator[denominator['class_vals']==item] # Calculate the standard dev of each class
            distance = numerator[numerator['class_vals']==item] # numerator part of the formula
            df = self.normalised_dim(distance, std, item, cols) # Divide the numerator wit Standard Dev
            
            st_df = pd.concat([st_df, df]) # Collect all the results
        #print(st_df.head())

        print("Shrinkage Calculated")
        return st_df.reset_index(drop=True)

    def adjust_shrinkage(self,X):
        shrink = self.shrinkage(X)
        pass

    def shrink_centroid(self, X, class_):
        """
        Equation 3 here
        """
        m_k = np.sqrt(X.shape[0] + X[X['class_vals']==class_].shape[0])
        print(m_k)
        
        shrinkage = self.shrinkage(X)
        shrink_class_centroid = self.class_centroid + m_k



if __name__ == "__main__":

    #jump_dataset = ['FullUnnormalized', 'Normalized', 'Unnormalized']
    #dataset = ['DuckDuckGeese', 'FaceDetection', 'MotorImagery', 'PEMS-SF']
    dataset = ['Cricket']
    for item in dataset:
        print(item)


        #NOTE: Code to read Jump Dataset
        #train_x = load_from_tsfile_to_dataframe(f"/home/bhaskar/Desktop/CentroidMTSC/MP/{item}/TRAIN_X.ts",return_separate_X_and_y=False)
        train_x = load_from_tsfile_to_dataframe(f"./data/{item}/{item}_TRAIN.ts",return_separate_X_and_y=False)
        print(train_x.shape)

        obj= standard_centroid(train_x)
        #print(train_x.shape)
        #df = obj.reduced_centroid(train_x)
        #res = obj.calculate_std(train_x,'1.0')
        #df = obj.standard_cent(train_x)
        r = obj.shrink_centroid(train_x, '1.0')
        #print(r.shape)
        #s = obj.shrinkage(train_x)
        break
