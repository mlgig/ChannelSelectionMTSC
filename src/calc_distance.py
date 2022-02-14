# Calcute distance matrix
import itertools
import pandas as pd
from .utils import *
import numpy as np


class distance_matrix:
    """
    Creates the distance matrix
    """

    def __init__(self, distance):
        self.distance_=str(distance)
        #pass

    def distance(self, centroid_frame):
        """
        centroid_frame = classes * channels 
        """
        
        distance_pair = list(itertools.combinations(range(0, centroid_frame.shape[0]),2))
         
        idx_class_map = centroid_frame.class_vals.to_dict()
        distance_frame = pd.DataFrame()
        for class_ in distance_pair:
    
            class_pair = []
            # calculate the distance of centroid here
            for _, (q, t) in enumerate(zip(centroid_frame.drop(['class_vals'],axis=1).iloc[class_[0],:], centroid_frame.iloc[class_[1],:])):
                #print(eu_dist(q.values, t.values))
                if self.distance_== "eu":
                    class_pair.append(eu(q.values, t.values))
                elif self.distance_ == "dtw":
                    class_pair.append(dtw(q.values, t.values))
                elif self.distance_ == "conv":
                    #print("conv")
                    class_pair.append(np.correlate(q.values, t.values, "valid")[0])
                dict_= {f"Centroid_{idx_class_map[class_[0]]}_{idx_class_map[class_[1]]}": class_pair}
                
                #print(class_[0])

            distance_frame = pd.concat([distance_frame, pd.DataFrame(dict_)], axis=1)

        #top_dims, break_idx = self.detect_knee_point(distance_frame.sum(axis=1).sort_values(ascending=False).values, distance_frame.sum(axis=1).sort_values(ascending=False).index)

        return distance_frame


