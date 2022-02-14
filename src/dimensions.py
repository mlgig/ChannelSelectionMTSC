import pandas as pd
import sys
import os
sys.path.insert(0, os.getcwd())

sys.path.append("..")
import numpy as np
from sktime.utils.data_io import load_from_tsfile_to_dataframe
import matplotlib.pyplot as plt
from scripts.red_centroid import standard_centroid
from scripts.utils import eu_dist, cosine_dist
import itertools

class dimension(standard_centroid):

    def detect_knee_point(self, values, indices):
        """
        From:
        https://stackoverflow.com/questions/2018178/finding-the-best-trade-off-point-on-a-curve
        """
        # get coordinates of all the points
        #print(values)
        #print(indices)
        n_points = len(values)
        #print(n_points)
        all_coords = np.vstack((range(n_points), values)).T
        # get the first point
        first_point = all_coords[0]
        # get vector between first and last point - this is the line
        line_vec = all_coords[-1] - all_coords[0]
        line_vec_norm = line_vec / np.sqrt(np.sum(line_vec ** 2))
        vec_from_first = all_coords - first_point
        scalar_prod = np.sum(
            vec_from_first * np.tile(line_vec_norm, (n_points, 1)), axis=1)
        vec_from_first_parallel = np.outer(scalar_prod, line_vec_norm)
        vec_to_line = vec_from_first - vec_from_first_parallel
        # distance to line is the norm of vec_to_line
        dist_to_line = np.sqrt(np.sum(vec_to_line ** 2, axis=1))
        # knee/elbow is the point with max distance value
        best_idx = np.argmax(dist_to_line)
        knee  = values[best_idx]

        print(f"Knee: {values[best_idx]}")
        
        #print([(elem, idx) for (elem, idx) in zip(values, indices)])
        #print([(elem, idx) for (elem, idx) in zip(values, indices)] [-2:] [0] [1], [(elem, idx) for (elem, idx) in zip(values, indices)] [-2:] [1] [1])
        
        e = [idx for (elem, idx) in zip(values, indices) if elem>=knee]
        worst_idx = [(elem, idx) for (elem, idx) in zip(values, indices)] [-2:] [0] [1], [(elem, idx) for (elem, idx) in zip(values, indices)] [-2:] [1] [1]     
        return e, best_idx, worst_idx


    def find_dims(self, X):

        #get centroid of each class    
        centroid_frame = self.standard_cent(X).reset_index(drop=True)

        distance_pair = list(itertools.combinations(range(0, centroid_frame.shape[0]),2))
        #exit()
         
        idx_class_map = centroid_frame.class_vals.to_dict()
        #print(idx_class_map)
        distance_frame = pd.DataFrame()
        for class_ in distance_pair:
    
            class_pair = []
            # calculate the distance of centroid here
            for _, (q, t) in enumerate(zip(centroid_frame.drop(['class_vals'],axis=1).iloc[class_[0],:], centroid_frame.iloc[class_[1],:])):
                #print(eu_dist(q.values, t.values))
                class_pair.append(eu_dist(q.values, t.values)) 
                dict_= {f"Centroid_{idx_class_map[class_[0]]}_{idx_class_map[class_[1]]}": class_pair}
                #print(class_[0])

            distance_frame = pd.concat([distance_frame, pd.DataFrame(dict_)], axis=1)
        #print(distance_frame.head())
        #exit()

        #TODO: Add automatic function
        #NOTE find function here

        top_dims, break_idx, worst_idx = self.detect_knee_point(distance_frame.sum(axis=1).sort_values(ascending=False).values, distance_frame.sum(axis=1).sort_values(ascending=False).index)

        return top_dims, distance_frame, break_idx, list(worst_idx)   

if __name__ == "__main__":

    dataset = ['FullUnnormalized', 'Normalized', 'Unnormalized']
    #dataset = ['DuckDuckGeese', 'FaceDetection', 'MotorImagery', 'PEMS-SF']
    #dataset = ['Cricket']
    for item in dataset:
        print(item)


        #NOTE: Code to read Jump Dataset
        train_x = load_from_tsfile_to_dataframe(f"./MP/{item}/TRAIN_X.ts",return_separate_X_and_y=False)
        #train_x = load_from_tsfile_to_dataframe(f"./data/{item}/{item}_TRAIN.ts",return_separate_X_and_y=False)
        obj=dimension()
        new_dims,distance_frame,break_idx,_ = obj.find_dims(train_x)
        #print(list(df))
        #print(set(dis_frame.index)-set(df))

        distance_frame.to_csv(f'./distances/{item}_ED_STD.csv')
        
        x = distance_frame.sum(axis=1).sort_values(ascending = False).index # Print the index of dimension on x axis
        y = distance_frame.sum(axis=1).sort_values(ascending = False).values   # Print the distance on the Y-axis

        #print("X: ", x)
        #print("Y: ", y)

        #ax= sns.lineplot(x= range(len(y)), y = y, marker='o')
        plt.figure(figsize=(10,8))
        plt.plot(y)
        plt.xticks(ticks=range(len(y)), labels=x)
        plt.axvline(break_idx, 0,1, c = 'red')
        #ax.set(xticklabels= x, xlabel='Dimensions', ylabel='Distance from centroid')
        plt.title(f'{item}')
        #plt.show()
        plt.savefig(f"./Images/{item}_CS_std_UCR.png")
        plt.clf()
        #break

   