import plotly.express as px
import sys
import os
sys.path.insert(0, os.getcwd())
sys.path.append("..")
import numpy
from sktime.utils.data_io import load_from_tsfile_to_dataframe
from dataset import dataset
from calculate_centroid import centroid
from  scripts.calculate_centroid import centroid
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from dataset import dataset
from color_dict import color_dict
from scripts.shrunk_cent import shrunk_centroid
import pandas as pd

def visualise_dataframe(df, label, title, plot_head=None):
    fig = make_subplots(rows=df.shape[1]-1, cols=1, subplot_titles=df.drop(label, axis=1).columns)
    #print((df[label]))
    flag = 0
    for i, dims in enumerate(df.drop(label, axis=1)):
        for j, data in enumerate(df.drop(label, axis=1).loc[:, dims]):
            #print(j)
            fig.add_trace(
               go.Scatter(y=data.values, name = df[label][j], marker=dict(color=color_dict[j])), 
            row=i+1, col=1
            )
            if plot_head:
                fig.update_xaxes(title_text=plot_head[i], row=i+1, col=1)

    fig.update_layout(showlegend=False, height=4000, width=2000, title_text = title)
    return fig

if __name__ == "__main__":


    full = {0 : 'RElbow', 1 : 'LAnkle', 2 : 'LEar', 3: 'LElbow', 4 : 'LHip', 5 : 'LKnee', 6 : 'LShoulder', 7 : 'LWrist', 8 : 'RAnkle', 9 : 'REar', 
    10 : 'RHip', 11:  'RKnee', 12 : 'RShoulder', 13 : 'RWrist'}

    small = {0 : 'Right Elbow', 1 : 'Left Elbow', 2 : 'Left Hip', 3 : 'Left Shoulder', 4 : 'Left Wrist', 5 : 'Right Hip', 6 : 'Right Shoulder', 7 : 'Right Wrist'}


    dataset = ['FullUnnormalized', 'Unnormalized']
    #dataset = ['Unnormalized']
    for item in dataset:
        print(item)


        #NOTE: Code to read Jump Dataset
        train_x = load_from_tsfile_to_dataframe(f"../CentroidMTSC/MP/{item}/TRAIN_X.ts",return_separate_X_and_y=False)
        print(train_x.shape)
        #train_x = load_from_tsfile_to_dataframe(f"./data/{item}/{item}_TRAIN.ts",return_separate_X_and_y=False)
        
        #When using plain centroid
        obj=centroid()
        centroid_df = obj.calculate_centroid(train_x)
        
        #When using Shrunken centroid
        ##obj = shrunk_centroid(0.1)
        ##centroid_df= obj.create_centroid(train_x)
        #print(centroid_df.head())
 
        #print()
        
        if train_x.shape[1]==15:
            plot_head=full
        elif train_x.shape[1]==9: plot_head = small
        else: plot_head=None
        
        fig = visualise_dataframe(centroid_df, label='class_vals', title=item, plot_head= plot_head)
        fig.show()
        fig.write_html(f"./notebooks/MP_images/{item}_Plane.html")
        #df, dis_frame, break_idx = obj.find_dims(train_x)
        #print(df)

        #dis_frame.to_csv(f'./distances/{item}_ED.csv')
        
        #x = dis_frame.sum(axis=1).sort_values(ascending = False).index # Print the index of dimension on x axis
        #y = dis_frame.sum(axis=1).sort_values(ascending = False).values   # Print the distance on the Y-axis

        #print("X: ", x)
        #print("Y: ", y)

        #ax= sns.lineplot(x= range(len(y)), y = y, marker='o')
        #plt.plot(y)
        #plt.xticks(ticks=range(len(y)), labels=x)
        #plt.axvline(break_idx, 0,1, c = 'red')
        #ax.set(xticklabels= x, xlabel='Dimensions', ylabel='Distance from centroid')
        #plt.title(f'{item}')
        #plt.show()
        
        #plt.savefig(f"./notebooks/MP_images/{item}_ED.png")
        #plt.clf()
        #break

