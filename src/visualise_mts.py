import plotly.express as px
import sys
import os
sys.path.insert(0, os.getcwd())
sys.path.append("..")
import numpy as np
from sktime.utils.data_io import load_from_tsfile_to_dataframe
from dataset import dataset
from calculate_centroid import centroid
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from dataset import dataset
from color_dict import color_dict

def visualise_dataframe(df, label, title, plot_head=None):
    fig = make_subplots(rows=df.shape[1]-1, cols=1, subplot_titles=df.drop(label, axis=1).columns)
    #print(df)
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


    jump_dataset = ['FullUnnormalized', 'Normalized', 'Unnormalized']
    dataset = ['FullUnnormalized']
    for item in dataset:
        print(item)


        #NOTE: Code to read Jump Dataset
        train_x = load_from_tsfile_to_dataframe(f"../CentroidMTSC/MP/{item}/TRAIN_X.ts",return_separate_X_and_y=False)
        #print(train_x.shape)
        #train_x = load_from_tsfile_to_dataframe(f"./data/{item}/{item}_TRAIN.ts",return_separate_X_and_y=False)
        
        sample_size = len(set(train_x['class_vals']))
        
        print(f"Number of classes: {sample_size}")
        fn = lambda obj: obj.loc[np.random.choice(obj.index, 1),:]
        plot_df = train_x.groupby('class_vals', as_index=False).apply(fn)
        print(plot_df.index)
        plot_df.reset_index(drop=True, inplace=True)
        
        print(plot_df.shape)
        
        if train_x.shape[1]==15:
            plot_head=full
        elif train_x.shape[1]==9: plot_head = small
        else: plot_head=None
        
        fig = visualise_dataframe(plot_df, label='class_vals', title=item, plot_head= plot_head)
        fig.show()
        fig.write_html(f"./notebooks/MP_images/{item}_data_point.html")
        break

