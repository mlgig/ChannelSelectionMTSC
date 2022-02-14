import click
import os
import sys
import numpy as np

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.getcwd())
sys.path.append(".")
sys.path.append("..")
import pandas as pd
from multiprocessing import Process, current_process
from sktime.utils.data_io import load_from_tsfile_to_dataframe
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sktime.classification.shapelet_based import MrSEQLClassifier
from sktime.transformations.panel.rocket import Rocket
from sklearn.linear_model import RidgeClassifierCV

import time

pd.set_option('display.max_columns', None)  

from sklearn.metrics import f1_score


from itertools import chain, combinations

def agent(path, dataset, folder,  paa=True):
    current_process().name = dataset
    
    print(current_process().name)

    if dataset in dataset: #['DuckDuckGeese', 'FaceDetection', 'MotorImagery', 'PEMS-SF']:
        train_x, train_y = load_from_tsfile_to_dataframe(f"{path}/{dataset}/{dataset}_TRAIN.ts")#, return_separate_X_and_y=False)
        test_x, test_y = load_from_tsfile_to_dataframe(f"{path}/{dataset}/{dataset}_TEST.ts")#, return_separate_X_and_y=False)
    
    elif dataset in ['FullUnnormalized', 'Normalized', 'Unnormalized']:
        train_x, train_y = load_from_tsfile_to_dataframe(f"./MP/{dataset}/TRAIN_X.ts")
        test_x, test_y = load_from_tsfile_to_dataframe(f"./MP/{dataset}/TEST_X.ts")#, return_separate_X_and_y=False)
    print(f"{dataset}:Before Train Shape {train_x.shape}")
#    #print(f"{dataset}:Before Test Shape {test_x.shape}")
#
##Create the subsets here of the dimensions
    #print(np.arange(train_x.shape[1]))
    start = time.time()
    s = np.arange(train_x.shape[1])
    dims_subset = list(chain.from_iterable(combinations(s, r) for r in range(len(s)+1)))

    results = pd.DataFrame({'Dataset': [dataset]})
    acc1=0
    for item in dims_subset:


        model = Pipeline(
        [
        #('MrSEQL', MrSEQLClassifier()),
        ('rocket', Rocket(random_state=0,normalise=False)),
        ('model', RidgeClassifierCV(alphas=np.logspace(-3, 3, 10),normalize=True ))
        ],
        #verbose=True,

        )
        if len(list(item))>=1:
 
            model.fit(train_x.iloc[:,list(item)], train_y)
            preds = model.predict(test_x.iloc[:,list(item)])
            tmp = accuracy_score(preds, test_y) * 100
            #print(tmp)
            #if acc1<tmp:
            #print(f"ACC:{tmp}, ITEM: {item}")
            acc1 = tmp    
            _acc = pd.DataFrame({
                f'Accuracy_{item}': [tmp],
                #'dimension':str(item),
            })
            results = pd.concat([results, _acc], axis=1)

            del model
    
    end = time.time()
    #results['dataset'] = dataset
    results['Time(min)']= [(end - start)/60]
    print(results)
 
    temp_path = './'+folder
    if not os.path.exists(temp_path):
        os.mkdir(temp_path)
    results.to_csv(os.path.join(temp_path + f'/{dataset}.csv'), index=False)


@click.command()
@click.option('--path', help="Path of datasets", required=True, type=click.Path(exists=True))
@click.option('--paa', help="PAA", type=click.Choice(['True', 'False'], case_sensitive=True))
@click.option('--folder', help="Folder to store result", required=True)
def cli(path, paa, folder):

    #dataset_name = ['FullUnnormalized', 'Normalized', 'Unnormalized']
    #dataset_name = ['DuckDuckGeese', 'FaceDetection', 'MotorImagery', 'PEMS-SF','FullUnnormalized', 'Normalized', 'Unnormalized']
    #dataset_name = ['Unnormalized']
    dataset_name = ["Epilepsy","EthanolConcentration"]#, "Handwriting", "UWaveGestureLibrary", "AtrialFibrillation", "Libras", "PenDigits"]
    processes = []
    #dataset_name = ['ERing']
    for data in dataset_name:
        #print("\n",data)
        #proc = Process(target=agent, args=(path, data, folder, paa))
        agent(path, data, folder, paa)
        #processes.append(proc)
        #proc.start()
    
    #for p in processes:
    #    p.join()

        

if __name__ == '__main__':
    #path = '../mtsc/data/'
    #paa = True
    #folder = 'centroid_50'
    #seg = "0.30 0.6 0.9"
    #cli(path, paa, folder, seg)
    cli()
    

#python3 -W ignore main.py -- -- --folder centroid_50 --
