import click
import os
import sys
import numpy as np
from scipy.signal.filter_design import normalize

sys.path.insert(0, os.getcwd())
import pandas as pd
from multiprocessing import Process
from sktime.utils.data_io import load_from_tsfile_to_dataframe
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import RidgeClassifierCV
import time

from sktime.transformations.panel.rocket import Rocket

from src.dataset import dataset_

from src.kmeans import kmeans
from multiprocessing import Process
from  src.classelbow import ElbowPair # ECP
from src.elbow import elbow # ECS..


def agent(path, dataset, folder, datatype, distancefn, strategy, center, mc, fft):
    #print(dataset)

    if datatype=='UCR':
        #print(f"{path}/{dataset}/{dataset}_TRAIN.ts")
        norm=True
        train_x, train_y = load_from_tsfile_to_dataframe(f"{path}/{dataset}/{dataset}_TRAIN.ts")#, return_separate_X_and_y=False)
        test_x, test_y = load_from_tsfile_to_dataframe(f"{path}/{dataset}/{dataset}_TEST.ts")#, return_separate_X_and_y=False)
    elif datatype=='MP':
        print("MP")
        norm=False
        train_x, train_y = load_from_tsfile_to_dataframe(f"{path}/FullUnnormalized25XY/TRAIN_default_X.ts")
        test_x, test_y = load_from_tsfile_to_dataframe(f"{path}/FullUnnormalized25XY/TEST_default_X.ts")#, return_separate_X_and_y=False)


    if strategy == "km":
        elb = kmeans(center=center) 
    elif strategy == "ecs":
        elb  = elbow(center=center,  mc=mc, fft=fft)
    elif strategy == "ecp":
        elb = ElbowPair(distance = distancefn, center = center, mc=mc, fft=fft)
    elif strategy == 'all':
        elb = None

    start = time.time()

    model = Pipeline(
            [
                ('classelbow', elb),
                ('rocket', Rocket(random_state=0, normalise=norm)),
                ('model', RidgeClassifierCV(alphas=np.logspace(-3, 3, 10),normalize=True, class_weight='balanced'))
            ],
            #verbose=True,
            )
    model.fit(train_x, train_y)
    end = time.time()
    
    preds = model.predict(test_x)
    acc1 = accuracy_score(preds, test_y) * 100

    

    if strategy in ["km", "ecp", "ecs"]:
         results = pd.DataFrame({
            'Dataset': dataset,
             'Accuracy': [acc1], 
            'Time(min)': [(end - start)/60],
            'dimension_used':[len(elb.relevant_dims)/train_x.shape[1]]
        })
        
    else:
        results = pd.DataFrame({
            'Dataset': dataset, 
        'Accuracy': [acc1], 
        'Time(min)': [(end - start)/60]
        })
    
    print(results)
    temp_path = './'+folder
    if not os.path.exists(temp_path):
        os.mkdir(temp_path)
    results.to_csv(os.path.join(temp_path + f'/{dataset}.csv'), index=False)
    os.system("rm -r $VIRTUAL_ENV/lib/python3.6/site-packages/sktime/transformations/panel/rocket/__pycache__/*")


@click.command()
@click.option('--datadir', help="Path of datasets", required=True, type=click.Path(exists=True))
@click.option('--center', help="Mean or Median centroid", type=click.Choice(['mean', 'median', 'mad'], case_sensitive=True),default="mean")
@click.option('--tempres', help="Folder to store result", required=True)
@click.option('--datatype', help="UCR data or MP data",  type=click.Choice(['UCR', 'MP'], case_sensitive=True),default="UCR")
@click.option('--distancefn', help="Function to calculate distance between centroid pair",  type=click.Choice(['eu', 'dtw','conv'], case_sensitive=True),default="eu")
@click.option('--strategy', help="KMeans, ECS or ECP, def:ECP",  type=click.Choice(['all','km', 'ecs', 'ecp'], case_sensitive=True),default="ecp")
@click.option('--mc', help="Mean Centering",  type=click.Choice(['True', 'False'], case_sensitive=True),default="False")
@click.option('--fft', help="FFT",  type=click.Choice(['True', 'False'], case_sensitive=True), default="False")
def cli(datadir, center, tempres, datatype, distancefn, strategy, mc, fft):

    
    #center = "median"
    dataset_name = dataset_
    #dataset_name = ['Epilepsy']
    mc= eval(mc)
    fft = eval(fft)
    processes = []
    if datatype=="UCR":
        
        for data in dataset_name:
            agent(datadir, data, tempres, datatype, distancefn, strategy, center, mc, fft)
    else:
        agent(datadir, datatype, tempres, datatype, distancefn, strategy, center, mc, fft)



if __name__ == '__main__':
    cli()
