import sys 
import os
sys.path.insert(0, os.getcwd())
sys.path.append(".")
sys.path.append("..")

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

from sklearn.pipeline import Pipeline
from sktime.transformations.panel.rocket import Rocket
from sklearn.linear_model import RidgeClassifierCV
from sktime.utils.data_io import load_from_tsfile_to_dataframe

from sktime.classification.shapelet_based import MrSEQLClassifier
from sktime.classification.dictionary_based import MUSE



from src.kmeans import kmeans
from multiprocessing import Process
from  src.classelbow import ElbowPair # ECP
from src.elbow import elbow # ECS..


# In[4]:


import seaborn as sns
import random
import numpy as np
import matplotlib.pyplot as plt

from pylab import rcParams
rcParams['figure.figsize'] = 25, 10


# In[5]:


def visualise(train, n_samples=15):
    train_x, train_y = load_from_tsfile_to_dataframe(train)
    
    samples = np.random.randint(low = 0, high = train_x.shape[1], size = n_samples)
    
    if train_x.shape[1] == 8:
        d = {0: "Right Elbow", 1: "Left Elbow", 2: "Left Hip", 3: "Left Shoulder",
        4 : "Left Wrist", 5: "Right Hip", 6: "Right Shoulder", 7: "Right Wrist"}
    elif train_x.shape[1] == 14:
        d = {0 : "RElbow", 1 : "LAnkle", 2 : "LEar", 3 : "LElbow", 4: "LHip", 5 : "LKnee", 6 : "LShoulder",
             7 : "LWrist", 8 : "RAnkle", 9 : "REar", 10 : "RHip", 11:  "RKnee", 12: "RShoulder", 13 : "RWrist"}

    elif train_x.shape[1] == 25:
        d = {0: 'Nose',1: 'Neck',2: 'RShoulder',3: 'RElbow',4: 'RWrist',5: 'LShoulder',6: 'LElbow',7: 'LWrist',
      8: 'MidHip',9: 'RHip',10: 'RKnee',11: 'RAnkle',12: 'LHip',13: 'LKnee',14: 'LAnkle', 
      15:'REye',16: 'LEye',17: 'REar',18: 'LEar',19: 'LBigToe',20: 'LSmallToe',21: 'LHeel',22: 'RBigToe',
      23: 'RSmallToe',24: 'RHeel'}
    
    
    for channel in range(train_x.shape[1]): 
        print(d[channel])
        for rows in samples:
            p = train_x.iloc[rows, channel]
            plt.plot(range(0, len(p.values)), p.values)
        plt.show()


# In[6]:


def cs(strategy):
    
    if strategy == "km":
        elb = kmeans()
    elif strategy == "ecs":
        elb  = elbow()
    elif strategy == "ecp":
        elb = ElbowPair(distance = 'eu')
    elif strategy == 'all':
        elb = None
        
    return elb


# In[7]:


def dict_map(train_x):
    if train_x.shape[1] == 8:
        d = {0: "RElbow", 1: "LElbow", 2: "LHip", 3: "LShoulder",
           4 : "LWrist", 5: "RHip", 6: "RShoulder", 7: "RWrist"}
        
    elif train_x.shape[1] == 14:
        d = {0 : "RElbow", 1 : "LAnkle", 2 : "LEar", 3 : "LElbow", 4: "LHip", 5 : "LKnee", 6 : "LShoulder",
             7 : "LWrist", 8 : "RAnkle", 9 : "REar", 10 : "RHip", 11:  "RKnee", 12: "RShoulder", 13 : "RWrist"}

    elif train_x.shape[1] == 25:
        d = {0: 'Nose',1: 'Neck',2: 'RShoulder',3: 'RElbow',4: 'RWrist',5: 'LShoulder',6: 'LElbow',7: 'LWrist',
          8: 'MidHip',9: 'RHip',10: 'RKnee',11: 'RAnkle',12: 'LHip',13: 'LKnee',14: 'LAnkle', 
          15:'REye',16: 'LEye',17: 'REar',18: 'LEar',19: 'LBigToe',20: 'LSmallToe',21: 'LHeel',22: 'RBigToe',
          23: 'RSmallToe',24: 'RHeel'}
        
    return d

from itertools import chain, combinations
import pandas as pd
def function(train, test, strategy="all"):
    train_x, train_y = load_from_tsfile_to_dataframe(train)
    test_x, test_y = load_from_tsfile_to_dataframe(test)

    print("Train shape: ", train_x.shape)
    
    #elb = cs(strategy)
    d = dict_map(train_x)


    
    s = np.arange(train_x.shape[1])
    dims_subset = list(chain.from_iterable(combinations(s, r) for r in range(len(s)+1)))
    results = pd.DataFrame()
    
    for item in dims_subset:
        
        model = Pipeline(
            [
            #('classelbow', elb),
            ('rocket', Rocket(random_state=0,normalise=False)),
            ('model', RidgeClassifierCV(alphas=np.logspace(-3, 3, 10),normalize=True ))
            #('SEQL', MrSEQLClassifier()),
            #('weasel_muse', MUSE(random_state=0)),
            ],
        )
        if len(list(item))>=8:
            #print(d)
            print([d.get(i) for i in item])
            model.fit(train_x.iloc[:,list(item)], train_y)
            preds = model.predict(test_x.iloc[:,list(item)])
            tmp = accuracy_score(preds, test_y) * 100
            acc1 = tmp    
            _acc = pd.DataFrame({
                f'Accuracy': [tmp],
                #'dimension':str(item),
            })
            results = pd.concat([results, _acc], axis=1)
            
            print(tmp)
            print("--"*50)

            del model
            
    print(results)
        


# In[50]:


strategy = "ecp"


train = "./MP/Unnormalized/TRAIN_X.ts"
test =  "./MP/Unnormalized/TEST_X.ts"

#function(train, test, strategy)
#visualise(train, 50)

train = "./MP/FullUnnormalized14/TRAIN_X.ts"
test =  "./MP/FullUnnormalized14/TEST_X.ts"

function(train, test, strategy)
#visualise(train)train = "/home/bhaskar/Desktop/ChannelSelection-Extend/MP/FullUnnormalized25/TRAIN_default_X.ts"
#test =  "/home/bhaskar/Desktop/ChannelSelection-Extend/MP/FullUnnormalized25/TEST_default_X.ts"

#function(train, test, strategy)
"""
#visualise(train)['Right Elbow', 'Left Elbow', 'Left Hip', 'Left Shoulder', 'Left Wrist', 'Right Shoulder', 'Right Wrist']
[3, 6, 12, 5, 2, 4]
# In[9]:


d_8 = {0: "RElbow", 1: "LElbow", 2: "LHip", 3: "LShoulder",
       4 : "LWrist", 5: "RHip", 6: "RShoulder", 7: "RWrist"}


# In[10]:


d_25 = {0: 'Nose',1: 'Neck',2: 'RShoulder',3: 'RElbow',4: 'RWrist',5: 'LShoulder',6: 'LElbow',7: 'LWrist',
      8: 'MidHip',9: 'RHip',10: 'RKnee',11: 'RAnkle',12: 'LHip',13: 'LKnee',14: 'LAnkle', 
      15:'REye',16: 'LEye',17: 'REar',18: 'LEar',19: 'LBigToe',20: 'LSmallToe',21: 'LHeel',22: 'RBigToe',
      23: 'RSmallToe',24: 'RHeel'}


# In[11]:


inv_d25 = {v: k for k, v in d_25.items()}


# In[12]:


d_14 = {0 : "RElbow", 1 : "LAnkle", 2 : "LEar", 3 : "LElbow", 4: "LHip", 5 : "LKnee", 6 : "LShoulder",
             7 : "LWrist", 8 : "RAnkle", 9 : "REar", 10 : "RHip", 11:  "RKnee", 12: "RShoulder", 13 : "RWrist"}


# In[16]:


find = ['RElbow', 'LElbow', 'LHip', 'LShoulder', 'LWrist', 'RShoulder', 'RWrist']


# In[18]:


ch = []
for item in find:#d_14.values():
    ch.append(inv_d25[item])


# In[19]:


ch


# In[20]:


train = "/home/bhaskar/Desktop/ChannelSelection-Extend/MP/FullUnnormalized25/TRAIN_default_X.ts"
test =  "/home/bhaskar/Desktop/ChannelSelection-Extend/MP/FullUnnormalized25/TEST_default_X.ts"

train_x, train_y = load_from_tsfile_to_dataframe(train)
test_x, test_y = load_from_tsfile_to_dataframe(test)

print(train_x.shape)
print(test_x.shape)

model = Pipeline(
            [
            #('classelbow', elb),
            ('rocket', Rocket(random_state=0,normalise=False)),
            ('model', RidgeClassifierCV(alphas=np.logspace(-3, 3, 10),normalize=True ))
            ],
            verbose=True,
        )


train_x = train_x.iloc[:, ch]
test_x = test_x.iloc[:, ch]

print("After CS")

print(train_x.shape)
print(test_x.shape)

model.fit(train_x, train_y)

preds = model.predict(test_x)
acc1 = accuracy_score(preds, test_y) * 100
print(f"Accuracy: {acc1}")



# In[ ]:





# In[ ]:


"""
print("END")
