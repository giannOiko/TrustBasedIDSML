import KitNET as kit
import numpy as np
import pandas as pd
import time
import csv
from sklearn.metrics import f1_score,precision_score,recall_score
import subprocess


df = pd.read_csv('mirai3.csv', header=None)
labels = [1] * 71000 + [-1] * 29000
df['label'] = labels

X = df.drop('label', axis = 1)
X_training = X.iloc[:35500]
X_final = pd.concat([X,X_training], axis = 0)
y = df['label']

print("X_final length is ",len(X_final))
# KitNET params:
maxAE = 10 #maximum size for any autoencoder in the ensemble layer
FMgrace = 5000 #the number of instances taken to learn the feature mapping (the ensemble's architecture)
ADgrace = 30500 #the number of instances used to train the anomaly detector (ensemble itself)

# Build KitNET
K = kit.KitNET(X_final.shape[1],maxAE,FMgrace,ADgrace)
RMSEs = np.zeros(X_final.shape[0]) # a place to save the scores
print("Running KitNET:")

start = time.time()


# Here we process (train/execute) each individual observation.
# In this way, X is essentially a stream, and each observation is discarded after performing process() method.
for i in range(FMgrace + ADgrace):
    if i % 1000 == 0:
        print(i)
    RMSEs[i] = K.process(X_final.iloc[i]) #will train during the grace periods, then execute on all the rest.
stop = time.time()
print("Training Done: "+ str(stop - start))


process = subprocess.Popen(['./monitor.sh'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
start = time.time()
# Here we process (train/execute) each individual observation.
# In this way, X is essentially a stream, and each observation is discarded after performing process() method.
predict_samples_num = X_final.shape[0] - FMgrace -ADgrace
print("predict samples num: ",predict_samples_num)


for i in range(predict_samples_num):
    if i % 1000 == 0:
        print(i)
    RMSEs[i] = K.process(X_final.iloc[i]) #will train during the grace periods, then execute on all the rest.


process.kill()
stop = time.time()
print("Predicting Time: "+ str(stop - start))