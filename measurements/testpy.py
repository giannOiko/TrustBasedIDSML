from lightgbm import LGBMClassifier
import json
import pandas as pd
from sklearn.model_selection import train_test_split
import time
import os
import subprocess


#########Read Prepare Dataset###################
df = pd.read_csv('mirai3.csv', header=None)
labels = [1] * 71000 + [-1] * 29000
df['label'] = labels

X = df.drop('label' , axis = 1)
y = df['label']

########model train test split##################
model = LGBMClassifier(objective='binary')


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42,stratify = y)

########train##########
model.fit(X_train, y_train)


# here measure mem and cpu usage by activating script ./monitor.sh.
# and time with time library   
monitor_process = subprocess.Popen(['./monitor.sh'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
start_time = time.time()

# Make predictions
preds = model.predict(X)

end_time = time.time()
time = end_time-start_time

monitor_process.kill()  # Forcefully kill the process

# append the total time in logfile.
with open('ps.log', 'a') as logfile:

    logfile.write(f"Python Execution time runs: {time} seconds") 


# pass the same model to test in goLang
""" booster = model.booster_
model_dump = booster.dump_model()

# Save the model in JSON format
with open('lightgbm_model.json', 'w') as f:
    json.dump(model_dump, f) """


