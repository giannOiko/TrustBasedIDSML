from lightgbm import LGBMClassifier
import json
import pandas as pd
from sklearn.model_selection import train_test_split
import time
import os
from memory_profiler import memory_usage
import tracemalloc



df = pd.read_csv('mirai3.csv', header=None)
labels = [1] * 71000 + [-1] * 29000
df['label'] = labels

X = df.drop('label' , axis = 1)
y = df['label']

model = LGBMClassifier(objective='binary')


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42,stratify = y)

#test time
model.fit(X_train, y_train)

start_time = time.time()

# Make predictions
preds = model.predict(X)

end_time = time.time()

# Print execution time
print(f"Execution time: {end_time - start_time} seconds")

#test memory
mem_usage = memory_usage((model.predict, (X,)))

print('Memory usage (in chunks of .1 seconds): %s' % mem_usage)
print('Maximum memory usage: %s' % max(mem_usage))

""" booster = model.booster_
model_dump = booster.dump_model()

# Save the model in JSON format
with open('lightgbm_model.json', 'w') as f:
    json.dump(model_dump, f) """


##################RESULTS#################

""" Execution time: 2.576854705810547 seconds

Memory usage (in chunks of .1 seconds): [423.99609375, 423.99609375, 439.171875, 457.734375, 476.0390625, 494.859375, 512.03515625, 512.03515625, 512.03515625, 512.03515625, 512.1640625, 512.1640625, 512.1640625, 424.42578125]

Maximum memory usage: 512.1640625 MB"""