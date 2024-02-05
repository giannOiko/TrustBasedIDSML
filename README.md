# TrustBasedIDSML
Network intruision detection using ML(Isolation Forest) in Golang
# USAGE

./extractDataset.sh "datasetname.csv" 
- removes unwanted features
- splits the dataset into 80% training and 20% testing
- creates training.csv and testing.csv
- extracts feature 'labels' into training_labels.csv and testing_labels.csv

Iso.ipynb
-K-Fold Cross Validation on the training.csv dataset (metrics Accuracy, F1 score)
-Predict on unseen data testing.csv dataset
-on the works...

Dataset https://www.kaggle.com/datasets/mryanm/luflow-network-intrusion-detection-data-set
