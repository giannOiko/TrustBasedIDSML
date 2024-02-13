# TrustBasedIDSML
Network intruision detection using ML(Isolation Forest) in Golang
# USAGE

./extractDataset.sh "datasetname.csv" 
- removes unwanted features from dataset.
- extracts feature labels from dataset and creates 2 files... dataset.csv and labels.csv.

Iso.ipynb
- Stratified split the dataset into training and testing set (80/20 of the original dataset).
- Stratified KFold on the training set and grid Search Cross Validation for configuring the best forest parameters, (Metric := F1 SCORE).
- Training with the best parameters on the whole training set(80%) and testing on the testing set(20%).


Dataset https://www.kaggle.com/datasets/mryanm/luflow-network-intrusion-detection-data-set
