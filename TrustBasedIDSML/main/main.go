package main

import (
	"TrustML/crossval"
	"TrustML/fextraction"
	"TrustML/forest"
	"TrustML/utils"
	"fmt"
	"os"
	"path/filepath"
)

func main() {
	// load from files the pr
	// Get the absolute path to the "data" folder relative to the executable.
	dataFolder := filepath.Join(filepath.Dir(os.Args[1]), "..", "data")

	// Construct the full path to the file.
	datafile := filepath.Join(dataFolder, os.Args[1])
	labelsfile := filepath.Join(dataFolder, os.Args[2])
	//HAHAHAHAHAAHAAH
	//load Data and Labels
	data := utils.LoadData(datafile)
	labels := utils.LoadLabels(labelsfile)

	//Run lasso Regression on Grid Search CrossValidation (K=3) for Alpha parameter (Scorer = R2)
	//Run lasso Regression on the best Alpha
	//Return column Indexes with non zero coeficcients
	indexes := fextraction.LassoGridSearchCV(data, labels)
	fmt.Println(indexes)
	//Remove zero coefficient Columns from Dataset and create NewDataset
	NewDataset := fextraction.RemoveColumns(data, indexes)

	//Split The Dataset into training and testing set
	training_data, testing_data, anomaly := utils.SplitDataset(NewDataset, labels, 0.8)
	fmt.Println("anomaly ratio is: ", anomaly)

	//Return the Best configuration{treeNum}{SubsamplingSize} with Iforest Grid Search CV(K=5), Scorer = F1
	configuration := crossval.GridSearchCV(training_data, 2000, 350, anomaly)

	// train Forest with the best conf
	fmt.Println("Training Model with parameters... ", configuration)
	forestt := forest.IsoForestTrain_Test(data, data, configuration.TreeNum, configuration.Subsamplesize, anomaly)

	// prepare the testing data for predict
	testing := utils.Merge(testing_data[0], testing_data[1])

	testDatalen := len(testing)
	testData0len := len(testing_data[0])

	//Build labels according to testing set
	labeldata := utils.BuildLabelsArray(testDatalen, testData0len)

	//Predict
	predictlabels, _, _ := forestt.Predict(testing)
	fmt.Println("Predict Accuracy is...", utils.AccuracyScore(predictlabels, labeldata))
	fmt.Println("Predict F1 is...", utils.F1Score(predictlabels, labeldata))

}
