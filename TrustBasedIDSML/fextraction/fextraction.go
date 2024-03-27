package fextraction

import (
	"TrustML/utils"
	"fmt"
	"math"
	"math/rand"

	"github.com/pa-m/sklearn/base"
	li "github.com/pa-m/sklearn/linear_model"
	"github.com/pa-m/sklearn/metrics"
	ms "github.com/pa-m/sklearn/model_selection"
	"gonum.org/v1/gonum/mat"
)

// Make dataset Balanced with undersampling for optimal lasso regression
func BalanceData(dataset [][]float64, labels []int) ([][]float64, []int) {

	//get 0's and 1's from dataset
	myMap := make(map[int][][]float64)
	count := 0
	for i, instance := range dataset {
		if labels[i] == 0 {
			myMap[0] = append(myMap[0], instance)
		} else {
			myMap[1] = append(myMap[1], instance)
			count++
		}
	}

	utils.ShuffleUnit(myMap)

	// determine minority and majority class
	var MajorityClass, MinorityClass [][]float64
	var Maj bool
	if len(myMap[0]) > len(myMap[1]) {
		MajorityClass = make([][]float64, len(myMap[0]))
		MinorityClass = make([][]float64, len(myMap[1]))
		copy(MajorityClass, myMap[0])
		copy(MinorityClass, myMap[1])
		Maj = true
	} else {
		MajorityClass = make([][]float64, len(myMap[1]))
		MinorityClass = make([][]float64, len(myMap[0]))
		copy(MajorityClass, myMap[1])
		copy(MinorityClass, myMap[0])
		Maj = false
	}

	fmt.Println("leeenn", len(MinorityClass))

	/* fmt.Println(MajorityClass)
	fmt.Println(MinorityClass) */

	// shuffle majority class for unbiased sampling
	rand.Shuffle(len(MajorityClass), func(i, j int) {
		MajorityClass[i], MajorityClass[j] = MajorityClass[j], MajorityClass[i]
	})

	// undersample majority class in length of minority class
	EqualClass := MajorityClass[:len(MinorityClass)]

	//Merge both classes
	var lenzerodata int
	var data [][]float64
	if Maj {
		data = utils.Merge(EqualClass, MinorityClass)
		lenzerodata = len(EqualClass)
		fmt.Println("zeros is maj")
	} else {
		data = utils.Merge(MinorityClass, EqualClass)
		lenzerodata = len(MinorityClass)
		fmt.Println("zeros is min")
	}

	//build new labels like the data we sampled
	lab := utils.BuildLabelsArray(len(data), lenzerodata)

	//shuffle the new dataset
	utils.ShuffleDataset(data, lab)

	return data, lab

}

func nearlyEqual(a, b float64) bool {
	const epsilon = 2.220446049250313e-4 //FLT_EPSILON for float64

	// Absolute difference between a and b
	diff := math.Abs(a - b)

	// Check if the difference is less than or equal to epsilon
	return diff <= epsilon
}

func LassoGridSearchCV(data [][]float64, labels []int) []int {

	//Get balanced data
	Balanced_Data, Balanced_Labels := BalanceData(data, labels)

	NSamples := len(labels)
	NFeatures := len(data[0])
	NSamples_bal := len(Balanced_Labels)
	NFeatures_bal := len(Balanced_Data[0])

	//Create data and label matrices
	X, Y := mat.NewDense(NSamples, NFeatures, utils.Flatten2D(data)), mat.NewDense(NSamples, 1, utils.ConvertToFloat(labels))
	X_bal, Y_bal := mat.NewDense(NSamples_bal, NFeatures_bal, utils.Flatten2D(Balanced_Data)), mat.NewDense(NSamples_bal, 1, utils.ConvertToFloat(Balanced_Labels))

	//initialiaze lasso parameters
	lasso := li.NewLasso()
	lasso.FitIntercept = true
	lasso.Normalize = true
	lasso.L1Ratio = 1
	lasso.MaxIter = 1e5
	lasso.Tol = 1e-4

	//Make alpha parameter grid. Static from 1e-10 multiplied by 10, 10 times until 0.1.
	gridLength := 12
	start_value := 1e-10

	type ParamGrid map[string][]interface{}

	paramGrid := ParamGrid{
		"Alpha": make([]interface{}, gridLength),
	}

	for i := 0; i < gridLength; i++ {
		paramGrid["Alpha"][i] = start_value
		start_value = start_value * 10
	}

	fmt.Println(paramGrid)

	//Scorer is R2
	scorer := func(Y, Ypred mat.Matrix) float64 {
		return metrics.R2Score(Y, Ypred, nil, "").At(0, 0)
	}

	RandomState := base.NewLockedSource(7)

	//Instatiate Grid Search CV for lasso regression.. 3 repetitions K-Fold
	lassocv := &ms.GridSearchCV{
		Estimator:          lasso,
		ParamGrid:          paramGrid,
		Scorer:             scorer,
		LowerScoreIsBetter: true,
		CV:                 &ms.KFold{NSplits: 3, RandomState: RandomState, Shuffle: true},
		Verbose:            true,
		NJobs:              -1}

	//Run Grid Search CV
	lassocv.Fit(X_bal, Y_bal)
	fmt.Println("Alpha", lassocv.BestParams["Alpha"])

	//Set lasso ALpha the best returned ALpha from CV
	//lasso.Alpha = lassocv.BestParams["Alpha"].(float64)
	lasso.Alpha = 1e-10

	//Run lasso regression
	/////////////////////pare ola ta data
	lasso.Fit(X, Y)

	//Predict coefficients see which are set to 0 and return the indexes that are not zero
	coef := lasso.Coef
	r, _ := coef.Dims()
	indexes := make([]int, 0)

	for i := 0; i < r; i++ {
		value := coef.At(i, 0)
		fmt.Printf("Matrix[%d][0]: %.5f\n", i, value)
		//valuee := float64(value)
		if !nearlyEqual(value, 0.0) {
			indexes = append(indexes, i)
		}

	}

	fmt.Printf("columns to keep = %v", indexes)
	return indexes
}

// remove zero coefficient columns from dataset
func RemoveColumns(dataset [][]float64, columnsToKeep []int) [][]float64 {

	newDataset := make([][]float64, len(dataset))
	for i := range newDataset {
		newDataset[i] = make([]float64, len(dataset[i])-len(columnsToKeep))
	}

	for i, row := range dataset {
		newRow := make([]float64, 0)
		for col, value := range row {
			for _, keepcol := range columnsToKeep {
				if col == keepcol {
					newRow = append(newRow, value)
				} else {
					continue
				}
			}
		}
		newDataset[i] = newRow
	}
	return newDataset
}
