package utils

import (
	"TrustML/structs"
	"bufio"
	"encoding/csv"
	"fmt"
	"math"
	"math/rand"
	"os"
	"strconv"

	"github.com/pa-m/sklearn/metrics"
	"gonum.org/v1/gonum/mat"
)

// load file Data and Return [][]float64 array
func LoadData(myfile string) [][]float64 {

	l := false
	file, err := os.Open(myfile)
	if err != nil {
		fmt.Println("Error opening file:", err)
		return nil
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		fmt.Println("Error reading CSV:", err)
		return nil
	}

	inputData := make([][]float64, len(records))
	for i, row := range records {

		inputData[i] = make([]float64, len(row))
		for j, value := range row {
			inputData[i][j], err = strconv.ParseFloat(value, 64)
			if err != nil {
				fmt.Printf("Error converting to float: line %d", i)
				l = true
				break
			}
		}
		if l {
			break
		}
	}
	return inputData
}

// load label data and return label array
func LoadLabels(filePath string) []int {
	file, err := os.Open(filePath)
	if err != nil {
		fmt.Println("Error opening file:", err)
		return nil
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)

	var labels []int

	for scanner.Scan() {
		label, err := strconv.Atoi(scanner.Text())
		if err != nil {
			// Handle the error if the conversion fails
			fmt.Println("Error:", err)
			return nil
		}
		labels = append(labels, label)
	}
	return labels
}

// Stratified split the dataset according to split Ratio, return training and testing set and anomaly score..
func SplitDataset(dataset [][]float64, labels []int, splitRatio float64) (map[int][][]float64, map[int][][]float64, float64) {

	myMap := make(map[int][][]float64)
	TrainingIndex_Data := make(map[int][][]float64)
	TestingIndex_Data := make(map[int][][]float64)
	count := 0
	for i, instance := range dataset {
		if labels[i] == 0 {
			myMap[0] = append(myMap[0], instance)
		} else {
			myMap[1] = append(myMap[1], instance)
			count++
		}
	}
	ShuffleUnit(myMap)
	split0 := int(math.Round(splitRatio * float64(len(myMap[0]))))
	split1 := int(math.Round(splitRatio * float64(len(myMap[1]))))

	TrainingIndex_Data[0] = myMap[0][:split0]
	TrainingIndex_Data[1] = myMap[1][:split1]
	TestingIndex_Data[0] = myMap[0][split0:]
	TestingIndex_Data[1] = myMap[1][split1:]

	anomaly := float64(count) / float64(len(labels))
	return TrainingIndex_Data, TestingIndex_Data, anomaly
}

// Convert labels to float for F1 and Accuracy Score metrics
func ConvertToFloat(int_array []int) []float64 {

	length := len(int_array)
	float_array := make([]float64, length)

	for i := 0; i < length; i++ {

		float_array[i] = float64(int_array[i])
	}
	return float_array

}

func F1Score(labelsPred []int, labelsTru []int) float64 {

	pred := ConvertToFloat(labelsPred)
	tru := ConvertToFloat(labelsTru)
	Ypred, Ytrue := mat.NewDense(len(pred), 1, pred), mat.NewDense(len(tru), 1, tru)
	var sampleWeight []float64
	/* fmt.Printf("F1 macro %.2f\n", metrics.F1Score(Ytrue, Ypred, "macro", sampleWeight))
	fmt.Printf("F1 micro %.2f\n", metrics.F1Score(Ytrue, Ypred, "micro", sampleWeight))
	fmt.Printf("F1 weighted %.2f\n", metrics.F1Score(Ytrue, Ypred, "weighted", sampleWeight)) */
	return metrics.F1Score(Ytrue, Ypred, "weighted", sampleWeight)

}

func AccuracyScore(labelsPred []int, labelsTru []int) float64 {

	pred := ConvertToFloat(labelsPred)
	tru := ConvertToFloat(labelsTru)

	var nilDense *mat.Dense
	normalize, sampleWeight := true, nilDense
	Ypred, Ytrue := mat.NewDense(len(pred), 1, pred), mat.NewDense(len(tru), 1, tru)
	return metrics.AccuracyScore(Ytrue, Ypred, normalize, sampleWeight)

}

func RocAucScore(labelsPred []int, labelsTru []int) float64 {
	pred := ConvertToFloat(labelsPred)
	tru := ConvertToFloat(labelsTru)

	Ypred, Ytrue := mat.NewDense(len(pred), 1, pred), mat.NewDense(len(tru), 1, tru)
	return metrics.ROCAUCScore(Ytrue, Ypred, "weighted", nil)
}

// return mean of float array
func Mean(array []float64) float64 {
	sum := 0.0
	for _, num := range array {
		sum += num
	}

	// Calculate the mean
	mean := sum / float64(len(array))
	return mean
}

// Compute Outlier Ratio of a label array
func OutlierRatio(lab []int) float64 {
	count := 0
	for _, v := range lab {
		if v == 1 {
			count++
		}
	}
	anomaly := float64(count) / float64(len(lab))
	return anomaly
}

// find index with max value of array
func FindMaxIndex(arr []float64) int {

	maxIndex := 0

	for i := 1; i < len(arr); i++ {

		if arr[i] > arr[maxIndex] {
			maxIndex = i
		}
	}

	return maxIndex
}

// Build new array of labels, because of all the data manipulation it is needed to keep track of the labels as well.
func BuildLabelsArray(lendata int, lenzerodata int) []int {

	labeldata := make([]int, lendata)
	for i := 0; i < lendata; i++ {
		if i < lenzerodata {
			labeldata[i] = 0
		} else {
			labeldata[i] = 1
		}
	}
	return labeldata
}

// Return the configuration with the highset metric score from a map with key : config , value : score
func FindBestConfig(themap map[structs.Config][]float64) *structs.Config {

	highest_mean := 0.0
	var bestconf *structs.Config
	for conf, f1s := range themap {
		mean := Mean(f1s)

		if mean > highest_mean {
			highest_mean = mean
			bestconf = &structs.Config{
				TreeNum:       conf.TreeNum,
				Subsamplesize: conf.Subsamplesize,
			}
		}
	}
	fmt.Println("Highest mean is: ", highest_mean)
	return bestconf
}

// Flatten 2D array to 1D for mat.Dense requirements
func Flatten2D(arr [][]float64) []float64 {
	var flattened []float64
	for _, row := range arr {
		flattened = append(flattened, row...)
	}
	return flattened
}

// shuffle both dataset and labels at the same indexes
func ShuffleDataset(dataset [][]float64, labels []int) {

	rand.Shuffle(len(labels), func(i, j int) {
		dataset[i], dataset[j] = dataset[j], dataset[i]
		labels[i], labels[j] = labels[j], labels[i]
	})
}

// Shuffle Training, Testing and Testing label set for K-Fold CV
func Shuffle(testing_set [][]float64, training_set [][]float64, label_set []int) {

	rand.Shuffle(len(training_set), func(i, j int) {
		training_set[i], training_set[j] = training_set[j], training_set[i]
	})
	rand.Shuffle(len(testing_set), func(i, j int) {
		testing_set[i], testing_set[j] = testing_set[j], testing_set[i]
		label_set[i], label_set[j] = label_set[j], label_set[i]
	})
}

// Shuffle each unit 0s and 1s individually
func ShuffleUnit(data_label map[int][][]float64) {

	rand.Shuffle(len(data_label[0]), func(i, j int) {
		data_label[0][i], data_label[0][j] = data_label[0][j], data_label[0][i]
	})
	rand.Shuffle(len(data_label[1]), func(i, j int) {
		data_label[1][i], data_label[1][j] = data_label[1][j], data_label[1][i]
	})
}

// Merge two 2D arrays
func Merge(data1 [][]float64, data2 [][]float64) [][]float64 {

	concatenated := make([][]float64, len(data1)+len(data2))
	copy(concatenated, data1)
	copy(concatenated[len(data1):], data2)
	return concatenated
}
