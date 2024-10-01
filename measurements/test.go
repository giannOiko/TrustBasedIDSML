package main

import (
	"fmt"
	"os"
	"math"
	"encoding/csv"
	"github.com/dmitryikh/leaves"
	"strconv"
	"bufio"
	"time"
	"runtime"
)


func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func main() {
	// 1. Read model
	useTransformation := false

	modelFile := "lightgbm_model.json"
	file, err := os.Open(modelFile)
	if err != nil {
		fmt.Printf("error opening file")
	}
	defer file.Close()

	model, err := leaves.LGEnsembleFromJSON(file, useTransformation)
	if err != nil {
		panic(err)
	}

	

	// 2. Do predictions!
	test_data := LoadData("mirai3.csv")
	//test_labels := LoadLabels("data/y_test.csv")

	start := time.Now()

	//Measure Time
	pred_labels := probToLabels(model, test_data)

	elapsed := time.Since(start)

	fmt.Printf("Function took %s to execute\n", elapsed)

	//Measure mem alloc
	before,_ := memoryUsage()

	pred_labels = probToLabels(model, test_data)

	after,peakMemory := memoryUsage()

	fmt.Printf("Memory used: %d bytes\n", after-before)
	fmt.Printf("Peak Memory: %d bytes\n", peakMemory)


	fmt.Println(pred_labels[71000])

	//////////////RESULTS//////////////////////
	//Function took 1.505303218s to execute
	//Memory used: 1602816 bytes = 1.53 MB
	//////////////////////////////////////////
	 
}

func probToLabels(model *leaves.Ensemble, data [][]float64)([]int){
	
	pred_labels := make([]int,len(data))
	for i , sample := range data{
		p := model.PredictSingle(sample, 0)
		sigm := sigmoid(p)
		if sigm >= 0.5{
			pred_labels[i] = 1
		}else
		{
			pred_labels[i] = -1
		}
	}
	return pred_labels
}

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

func memoryUsage() (uint64,uint64) {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	peakMemory := m.HeapAlloc  // This will give the peak heap memory allocated
	return m.Alloc,peakMemory // bytes allocated and still in use
}