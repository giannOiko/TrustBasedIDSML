package main

import (
	"fmt"
	"os"
	"math"
	"encoding/csv"
	"github.com/dmitryikh/leaves"
	"strconv"
	"time"
	"syscall"
	"os/exec"
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

	
	// 2. Load csv file
	test_data := LoadData("mirai3.csv")

	//Measure cpu,mem alloc by calling monitor.sh script

	cmd := exec.Command("./monitor.sh")

	// Set the command to run in the background (as a child process)
	cmd.SysProcAttr = &syscall.SysProcAttr{
		Setpgid: true, // Ensure the process is put in a separate process group
	}

	// Start the Bash script
	err = cmd.Start()
	if err != nil {
		fmt.Println("Error starting the script:", err)
		return
	}
	//start measuring time
	start := time.Now()
	
	pred_labels := probToLabels(model, test_data)

	elapsed := time.Since(start)


	// Kill the process (gracefully)
	fmt.Println("Killing the memory monitor...")
	err = cmd.Process.Kill()
	if err != nil {
		fmt.Println("Error killing the process:", err)
		return
	}

	//append the time took to log file
	file, err = os.OpenFile("ps.log",os.O_APPEND|os.O_WRONLY,0644)
	if err != nil {
		fmt.Println("Error creating file:", err)
		return
	}
	defer file.Close()  // Ensure the file is closed after the operation

	_, err = fmt.Fprintf(file, "Go Predict took %s to execute\n", elapsed)
	if err != nil {
		fmt.Println("Error writing to file:", err)
		return
	}

	fmt.Println(pred_labels[71000])

	 
}

//convert probabilities to Labels
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

//load csv file
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