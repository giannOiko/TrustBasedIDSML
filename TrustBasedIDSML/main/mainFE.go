package main

import (
	"TrustML/fextraction"
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

	data := utils.LoadData(datafile)
	labels := utils.LoadLabels(labelsfile)

	indexes := fextraction.LassoGridSearchCV(data, labels)
	fmt.Println(indexes)
	//NewDataset := fextraction.RemoveColumns(data, indexes)
	//fmt.Println(data)
	fmt.Println(len(data))
	/* fmt.Println(NewDataset)
	fmt.Println(len(NewDataset)) */
}
