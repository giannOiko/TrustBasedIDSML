package main

import (
	"TrustML/utils"
	"fmt"
	"os"
	"path/filepath"
)

func main() {

	dataFolder := filepath.Join(filepath.Dir(os.Args[1]), "..", "data")

	// Construct the full path to the file.
	//datafile := filepath.Join(dataFolder, os.Args[1])
	labelsfile := filepath.Join(dataFolder, os.Args[1])

	labels := utils.LoadLabels(labelsfile)
	fmt.Println(utils.OutlierRatio(labels))
}
