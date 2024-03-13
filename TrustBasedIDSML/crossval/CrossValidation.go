package crossval

import (
	"TrustML/forest"
	"TrustML/structs"
	"TrustML/utils"
	"fmt"
	"math"
)

func StratifiedKFold(K int, data_label map[int][][]float64) (ch chan structs.Split) {

	foldSizezeros := int(math.Round(float64((len(data_label[0]) / K))))
	foldSizeones := int(math.Round((float64(len(data_label[1]) / K))))

	ch = make(chan structs.Split)

	go func() {

		var sp *structs.Split
		for i := 0; i < K; i++ {
			startIndex0 := i * foldSizezeros
			endIndex0 := (i + 1) * foldSizezeros
			startIndex1 := i * foldSizeones
			endIndex1 := (i + 1) * foldSizeones

			if i == K-1 {
				endIndex0 = len(data_label[0])
				endIndex1 = len(data_label[1])
			}
			trainData0 := make([][]float64, 0)
			trainData1 := make([][]float64, 0)

			// Merge all data except the test data into the training data
			trainData0 = append(trainData0, data_label[0][:startIndex0]...)
			trainData0 = append(trainData0, data_label[0][endIndex0:]...)
			trainData1 = append(trainData1, data_label[1][:startIndex1]...)
			trainData1 = append(trainData1, data_label[1][endIndex1:]...)

			trainData := utils.Merge(trainData0, trainData1)

			validData0 := data_label[0][startIndex0:endIndex0]
			validData1 := data_label[1][startIndex1:endIndex1]

			validData := utils.Merge(validData0, validData1)

			validDatalen := len(validData)

			labeldata := utils.BuildLabelsArray(validDatalen, len(validData0))

			utils.Shuffle(validData, trainData, labeldata)

			sp = &structs.Split{
				TrainData:    trainData,
				ValidateData: validData,
				LabelData:    labeldata,
				//trainLabelData: trainlabeldata,
			}
			ch <- *sp
		}
		close(ch)
	}()
	return ch
}

func GridSearchCV(data map[int][][]float64, treenummax int, subsamplmax int, anomaly float64) *structs.Config {

	ConfF1s := make(map[structs.Config][]float64, 0)
	var conf structs.Config
	i := 1

	for s := range StratifiedKFold(5, data) {
		treenumStep := 10
		fmt.Println("Entering Iteration num ", i)
		for tr_n := 10; tr_n <= treenummax; tr_n += treenumStep {
			for sss := 50; sss <= subsamplmax; sss += 50 {
				conf = structs.Config{
					TreeNum:       tr_n,
					Subsamplesize: sss,
				}
				forest := forest.IsoForestTrain_Test(s.TrainData, s.ValidateData, tr_n, sss, anomaly)
				f1Score := utils.F1Score(s.LabelData, forest.Labels)
				ConfF1s[conf] = append(ConfF1s[conf], f1Score)
			}
			if tr_n == 100 {
				treenumStep = 100
			} else if tr_n == 1000 {
				treenumStep = 500
			}
		}
		i++
	}
	bestConfig := utils.FindBestConfig(ConfF1s)
	fmt.Println(ConfF1s)

	return bestConfig
}
