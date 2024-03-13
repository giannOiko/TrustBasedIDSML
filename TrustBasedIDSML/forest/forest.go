package forest

//Isolation Forest algorithm
import (
	"github.com/e-XpertSolutions/go-iforest/iforest"
)

func IsoForestTrain_Test(training_data [][]float64, testing_data [][]float64, treesNumber int, subsampleSize int, outliers float64) *iforest.Forest {
	//model initialization
	forest := iforest.NewForest(treesNumber, subsampleSize, outliers)

	//training stage - creating trees
	forest.Train(training_data)

	//testing stage - finding anomalies
	forest.Test(testing_data)

	//threshold := forest.AnomalyBound
	//labels := forest.Labels

	//fmt.Println("threshold is",threshold)

	return forest
}
