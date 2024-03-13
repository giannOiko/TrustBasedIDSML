package structs

type Split struct {
	TrainData    [][]float64
	ValidateData [][]float64
	LabelData    []int
	//trainLabelData []int for unit testing
}
type Config struct {
	TreeNum       int
	Subsamplesize int
}
