#!/bin/bash

dataset="$1"
labels="$2"
start_line=121000
end_line=122000

ndataset="ndataset.csv"
nlabels="nlabels.csv"

# Use sed to extract lines from start_line to end_line
sed -n "${start_line},${end_line}p" "$dataset" > "$ndataset"
sed -n "${start_line},${end_line}p" "$labels" > "$nlabels"

