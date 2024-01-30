#!/bin/bash


temp_file="temp_file.csv"
dataset="dataset.csv"

# remove first line
for file in "$@" 
do

    tail -n +2 $file > "$temp_file"
    cat $temp_file  > "$dataset"
done
rm "$temp_file"


labels="labels.csv"
temp_file1="temp.csv"

# Column to extract
column_to_extract="15"  

# Check if input file exists
if [ ! -f "$dataset" ]; then
    echo "Error: Input file '$dataset' not found."
    exit 1
fi

# Extract the specified column and create a new CSV file
cut -d',' -f"$column_to_extract" "$dataset" > "$labels"

awk -v col="$column_to_extract" -F',' '{ for (i = 1; i <= NF; i++) if (i != col) printf "%s%s", $i, (i < NF ? "," : ""); print "" }' "$dataset" > "$temp_file1" && mv "$temp_file1" "$dataset"


