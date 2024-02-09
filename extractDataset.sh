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


# column extraction
labels="labelss.csv"
num_labels="labels.csv"
temp_file1="temp.csv"

column_to_extract1="4"
column_to_extract2="5" 
column_to_extract3="10" 
column_to_extract4="11" 
column_to_extract5="15"


# Extract the specified column and create a new CSV file
cut -d',' -f"$column_to_extract5" "$dataset" > "$labels"

#convert labels into  and 1s
sed 's/malicious/1/g; s/benign/0/g; s/outlier/1/g' "$labels" > "$num_labels"
rm "$labels"

#extract undesired columns
awk -v col1="$column_to_extract1" -v col2="$column_to_extract2" -v col3="$column_to_extract3" -v col4="$column_to_extract4" -v col5="$column_to_extract5" -F',' '{ for (i = 1; i <= NF; i++) if (i != col1 && i != col2 && i != col3 && i != col4 && i != col5) printf "%s%s", $i, (i < NF ? "," : ""); print "" }' "$dataset" > "$temp_file1" && mv "$temp_file1" "$dataset"

