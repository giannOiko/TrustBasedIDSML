#!/bin/bash

# File name
output_file1=$1
output_file2=$2

# Number of rows and columns
rows=100
columns=2

# Function to generate a random number between 0 and 100
generate_random_number50() {
  echo $((RANDOM % 100 + 50))
}

generate_random_number100() {
  # Generate a random number between 0 and 200
  random_number=$((RANDOM % 201))

  # Check if the number is in the desired ranges
  #if [ $random_number -le 75 ] || [ $random_number -ge 125 ]; then
  #echo $random_number
  #else
    # If not in the desired ranges, recursively call the function again
   # generate_random_number100
  #fi
  echo $random_number
}

# Create CSV file
#echo "Column1,Column2" > "$output_file"

# Generate random numbers and append to the CSV file
for ((i = 0; i < rows; i++)); do
  row="$(generate_random_number50),$(generate_random_number50)"
  echo "$row" >> "$output_file1"
done

for ((i = 0; i < rows; i++)); do
  row="$(generate_random_number100),$(generate_random_number100)"
  echo "$row" >> "$output_file2"
done

echo "CSV file '$output_file' generated with $rows rows and $columns columns."
