#!/bin/bash

# Auto-run the program "compile_and_run.sh" with groups of arguments combinations

# Go to where the compile_and_run.sh is

# Output file to store the results
output_file="output_table.txt"
input_file="TestingFiles/Names.txt"

# Clean out the output file if it already exists
if [ -f "$output_file" ]; then
    rm "$output_file"
fi

# Define the options for each argument
size=(16 32)
es=(0 1 2 3 4)

# Format the title with fixed-size spaces 
formatted_title=$(printf "%-20s %-15s %-15s %-15s %-30s %-30s %-30s %-30s %-20s %-20s %-20s \n" "Dataset name" "Total data" "Posit Size" "Posit Es" "AbsErr(Posit vs. double)" "AbsErr(single vs. double)"  "RelErr(Posit vs. double)" "RelErr(single vs. double)" "RelErr compare" "Tolcheck(Posit)" "Tolcheck(single)")

# Store the title in a file
echo "$formatted_title" >> "$output_file"

# echo "----------------------------------------------------------------------------" >> "$output_file"


# Run the program with different argument combinations
# Double precision
while read -r item; do

    # Skip empty lines
    if [ -z "$item" ]; then
        continue
    fi 
    echo "Processing matrix: $item"
    

    for arg2 in "${size[@]}"; do
        for arg3 in "${es[@]}"; do

            echo "At Posit($arg2, $arg3)"

            result=$(./main TestingFiles/"$item" "$arg2" "$arg3" 0 0 0)

            # Extract the numbers using grep and store it in the output file

            filename=$item
            # Extract the desired double precision number using grep and store it in the output file
            totalData=$(echo "$result" | grep -Po 'The total number of data is:\s\K[0-9]+')

            # Posits vs Double
            AbsErr_P=$(echo "$result" | grep -Po 'Posit vs Double absolute error:\s\K[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?')
            RelErr_P=$(echo "$result" | grep -Po 'Posit vs Double relative error:\s\K[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?')
            Tol_P=$(echo "$result" | grep -Po 'Posit vs Double tolerance test pass rate:\s\K[-+]?[0-9]*\.?[0-9]+?%')

            # Single vs Double
            single_vs_double_section=$(echo "$result" | awk '/Single vs Double:/, /Double vs Double:/')

            AbsErr_F=$(echo "$single_vs_double_section" | grep -oP 'Absolute error:\s\K[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?')
            RelErr_F=$(echo "$single_vs_double_section" | grep -Po 'Relative error:\s\K[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?')
            Tol_F=$(echo "$single_vs_double_section" | grep -Po 'Tolerance test pass rate:\s\K[-+]?[0-9]*\.?[0-9]+?%')

            RelErr_Compare=$(awk "BEGIN { diff = $RelErr_F / $RelErr_P; printf \"%.3e\", diff}")
           

            formatted_output=$(printf "%-20s %-15s %-15s %-15s %-30s %-30s %-30s %-30s %-20s %-20s %-20s\n" "$filename" "$totalData" "$arg2" "$arg3" "$AbsErr_P" "$AbsErr_F" "$RelErr_P" "$RelErr_F" "$RelErr_Compare" "$Tol_P" "$Tol_F")
            echo "$formatted_output" >> "$output_file"

        
        done
      
    done
done < "$input_file"

