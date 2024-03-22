#!/bin/bash

DATADIR=data/eterna100_v2
METALEARNA_DIR=data/metalearna
LIBLEARNA_DIR=data/liblearna

echo $DATADIR

echo "Extracting Eterna data..."

cd data

if [ -d "$DATADIR" ]; then
    echo "Eterna data already extracted."
else
    tar -xzf eterna100_v2.tar.gz
fi


cd ..

# Ensure output directories exist
mkdir -p "$METALEARNA_DIR"
mkdir -p "$LIBLEARNA_DIR"


# run Meta-LEARNA and libLEARNA on the Eterna100 dataset

for file in "$DATADIR"/*.rna; do
    # Get the base filename without extension
    base_filename=$(basename "$file" .rna)
    
    # Read the entire file content into variables
    IFS=$' \t' read -r id structure sequence gc_content energy < "$file"
    
    # Reset IFS to its default value
    unset IFS

    # Create first file in metalearna directory
    echo ">$id
$structure" > "$METALEARNA_DIR/$base_filename.input"
    
    # Create second file in liblearna directory
    echo ">$id
#seq $sequence
#str $structure" > "$LIBLEARNA_DIR/$base_filename.input"
    
    echo "Running libLEARNA on $base_filename"
    liblearna --input_file $LIBLEARNA_DIR/$base_filename.input --timeout 3 --show_all_designs --results_dir results/liblearna_one_shot
    
    echo "Running Meta-LEARNA on $base_filename"
    meta-learna --input_file $METALEARNA_DIR/$base_filename.input --num_solutions 1 --hamming_tolerance 1000000000 --results_dir results/metalearna_one_shot
    
    # echo "Files for $base_filename created."
done

rm -rf $METALEARNA_DIR
rm -rf $LIBLEARNA_DIR