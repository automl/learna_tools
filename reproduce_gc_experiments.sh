#! /bin/bash

DATADIR=data/ArchiveII

LIBLEARNA_DIR=data/liblearna

echo "Extracting Eterna data..."

cd data

if [ -d "$DATADIR" ]; then
    echo "ArchiveII data already extracted."
else
    tar -xzf ArchiveII.tar.gz
fi


cd ..

# Ensure output directories exist
mkdir -p "$LIBLEARNA_DIR"

# run Meta-LEARNA and libLEARNA on the Eterna100 dataset

for file in "$DATADIR"/*.rna; do
    # Get the base filename without extension
    base_filename=$(basename "$file" .rna)
    
    # Read the entire file content into variables
    IFS=$' \t' read -r id structure sequence gc_content energy < "$file"
    
    # Reset IFS to its default value
    unset IFS
    
    # Create file in liblearna directory
    echo ">$id
#seq $sequence
#str $structure" > "$LIBLEARNA_DIR/$base_filename.input"
    
    echo "Running libLEARNA on $base_filename"
    liblearna --input_file $LIBLEARNA_DIR/$base_filename.input --timeout 3600 --show_all_designs --results_dir results/liblearna_one_shot --restart_timeout 1800 --gc_tolerance 0.01 --desired_gc $gc_content
        
    # echo "Files for $base_filename created."
done

rm -rf $LIBLEARNA_DIR