#! /bin/bash

cd data

echo "Extracting training data..."

if [ -d "rfam_PD_short_train" ]; then
    echo "Short training data already exists. Skipping extraction."
else
    tar -xzf rfam_PD_short_train.tar.gz
fi

if [ -d "rfam_PD_long_train" ]; then
    echo "Long training data already exists. Skipping extraction."
else
    tar -xzf rfam_PD_long_train.tar.gz
fi

if [ -d "rfam_PD_train" ]; then
    echo "Random training data already exists. Skipping extraction."
else
    tar -xzf rfam_PD_train.tar.gz
fi

echo "Extracting validation data..."
if [ -d "rfam_PD_validation" ]; then
    echo "Validation data already exists. Skipping extraction."
else
    tar -xzf rfam_PD_validation.tar.gz
fi

if [ "$2" == "test" ]; then
    echo "creating test training and validation data..."
    mkdir -p train_test
    cd train_test
    echo -e "1\t(((...)))\tNNNNNNNNN\t0\t0" > 1.rna
    cd ..
    mkdir -p validation_test
    cd validation_test
    echo -e "1\t...(((...)))...\tNNNNNNNNNNNNNNN\t0\t0" > 1.rna
    cd ..
fi

cd ..

echo "Start Meta-Optimization..."

if [ "$2" == "reproduce" ]; then
    echo "Full execution of the optimization process requested."

    echo "Running BOHB with the following parameters:"
    echo "min_budget: 400"
    echo "max_budget: 3600"
    echo "n_iter: 512"
    echo "n_cores: 20"
    echo "shared_dir: results/bohb/$1"
    echo "run_id: $1"
    echo "nic_name: lo"
    echo "data_dir: data"
    echo "n_train_seqs: 100000"
    echo "validation_timeout: 60"
    echo "mode: reproduce"

    echo "This may take a while..."
    python -m learna_tools.liblearna.optimization.bohb --min_budget 400 --max_budget 3600 --n_iter 512 --n_cores 20 --shared_dir results/bohb/$1 --run_id $1 --nic_name lo --data_dir data --n_train_seqs 100000 --validation_timeout 60 --mode reproduce
else
    echo "Test execution of the optimization process requested."
    
    echo "Running BOHB with the following parameters:"
    echo "min_budget: 1"
    echo "max_budget: 9"
    echo "n_iter: 1"
    echo "n_cores: 1"
    echo "shared_dir: results/bohb/$1"
    echo "run_id: $1"
    echo "nic_name: lo"
    echo "data_dir: data"
    echo "n_train_seqs: 1"
    echo "validation_timeout: 1"
    echo "mode: test"

    python -m learna_tools.liblearna.optimization.bohb --min_budget 1 --max_budget 9 --n_iter $3 --n_cores 1 --shared_dir results/bohb/$1 --run_id $1 --nic_name lo --data_dir data  --n_train_seqs 1 --validation_timeout 1 --mode test
fi

python -m learna_tools.liblearna.optimization.analyse.analyse_bohb_results --run $1