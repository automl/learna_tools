# LEARNA-tools
Generative RNA Design with Automated Reinforcement Learning

`LEARNA-tools` is a Python package that provides the commandline interface of
LEARNA and libLEARNA. The package covers the source code of the following publications

- [Learning to Design RNA](https://openreview.net/pdf?id=ByfyHh05tQ)
- [Partial RNA Design](https://www.biorxiv.org/content/10.1101/2023.12.29.573656v1.full.pdf)

---
## Installation

---
### Requirements

`LEARNA-tools` requires

- Python 3.6
- RNAfold from the ViennaRNA package

However, we provide a `conda` environment for a more convenient installation of `LEARNA-tools`.

### Install conda environment

To install the current version of `LEARNA-tools` from the github repository, first clone the repo as follows

```
git clone https://github.com/Rungetf/learna_tools.git
```

And `cd` into the cloned directory

```
cd learna_tools
```

You can setup the conda environment to include all requirements with

```
conda env create -f environment.yml
```

and

```
conda activate learna_tools
```

### Installation from github repository

When your system satisfies all requirements, you can install `LEARNA-tools` via pip, either directly within the `learna_tools` by running

```
pip install .
```

or from the PyPi package

```
pip install learna_tools
```

### CM design with libLEARNA
To run the CM design with libLEARNA, you need to install Infernal:
```
conda install -c bioconda infernal
```

You can use the latest Rfam database CMs as follows

```
mkdir rfam_cms
```
no go to the directory
```
cd rfam_cms
```
and download Rfam CMs
```
wget https://ftp.ebi.ac.uk/pub/databases/Rfam/CURRENT/Rfam.cm.gz
```
You can unzip the files with
```
gunzip Rfam.cm.gz
```
Then run
```
cmpress Rfam.cm
```

## Usage

We provide simple command line interfaces for the following algorithms

- LEARNA
- Meta-LEARNA
- Meta-LEARNA-Adapt
- libLEARNA

These tools run with the default parameters of each of the algorithms. However, it is also possible to change the parameters.
You can run

```
$ <tool> -h
```
, where `<tool>` is one of `learna, meta-learna, meta-learna-adapt, liblearna`, to see a list of available options for the respective tool.

In the following, we provide some information about the different approaches for RNA design as well as on how to run each individual tool.

### LEARNA

The LEARNA algorithm takes a secondary structure in dot-bracket notation as input to generate a RNA sequence that folds into the desired structure.
The algorithm updates its policy each time it has generated a new sequence and, thus, gets better and better over time by successively updating its weights based on previous predictions.
We provide a version of *LEARNA* with tuned hyperparameters as described in our ICLR'19 paper [Learning to Design RNA](https://openreview.net/pdf?id=ByfyHh05tQ).

#### Input
*LEARNA* either reads a secondary structure directly from the commandline, or from an input file, starting with a structure Id, followed by the desired structure in dot-bracket notation.

An example input file might look as follows:

```
> Test structure
....((((....))))....
```

The easiest way of running `LEARNA` from commandline is to simply type

```
$ learna --target-structure <RNA structure in dot-bracket format>
```

This will run the *LEARNA* algorithm on the secondary structure in dot-bracket notation.

**Note:** LEARNA Does not support pseudoknots. The input structure has to be in standard dot-bracket notation, i.e. the input may only contain `'.', '(', and ')'`.

A real example of a `LEARNA` call then looks as follows

```
$ learna --target-structure ...(((((....)))))...
```

You can use the `--min_solutions` argument to define the number of (optimal) solutions that LEARNA should provide.
Using the `hamming_tolerance` argument, you can further define a distance (Hamming distance between the input structure and the folded candidate sequence) threshold to ask LEARNA to additionally output all sub-optimal solutions with a distance below the given threshold.

For example, the output of the call
```
$ learna --target-structure ...(((((....)))))... --min_solutions 10 --hamming_tolerance 10
```
could then look as follows

|    |   Id |      time |   hamming_distance |   rel_hamming_distance | sequence             | structure            |
|---:|-----:|----------:|-------------------:|-----------------------:|:---------------------|:---------------------|
|  0 |    1 | 0.0187199 |                  0 |                    0   | GUCUACAGCUCUCUGUAUUG | ...(((((....)))))... |
|  1 |    1 | 0.0293458 |                  0 |                    0   | AUUCGAUCCUGCGAUCGCGC | ...(((((....)))))... |
|  2 |    1 | 0.033498  |                  0 |                    0   | GCCGGCGUGCUGACGCCCAA | ...(((((....)))))... |
|  3 |    1 | 0.0387537 |                  0 |                    0   | AAUACUACACCCGUAGUGAA | ...(((((....)))))... |
|  4 |    1 | 0.0474875 |                  0 |                    0   | CUCGAUGACCCCUCAUCCAC | ...(((((....)))))... |
|  5 |    1 | 0.0523767 |                  0 |                    0   | CGGCCAUCAUAUGAUGGACG | ...(((((....)))))... |
|  6 |    1 | 0.116002  |                  0 |                    0   | GCACUAGCUGGAGCUAGCUC | ...(((((....)))))... |
|  7 |    1 | 0.120159  |                  0 |                    0   | ACCAGUUUGUUUAAACUCAC | ...(((((....)))))... |
|  8 |    1 | 0.124296  |                  0 |                    0   | GGAGAAGCUCGGGCUUCGGC | ...(((((....)))))... |
|  9 |    1 | 0.128402  |                  0 |                    0   | AAUUGGAGCGCUCUCCAUCC | ...(((((....)))))... |
| 10 |    1 | 0.0246227 |                  6 |                    0.3 | CUGGGCACUGCGGUGCCCAG | ((((((((....)))))))) |
| 11 |    1 | 0.0428925 |                  6 |                    0.3 | GAUAUGAUGACAAUCAUCAC | ....((((((...)))))). |

**Note:** The last two predictions are sub-optimal with a Hamming distance of 6 each. The output is sorted by Hamming distance.

To run LEARNA from a given file input type

```
$ learna --input_file learna_example_input.in --min_solutions 10000 --timeout 100000
```
This will run learna until it gathered 10000 solutions for the target defined in `learna_example_input.in`.

**Note** that we set the timeout to 100000 seconds in order to ensure that all predictions will be provided correctly.
The default timeout for all learna-based approaches is set to 600 seconds, which might be to small to find 10000 solutions, depending on the input.
However, 100000 seconds is a very high threshold for comon usecases and serves just as an example here.


### Meta-LEARNA

Meta-LEARNA is a version of the LEARNA algorithm that has meta-learned an RNA design policy across thousands of different RNA design tasks.
The algorithm samples sequences from the learned policy without further parameter updates and, thus, allows to find solutions very quickly.
The down-side of the Meta-LEARNA approach is that the learned policy might leverage certain short-cuts in the folding engine it was trained on (RNAfold).
This means that the sequences might be biased towards predictions with G-C pairs.
However, Meta-LEARNA is very useful to quickly provide solutions for a given input structure, however, typically with little sequence diversity.

To run Meta-LEARNA instead of LEARNA, simply replace `learna` with `m̀eta-learna` in the previous calls.
an example run of Meta-LEARNA then looks as follows:

```
$ meta-learna --input_file learna_example_input.in --min_solutions 10000 --timeout 100000
```


### Meta-LEARNA-Adapt

The Meta-LEARNA-Adapt algorithm seeks to combine the best of both, learna and Meta-LEARNA.
The algorithm samples sequences for a given input target structure from a learned policy. However, Meta-LEARNA-Adapt updates its parameters with every prediction such that it adapts to a given input target.

To run Meta-LEARNA-Adapt, simply call

```
$ meta-learna-adapt --input_file learna_example_input.in --min_solutions 10000 --timeout 100000
```

**Nonte:** To increase the diversity of predictions, we provide the `--diversity_loss` option for all LEARNA-based algorithms.
With this option, a loss is added to the general structural loss function that penalizes predictions of the same sequence multiple times.
While we did not use this option during training, our adaptive approaches LEARNA and Meta-LEARNA-Adapt will be informed about the prediction diversity during inference.
This option is particularly useful when running the algorithms to provide larger amounts of candidates for a given target, since both algorithms learn to solve the target better and better with every prediction. This sometimes results in predicting very similar sequences at late prediction stages.

an example call including the diversity loss looks as follows.

```
$ meta-learna-adapt --input_file learna_example_input.in --min_solutions 10000 --timeout 100000 --diversity_loss
```

### libLEARNA

libLEARNA is an algorithm that

## Python Interfaces

All our tools can also be run directly using python, or imported via `learna_tools`.
We will now explain how to run the different tools using python and how to import the tools as modules in your research.

### Run `learna_tools` via Python


### Import `learna_tools` into an existing python script




## Automated Reinforcement Learning

LEARNA as well as libLEARNA are [automated reinforcement learning](https://jair.org/index.php/jair/article/view/13596/26808) algorithm that uses an efficient Bayesian Optimization method, [BOHB](https://proceedings.mlr.press/v80/falkner18a/falkner18a.pdf),
to automatically find the best model for solving the RNA design problem. To learn more about automated machine learning, we refer to the [autoML website](https://www.automl.org/). For more about BOHB, see the [documentation](https://automl.github.io/HpBandSter/build/html/optimizers/bohb.html).

We will continue with explaining how you can rerun the meta-optimization of the different `learna_tools`, or setup an own meta-optimization loop if needed for your research.

### Configuration Space

### Worker

### Meta-Optimization


```
python -m bohb
```

