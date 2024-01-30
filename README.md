# LEARNA-tools
Generative RNA Design with Automated Reinforcement Learning.

The `learna_tools` package provides commandline interfaces for
LEARNA, Meta-LEARNA, Meta-LEARNA-Adapt, and libLEARNA as described in the following publications

- [Learning to Design RNA](https://openreview.net/pdf?id=ByfyHh05tQ) (ICLR'19)
- [Partial RNA Design](https://www.biorxiv.org/content/10.1101/2023.12.29.573656v1.full.pdf) (Under Review @ISMB 2024)

---

## Installation

---
### Requirements

`learna_tools` requires

- Python 3.6
- RNAfold from the ViennaRNA package
- [Infernal](https://academic.oup.com/bioinformatics/article/29/22/2933/316439) [optional]
- [intaRNA](https://academic.oup.com/nar/article/45/W1/W435/3796327) [optional]
- [VARNA](https://academic.oup.com/bioinformatics/article/25/15/1974/210730) [optional]

However, we provide a `conda` environment for a more convenient installation of `learna_tools`.

### Install conda environment

To install the current version of `learna_tools` from the github repository, first clone the repo as follows

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

When your system satisfies all requirements, you can install `learna_tools` via pip

```
pip install .
```

---

## libLEARNA
libLEARNA is the most recent algorithm from the LEARNA family of algorithms. It provides an interface to design RNAs for the partial RNA design paradigm.
In essence, libLEARNA can design RNAs from sequence and structure motifs under different objectives.
For more information, take a look into our [bioRxiv paper](https://www.biorxiv.org/content/10.1101/2023.12.29.573656v1.full.pdf) that is currently under review at ISMB 2024.

### General usage
The general interface to libLEARNA is as follows
```
liblearna --input_file <path to file> [options]
```
To see a list of available command line options, run
```
liblearna -h
```

### Program Input
libLEARNA requires sequence and structure inputs to be defined in an input file with specific tags.
Sequence input follow after a `#seq` tag, and structure inputs follow after a `#str` tag.
A typical input file for inverse RNA folding on the Frog Foot example of the [Eterna100 benchmark](https://github.com/eternagame/eterna100-benchmarking) then looks as follows

```
>Frog Foot
#seq NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN
#str ..........((((....))))((((....))))((((...))))
```
*Note:* We use the letter `N` to denote unconstrained positions.

You can find the example file for the Frog Foot task in the `examples` directory as `if_frog_foot_example_liblearna.input`.

To run libLEARNA on the example, requesting 10 candidates, you can use
```
liblearna --input_file examples/if_frog_foot_example_liblearna.input
```

If you would like to design RNAs from sequence and structure motifs, you can use `whitespaces` or `X` ('Xtend here') to mark positions for exploration (denoted $\overset{\ast}{?}$ in out ISMB submission).
The input file for the design of theophylline riboswitch constructs for example looks as follows

```
>theophylline riboswitch example
#seq AAGUGAUACCAGCAUCGUCUUGAUGCCCUUGGCAGCACUUCA NNNNNN NNNNNNNNNN UUUUUUUU
#str ........NNN(((((.....)))))...NNN(((((((((( NN.... )))))))))) N.......
```
which is the same as
```
>theophylline riboswitch example with X
#seq AAGUGAUACCAGCAUCGUCUUGAUGCCCUUGGCAGCACUUCAXNNNNNNXNNNNNNNNNNXUUUUUUUU
#str ........NNN(((((.....)))))...NNN((((((((((XNN....X))))))))))XN.......
```
To run libLEARNA on the riboswitch design task, you can use

```
liblearna --input_file examples/riboswitch_design_example.input --min_length 66 --max_length 91
```
To additionally specify a desired GC-content for the design, one can use the `--desired_gc` option with a given tolerance via the `--gc_tolerance` option.
The default tolerance is set to `0.01`.
An example call could look as follows:
```
liblearna --input_file examples/riboswitch_design_example.input --min_length 66 --max_length 91 --desired_gc 0.5 --gc_tolerance 0.1
```
Alternatively, a desired GC-content can also be specified in the input file via the `#globgc` tag:
```
>theophylline riboswitch example with GC content
#seq AAGUGAUACCAGCAUCGUCUUGAUGCCCUUGGCAGCACUUCA NNNNNN NNNNNNNNNN UUUUUUUU
#str ........NNN(((((.....)))))...NNN(((((((((( NN.... )))))))))) N.......
#globgc 0.5
```

#### Specifiying Multiple Inputs
You can define multiple targets in a single input file for libLEARNA.
As an example, we use the Riboswitch design examples from above to design RNAs with multiple desired GC contents.

```
>theophylline riboswitch example 1
#seq AAGUGAUACCAGCAUCGUCUUGAUGCCCUUGGCAGCACUUCA NNNNNN NNNNNNNNNN UUUUUUUU
#str ........NNN(((((.....)))))...NNN(((((((((( NN.... )))))))))) N.......
#globgc 0.3
>theophylline riboswitch example 2
#seq AAGUGAUACCAGCAUCGUCUUGAUGCCCUUGGCAGCACUUCA NNNNNN NNNNNNNNNN UUUUUUUU
#str ........NNN(((((.....)))))...NNN(((((((((( NN.... )))))))))) N.......
#globgc 0.4
>theophylline riboswitch example 3
#seq AAGUGAUACCAGCAUCGUCUUGAUGCCCUUGGCAGCACUUCA NNNNNN NNNNNNNNNN UUUUUUUU
#str ........NNN(((((.....)))))...NNN(((((((((( NN.... )))))))))) N.......
#globgc 0.5
>theophylline riboswitch example 4
#seq AAGUGAUACCAGCAUCGUCUUGAUGCCCUUGGCAGCACUUCA NNNNNN NNNNNNNNNN UUUUUUUU
#str ........NNN(((((.....)))))...NNN(((((((((( NN.... )))))))))) N.......
#globgc 0.6
```
libLEARNA will successively process each task, reporting results whenever a task is done.

### Command Line Options
#### Generating Multiple Candidates
By default libLEARNA generates a single solution. 
However, one can specify any number of solutions for a given design via the `--num_solutions <natural number>` option.
The requested number of solutions counts for each individual task provided, in case multiple input tasks are provided.
An example call to libLEARNA then could look as follows:
```
liblearna --input_file examples/if_frog_foot_example_liblearna.input --num_solutions 20
```

#### Changing the Folding Algorithm

To change the folding algorithm from RNAfold MFE predictions to RNAfold with MEA predictions, you can use the `--algorithm` option.
For example
```
liblearna --input_file examples/if_frog_foot_example_liblearna.input --num_solutions 10 --algorithm rnafold_mea
```
runs libLEARNA on the Frog Foot example with the MEA folding objective.

#### Plotting Results
You can directly generate logo plots using the logo_maker Python package.
```
liblearna --input_file examples/if_frog_foot_example_liblearna.input --num_solutions 10 --plot_logo
```
The plots will by default be saved into the `plots` directory relative to the current working directory.
However, you can specify a directory for saving plots via the `--plotting_dir <path to directory>` option.
To visualize plots besides saving, you can use the `--show_plots` flag.

libLEARNA also supports plotting of the structures via VARNA via the `plot_structure` flag. 
This will generate secondary structure plots of all generated solutions.
These plots will also be saved into the `plots` directory (or the directory for saving plots provided with the `--plotting_dir` option), and can be directly visualized with the `--show_plots` flag.

#### Handling Results
You can save the predictions of libLEARNA with the `--results_dir <path to directory` option. libLEARNA allows to select from three output formats: `pickle`, `csv`, and `fasta`.
The output format is specified via the `--output_format <string>` option.
By default, results are saved in a pandas data frame in `pickle` format.
You can specify a comma separated output of the dataframe when choosing the `csv` format.
The `fasta` format outputs a file with the following lines:

```
>ID of task
<sequence>
<structure in dot-bracket format>
```

#### Limiting the Runtime
You can limit the runtime of libLEARNA via the `--timeout <time in seconds>` option.
The provided limit counts for each target if providing multiple input tasks.

For resetting the weights of the policy network to their initial values, you can use the `--restart_timeout <time in seconds>` option.
The algorithm will start from scratch after the provided time.

#### Show all Designs
While most of the time w are interested in the solutions, we can also add the `--show_all_designs` flag to output all the designed candidates.
We note, however, that this option should be used with care because, depending on the specified runtime and the task, there might be a large number of candidates.

### CM design with libLEARNA
To run the CM design with libLEARNA, you need to install Infernal:
```
conda install -c bioconda infernal
```
If you use the recommended install via the provided `environment.yml` file, Infernal is already installed.

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

To design RNAs that match the Hammrhead ribozyme (Type III) family as described in our recent [paper](https://www.biorxiv.org/content/10.1101/2023.12.29.573656v1.full.pdf), you can then for example run

```
liblearna --input_file examples/cm_design.input --num_solutions 20 --cm_design --cm_path rfam_cms/Rfam.cm --cm_name RF00008 --cm_threshold 10 --min_length 50 --max_length 60
```
with the following options
- `--input_file`: The sequence and structure restrictions on the design space.
- `--num_solutions`: The number of solutions provided by libLEARNA.
- `--cm_design`: Flag to design RNAs for a given covariance model.
- `--cm_path`: The path to the covariance model database.
- `--cm_name`: The name of the covariance model that we want to design RNAs for.
- `--cm_threshold`: A bitscore threshold to decide which candidates count as solutions.
- `--min_length`: The minimum length of the designed candidates.
- `--max_length`: The maximum length of the designed candidates.

### RRI Design with libLEARNA
To run the RRI design with libLEARNA, you require [intaRNA](https://rna.informatik.uni-freiburg.de/IntaRNA/Input.jsp).
You can install intaRNA via conda as follows
```
conda install intarna=1.2.5 -c bioconda 
```
If you are using the recommended install via the provided `environment.yml`, intaRNA is already installed.

To run libLEARNA for RRI design, you can use the following call

```
liblearna --input_file examples/rri_design.input --rri_design --rri_threshold 10 --min_length 50 --max_length 60
```
This will use the example design space without any restrictions (unconstrained design space in our ISMB submission) with the default target mRNA.
However, you can use a new target via the `--rri_target` option, followed by the target sequence.
The `--rri_threshold` parameter allows to set a threshold for the reported canidates.
We use positive numbers here, because the threshold is based on the reward of libLEARNA, however, since we are optimizing for energy, the actual threshold from our example corresponds to an energy threshold of -10.

---
## LEARNA

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

