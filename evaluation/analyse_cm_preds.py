from pathlib import Path

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--results_dirs', type=str, nargs="+", default='results', help='Directory containing the results files')
parser.add_argument('--chunk_size', type=int, default=500, help='Chunk size for calculating mean and std dev')
parser.add_argument('--num_evaluations', type=int, default=500000, help='Maximum number of evaluations to consider')

args = parser.parse_args()

def process_file(file_path, chunk_size, num_evals):
    df = pd.read_pickle(file_path)

    # Mapping experiment ID and seed
    df = df[:num_evals]
    experiment_id = '-'.join(file_path.stem.split('_')[:-2])
    # print(file_path.stem, df['bitscore'].max(), df['bitscore'].min())
    # df['bitscore'] = df['bitscore'].astype('float').replace(-200, 0.0)

    # Chunking data every N steps and calculating mean and std dev
    df['chunk'] = df.index // chunk_size
    try:
        chunked = df.groupby('chunk')['reward'].agg(['mean', 'std']).reset_index()
    except:
        chunked = df.groupby('chunk')['bitscore'].agg(['mean', 'std']).reset_index()
    chunked['experiment_id'] = experiment_id
    return chunked

paths = []
for results_dir in args.results_dirs:
    paths += list(Path(results_dir).glob('*'))

print(paths)

all_results = []
chunk_size = args.chunk_size  # Change this to your desired chunk size
num_evals = args.num_evaluations
for p in paths:
    print('### Processing file {} ###'.format(p))
    chunked_data = process_file(p, chunk_size, num_evals)
    all_results.append(chunked_data)

# Combine all results into a single DataFrame
combined_df = pd.concat(all_results)

# Aggregate across seeds for each experiment ID
final_agg = combined_df.groupby(['experiment_id', 'chunk']).agg({'mean': ['mean', 'std'], 'std': ['mean', 'std']}).reset_index()

for exp_id in final_agg['experiment_id'].unique():
    exp_data = final_agg[final_agg['experiment_id'] == exp_id]
    
    # Extracting mean and std deviation
    mean_reward = exp_data['mean', 'mean']
    std_dev = exp_data['mean', 'std']
    chunks = exp_data['chunk']

    # outdir = Path('cm_analysis_test')
    # outdir.mkdir(exist_ok=True, parents=True)
    # outpath = Path(outdir, f"{exp_id.replace(' ', '-')}_chunksize_{chunk_size}_ci.tsv")
    
    # with open(outpath, 'w+') as f:
    #     f.write('time'+'\t'+'high_ci_0.05'+'\t'+'low_ci_0.05'+'\t'+'mean'+'\n')
    #     # f.write('1e-10'+'\t'+'0.0'+'\t'+'0.0'+'\t'+'0.0'+'\n')
    #     for i, m, s in zip(chunks, mean_reward, std_dev):
    #         low = m-s
    #         high = m+s
    #         f.write(f"{i}\t{low}\t{high}\t{m}\n")


    # Plotting the mean reward
    plt.plot(chunks, mean_reward, label=f'Experiment {exp_id}')

    # Adding the std deviation as a shaded area
    plt.fill_between(chunks, mean_reward - std_dev, mean_reward + std_dev, alpha=0.3)

# plt.xscale('log')  # For logarithmic x-axis
# plt.yscale('log')  # For logarithmic y-axis (uncomment this line if needed)

plt.xlabel('Step (in chunks of {})'.format(chunk_size))
plt.ylabel('Average Bitscore')
plt.title('Mean Bitscore with Standard Deviation as Confidence Bounds')
plt.legend()
plt.show()