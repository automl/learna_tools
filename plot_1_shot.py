import argparse
import pandas as pd

import matplotlib.pyplot as plt

from pathlib import Path
from distance import hamming

from RNA import fold


def plot_hamming_distances(df1, df2):
    """
    Plot hamming distances for each ID from two dataframes using different marker types.
    Connects the points with a green line if the first dataframe has a lower hamming distance,
    and a purple line if the second dataframe has a lower hamming distance.
    
    Parameters:
    - df1: DataFrame with columns 'Id' and 'Relative_Hamming'
    - df2: DataFrame with columns 'Id' and 'Relative_Hamming'
    """
    df1['Id'] = pd.to_numeric(df1['Id'])
    df2['Id'] = pd.to_numeric(df2['Id'])
    
    df1 = df1.sort_values(by='Id')
    df2 = df2.sort_values(by='Id')
    
    plt.figure(figsize=(10, 6))

    plt.scatter(df1['Id'], df1['Relative_Hamming'], marker='o', color='blue', label='libLEARNA')
    plt.scatter(df2['Id'], df2['Relative_Hamming'], marker='x', color='red', label='Meta-LEARNA')
    
    for id, hd1, hd2 in zip(df1['Id'], df1['Relative_Hamming'], df2['Relative_Hamming']):
        line_color = 'green' if hd1 < hd2 else 'purple'
        plt.plot([id, id], [hd1, hd2], color=line_color)
    
    plt.xlabel('Id')
    plt.ylabel('Normalized Hamming Distance')
    plt.title('1-Shot Performance')
    plt.legend()
    plt.show()

def raw_preds_to_df(directory):
    first = []
    for p in directory.glob('*.res'):
        seq = p.read_text().split('\n')[0].split()[-1]
        # meta-learna writes args in first two lines...
        if 'args' in seq:
            seq = p.read_text().split('\n')[2].split()[-1]
        structure = fold(seq)[0]
        id = p.stem
        first.append({'Id': id, 'structure': structure})
    return pd.DataFrame(first)

parser = argparse.ArgumentParser()

parser.add_argument('--liblearna_directory', type=str, help='Path to the directory containing the results')
parser.add_argument('--metalearna_directory', type=str, help='Path to the directory containing the results')
parser.add_argument('--read_raw', action='store_true', help='Read raw data')

args = parser.parse_args()

liblearna_dir = Path(args.liblearna_directory)
metalearna_dir = Path(args.metalearna_directory)

liblearna_preds = list(Path(liblearna_dir).glob('*.pkl'))
metalearna_preds = list(Path(metalearna_dir).glob('*.pkl'))

def get_first_preds(preds):
    return preds.iloc[0]

refs = [{'Id': p.read_text().split()[0], 'structure': p.read_text().split()[1]} for p in Path('data/eterna100_v2').glob('*.rna')]

refs_df = pd.DataFrame(refs)

if args.read_raw:
    metalearna_first = raw_preds_to_df(Path('1-shot-predictions-raw/meta-learna'))
    liblearna_first = raw_preds_to_df(Path('1-shot-predictions-raw/liblearna'))

    print(metalearna_first)
    print(liblearna_first)


    liblearna_rel_hammings = []

    for i, row in liblearna_first.iterrows():
        ref = refs_df[refs_df['Id'] == row['Id']]
        if len(ref) == 0:
            raise
        rel_hamm = hamming(ref['structure'].iloc[0], row['structure']) / len(row['structure'])
    
        ham = {'Id': row['Id'], 'Relative_Hamming': rel_hamm}
        liblearna_rel_hammings.append(ham)
    
    liblearna_results = pd.DataFrame(liblearna_rel_hammings)

    metalearna_rel_hammings = []

    for i, row in metalearna_first.iterrows():
        ref = refs_df[refs_df['Id'] == row['Id']]
        if len(ref) == 0:
            raise
        rel_hamm = hamming(ref['structure'].iloc[0], row['structure']) / len(row['structure'])
    
        ham = {'Id': row['Id'], 'Relative_Hamming': rel_hamm}
        metalearna_rel_hammings.append(ham)

    metalearna_results = pd.DataFrame(metalearna_rel_hammings)

    print(liblearna_results)
    print(metalearna_results)

else:

    dfs = []
    for p in liblearna_preds:
        preds = pd.read_pickle(p)
        first = get_first_preds(preds)
        dfs.append(first)
    liblearna_first = pd.DataFrame(dfs)
    
    print(liblearna_first)
    
    dfs = []
    for p in metalearna_preds:
        preds = pd.read_pickle(p)
        dfs.append(preds)
    meatalearna_first = pd.concat(dfs)

    print(meatalearna_first)

    refs = [{'Id': p.read_text().split()[0], 'structure': p.read_text().split()[1]} for p in Path('data/eterna100_v2').glob('*.rna')]
    
    refs_df = pd.DataFrame(refs)
    
    liblearna_rel_hammings = []
    for i, row in liblearna_first.iterrows():
        ref = refs_df[refs_df['Id'] == row['Id']]
        if len(ref) == 0:
            raise
        rel_hamm = hamming(ref['structure'].iloc[0], row['structure']) / len(row['structure'])
    
        ham = {'Id': row['Id'], 'Relative_Hamming': rel_hamm}
        liblearna_rel_hammings.append(ham)
    
    liblearna_results = pd.DataFrame(liblearna_rel_hammings)
    
    metalearna_results = meatalearna_first[['Id', 'rel_hamming_distance']]
    
    metalearna_results.rename(columns={"rel_hamming_distance": "Relative_Hamming"}, inplace=True)
    
    print(liblearna_results)
    print(metalearna_results)
    
plot_hamming_distances(liblearna_results, metalearna_results)
    
