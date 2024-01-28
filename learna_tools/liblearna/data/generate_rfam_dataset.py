import pandas as pd
import numpy as np
import string
from pathlib import Path
from ..learna.environment import _encode_pairing

class DatasetError(Exception):
    pass

def tsv_to_df(path):
    try:
        return pd.read_csv(Path(path), sep='\t')
    except:
        print(f"error with file {path}")

def drop_bad_sequences(df):
    valid_rows = [row for row in df.itertuples() if set(row.sequence).issubset('ACGU') and set(row.structure).issubset('.()')]
    print(f"{len(valid_rows)} valid targets of {df['sequence'].size} -- drop {df['sequence'].size - len(valid_rows)}")
    data = pd.DataFrame(valid_rows)
    return data

def validate_datasets(datasets):
    validation = []
    for dataset in datasets:
        validation.append(all([set(sequence).issubset('AGCU') for sequence in dataset['sequence']]))
    return all(validation)

def validate_structure_length(df):
    return all([x == len(y) for x, y in zip(df['length'], df['structure'])])

def split_structure_and_mfe(df):
    struc_and_mfe = df.structure_and_mfe.values.tolist()
    struc_and_mfe = [(str(x.strip("[]").split(', ')[0].strip("''")), float(x.strip("[]").split(', ')[1])) for x in struc_and_mfe]
    df[['structure', 'mfe']] = pd.DataFrame(struc_and_mfe, index=df.index)
    return df[['family', 'length', 'sequence', 'structure', 'mfe', 'gc_content']]

def remove_duplicates(df):
    return df.drop_duplicates(subset=['structure'], keep=False)

def satisfy_length_criteria(df, min, max):
    return df.loc[(df['length'] >= min) & (df['length'] <= max)]

def generate_datasets(df, size, train_multiplier, validation_multiplier):
    samples = size + (size * train_multiplier) + (size * validation_multiplier)
    if samples > df.size:
        raise DatasetError(f"Not enough entries in the database to create disjoint datasets\n database entries: {df.size}\n desired dataset size: {samples}")

    data = df.sample(n=samples)
    test_set = data.iloc[:size]
    training_set = data.iloc[size: size + (size * train_multiplier)]
    validation_set = data.iloc[size + (size * train_multiplier):]

    return [test_set, training_set, validation_set]

def generate_local_random(df, max_replacements, replace_quantile, suffix_size):
    local_structures = []

    for structure, sequence in zip(df['structure'], df['sequence']):
        suf_size = np.random.randint(1, suffix_size)
        replacements = np.random.randint(1, max_replacements)

        struc = [s for s in structure[:-suf_size]]

        current_site = 0
        for i in range(replacements):
            try:
                current_replacement_site = np.random.randint(1, int(replace_quantile * len(structure)))

                struc[current_site:current_site + current_replacement_site] = sequence[current_site:current_site + current_replacement_site]
                struc = ''.join(struc)
                struc = [c for c in struc]

                step = np.random.randint(1, int(replace_quantile * len(structure)))
                current_site += current_replacement_site + step

            except Exception as e:
                print('**********************************************************************')
                print(e)
                raise DatasetError(f"Error with structure {structure} and sequence {sequence}")
        struc = struc + sequence[len(struc):].split()
        struc = ''.join(struc)
        local_structures.append(struc)

    df['local_random'] = local_structures
    return df

def extract_motifs(df):
    print('Extract motifs from data')
    motifs = []
    for structure, sequence in zip(df['structure'], df['sequence']):
        pairing_encoding = _encode_pairing(structure)
        structure_motifs = [(structure[index:pair+1], sequence[index:pair+1], index, pair) for index, pair in enumerate(pairing_encoding) if pair and structure[index:pair+1]]
        motifs.append(structure_motifs)
    df['motifs'] = motifs
    return df


def generate_local_motif_based(df):
    print('Generate motif based random local structures')
    motif_based_local_structures = []
    for sequence, motifs in zip(df['sequence'], df['motifs']):
        try:
            motif = motifs[np.random.randint(0, len(motifs) - 2)]
            seq = sequence[:motif[2]] + motif[0] + sequence[motif[3] + 1:]
            motif_based_local_structures.append(seq)
        except:
            try:
                if len(motifs) > 2:
                    motif = motifs[np.random.randint(0, len(motifs) - 2)]
                    seq = sequence[:motif[2]] + motif[0] + sequence[motif[3] + 1:]
                    motif_based_local_structures.append(seq)
                else:
                    motif_based_local_structures.append(sequence)
                    print("Not enough motifs")
                    print("Use original sequence instead")

            except Exception as e:
                print('********************************************************************************')
                print(f"Please try again")

                raise DatasetError('Please restart.')

    df['local_motif'] = motif_based_local_structures
    return df

def datasets_to_tsv(datasets, out_dir, name, args):
    test_path = Path(out_dir, f"{name}_test")
    train_path = Path(out_dir, f"{name}_train")
    validation_path = Path(out_dir, f"{name}_validation")

    test_path.mkdir(parents=True, exist_ok=True)
    train_path.mkdir(parents=True, exist_ok=True)
    validation_path.mkdir(parents=True, exist_ok=True)

    print('Write test set data')
    datasets[0].to_csv(path_or_buf=Path(out_dir, f"{name}_test", "test.interim"), sep='\t', index=False)
    datasets[0]['structure'].to_csv(Path(out_dir, f"{name}_test", "test.rna"), sep='\t', index=False)
    datasets[0]['gc_content'].to_csv(Path(out_dir, f"{name}_test", "test.gc"), sep='\t', index=False)
    datasets[0]['mfe'].to_csv(Path(out_dir, f"{name}_test", "test.mfe"), sep='\t', index=False)
    datasets[0]['sequence'].to_csv(Path(out_dir, f"{name}_test", "test.seq"), sep='\t', index=False)

    if args.local_random:
        datasets[0]['local_random'].to_csv(Path(out_dir, f"{name}_test", "test.local"), sep='\t', index=False)
        datasets[0]['local_motif'].to_csv(Path(out_dir, f"{name}_test", "test.motif"), sep='\t', index=False)
    if args.motifs:
        datasets[0]['motifs'].to_csv(Path(out_dir, f"{name}_test", "test.motifs"), sep='\t', index=False)

    print('Write training set data')
    datasets[1].to_csv(path_or_buf=Path(out_dir, f"{name}_train", "train.interim"), sep='\t', index=False)
    datasets[1]['structure'].to_csv(Path(out_dir, f"{name}_train", "train.rna"), sep='\t', index=False)
    datasets[1]['mfe'].to_csv(Path(out_dir, f"{name}_train", "train.mfe"), sep='\t', index=False)
    datasets[1]['sequence'].to_csv(Path(out_dir, f"{name}_train", "train.seq"), sep='\t', index=False)
    datasets[1]['gc_content'].to_csv(Path(out_dir, f"{name}_train", "train.gc"), sep='\t', index=False)
    if args.local_random:
        datasets[1]['local_random'].to_csv(Path(out_dir, f"{name}_train", "train.local"), sep='\t', index=False)
        datasets[1]['local_motif'].to_csv(Path(out_dir, f"{name}_train", "train.motif"), sep='\t', index=False)
    if args.motifs:
        datasets[1]['motifs'].to_csv(Path(out_dir, f"{name}_train", "train.motifs"), sep='\t', index=False)

    print('Write validation set data')
    datasets[2].to_csv(path_or_buf=Path(out_dir, f"{name}_validation", "validation.interim"), sep='\t', index=False)
    datasets[2]['structure'].to_csv(Path(out_dir, f"{name}_validation", "validation.rna"), sep='\t', index=False)
    datasets[2]['mfe'].to_csv(Path(out_dir, f"{name}_validation", "validation.mfe"), sep='\t', index=False)
    datasets[2]['sequence'].to_csv(Path(out_dir, f"{name}_validation", "validation.seq"), sep='\t', index=False)
    datasets[2]['gc_content'].to_csv(Path(out_dir, f"{name}_validation", "validation.gc"), sep='\t', index=False)
    if args.local_random:
        datasets[2]['local_random'].to_csv(Path(out_dir, f"{name}_validation", "validation.local"), sep='\t', index=False)
        datasets[2]['local_motif'].to_csv(Path(out_dir, f"{name}_validation", "validation.motif"), sep='\t', index=False)
    if args.motifs:
        datasets[2]['motifs'].to_csv(Path(out_dir, f"{name}_validation", "validation.motifs"), sep='\t', index=False)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument(
        "--tsv_path",
        default="data/rfam_interim",
        help="Path to interim tsv",
    )
    parser.add_argument("--out_dir", default="data/rfam_processed", help="Output path")

    # Dataset specification
    parser.add_argument(
        "--family",
        default=None,
        required=False,
        type=str,
        nargs="+",
        help="List of Rfam family ids to use for dataset. Format: <int> id of family, e.g., 1 3 180",
    )

    parser.add_argument(
        "--name", default="rfam_learn_local", required=False, type=str, help="Dataset name"
    )

    # Hyperparameters
    parser.add_argument("--size", type=int, default=100, help="The size of the test set")
    parser.add_argument(
        "--train_multiplier",
        required=False,
        type=int,
        default=1000,
        help="Trainig set size <multiplier> times <size> ",
    )
    parser.add_argument(
        "--validation_multiplier",
        required=False,
        type=int,
        default=1,
        help="Validation set size <multiplier> times <size> ",
    )

    parser.add_argument(
        "--minimum_length",
        required=False,
        type=int,
        default=None,
        help="The minimum sequence length of the targets"
    )

    parser.add_argument(
        "--maximum_length",
        required=False,
        type=int,
        default=None,
        help="The maximum sequence length of the targets"
    )

    parser.add_argument(
        "--unique",
        action="store_true",
        help="Only use unique structures for dataset",
    )

    parser.add_argument(
        "--local_random",
        action="store_true",
        help="Create dataset for local RNA Design",
    )

    parser.add_argument(
        "--motifs",
        action="store_true",
        help="Create motif based sequence-structure targets",
    )


    parser.add_argument(
        "--maximum_replacements",
        required=False,
        type=int,
        default=5,
        help="The number of replacements for random local sequences"
    )

    parser.add_argument(
        "--suffix_size",
        required=False,
        type=int,
        default=20,
        help="The number of replacements for random local sequences"
    )


    parser.add_argument(
        "--replace_quantile",
        required=False,
        type=float,
        default=0.3,
        help="The percentage of the structure that can be replaced with sequence for each replacement"
    )

    args = parser.parse_args()

    print(f"Read data from {args.tsv_path}")

    dfs = []

    # if certain families were chosen, look only at these, otherwise use all families
    if args.family:
        print("Satisfy family criteria")
        for family in args.family:
            id = ("0" * (5 - len(family))) + family
            dfs.append(tsv_to_df(Path(args.tsv_path, 'targets_' + id + '.tsv')))
    else:
        print('No family criteria provided -- Read all data')
        for path in Path(args.tsv_path).glob('*.tsv'):
            dfs.append(tsv_to_df(path))

    df = pd.concat(dfs)
    print(f"whole data : {df['sequence'].size}")

    # remove structures that aren't in the chosen length interval
    if args.maximum_length or args.minimum_length:
        print("Satisfy length criteria")
        min = 0 if args.minimum_length is None else args.minimum_length
        max = 100000000 if args.maximum_length is None else args.maximum_length
        df = satisfy_length_criteria(df, min, max)

    # split structure and mfe data into two columns (provided as list from RNAfold)
    df = split_structure_and_mfe(df)

    # only use unique sequences if desired by user
    if args.unique:
        print("Use unique structures only")
        df = remove_duplicates(df)

    print(f"Unique structures: {df['sequence'].size}")

    # print('Validate data')
    # assert validate_structure_length(df)
    # df = drop_bad_sequences(df)

    # print('Generate datasets')
    # print(f"test set size: {args.size}")
    # print(f"training set size: {args.size * args.train_multiplier}")
    # print(f"validation set size: {args.size * args.validation_multiplier}")
    # datasets = generate_datasets(df, args.size, args.train_multiplier, args.validation_multiplier)
    # iteration = 0
    # while not validate_datasets(datasets):
    #     iteration += 1
    #     print(f"datasets not valid in iteration {iteration}")
    #     datasets = generate_datasets(df, args.size, args.train_multiplier, args.validation_multiplier)

    # # generate randomly generated local design structures if desired
    # if args.local_random:
    #     print("Generate random structures for local design")
    #     datasets = [generate_local_random(dataset, args.maximum_replacements, args.replace_quantile, args.suffix_size) for dataset in datasets]

    # if args.motifs:
    #     # extract motifs from structures and generate local data by inserting structural motifs into sequence
    #     print('Generate motif based structures for local design ')
    #     datasets = [generate_local_motif_based(extract_motifs(dataset)) for dataset in datasets]

    # print(f"Write data to {args.out_dir}")
    # name = args.name if args.name else 'rfam_learn_local'
    # datasets_to_tsv(datasets, args.out_dir, name=name, args=args)
