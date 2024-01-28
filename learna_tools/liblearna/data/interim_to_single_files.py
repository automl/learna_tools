import pandas as pd
import numpy as np
from pathlib import Path

def split_local_data_to_single_files(data_dir, extension='.rna'):
    target_file_paths = Path(data_dir, dataset).glob('*.local')
    for target_file_path in target_file_paths:
        [Path(data_dir, f"{index+1}{extension}").resolve().write_text(target.rstrip('\n')) for index, target in enumerate(target_file_path.read_text().rstrip('\n').split('\n'))]

def split_via_dataframe(data_dir, dataset, in_extension='.interim', out_extension='.rna'):
    path = Path(data_dir, dataset, f"{str(dataset).split('_')[-1]}{in_extension}").resolve()
    print('Read interim data')
    df = pd.read_csv(path, sep='\t')
    print('Write single files')
    # dot_brackets = []
    # sequence_constraints = []
    # for target in df['local_random']:
    #     dot_brackets.append(target.upper().replace('A', 'N').replace('G', 'N').replace('C', 'N').replace('U', 'N'))
    #     sequence_constraints.append(target.upper().replace('.', 'N').replace('(', 'N').replace(')', 'N'))
    # [pd.DataFrame([(current_index +1, row[0], row[1], row[2], row[3], row[4], row[5])]).to_csv(path_or_buf=Path(data_dir, dataset, f"{current_index+1}{out_extension}"), sep='\t', index=False) for current_index, row in enumerate(zip(df['structure'], df['sequence'], df['local_random'], df['local_motif'], df['gc_content'], df['mfe']))]
    [Path(data_dir, dataset, f"{index+1}{out_extension}").resolve().write_text(f"{index+1}\t{row[0]}\t{row[1]}\t{row[2]}\t{row[3]}\t{row[4]}\t{row[5]}") for index, row in enumerate(zip(df['structure'], df['sequence'], df['local_random'], df['local_motif'], df['gc_content'], df['mfe']))]
    # [Path(data_dir, dataset, f"{index+1}{out_extension}").resolve().write_text(f"{index+1}\t{row[0]}\t{row[1]}\t{row[2]}\t{row[3]}") for index, row in enumerate(zip(dot_brackets, sequence_constraints, df['gc_content'], df['mfe']))]

def split_rfam_anta(data_dir, dataset, in_extension='.interim', out_extension='.rna'):
    path = Path(data_dir, dataset, f"{str(dataset)}{in_extension}").resolve()
    print('Write single files')
    with open(path) as f:
        for l in f:
            id = l.split()[0].rstrip()
            omega = l.split()[1].rstrip()
            psi = l.split()[2].rstrip().upper().replace('T', 'U')
            gc = id
            mfe = id
            Path(data_dir, dataset, f"{id}{out_extension}").resolve().write_text(f"{id}\t{omega}\t{psi}\t{gc}\t{mfe}")

def generate_sc_data(data_dir, dataset, in_extension='.interim', out_extension='.rna'):
    path = Path(data_dir, dataset, f"{str(dataset).split('_')[-1]}{in_extension}").resolve()
    print('Read interim data')
    df = pd.read_csv(path, sep='\t')
    print('Write single files')
    df['local_random'] = df['local_random'].apply(lambda x: x.upper().replace(')', 'N').replace('(', 'N').replace('.', 'N'))
    # print(df)
    # print(df['local_random'])
    out_path = Path(data_dir, dataset + '_sc')
    out_path.mkdir(exist_ok=True, parents=True)
    [Path(out_path,  f"{index+1}{out_extension}").resolve().write_text(f"{index+1}\t{row[0]}\t{row[1]}\t{row[2]}\t{row[3]}") for index, row in enumerate(zip(df['structure'], df['local_random'], df['gc_content'], df['mfe']))]

def generate_inverse_folding_data(data_dir, dataset, in_extension='.interim', out_extension='.rna'):
    path = Path(data_dir, dataset, f"{str(dataset).split('_')[-1]}{in_extension}").resolve()
    print('Read interim data')
    df = pd.read_csv(path, sep='\t')
    print('Write single files')
    df['sequence'] = df['sequence'].apply(lambda x: x.upper().replace('A', 'N').replace('G', 'N').replace('C', 'N').replace('U', 'N'))
    # print(df)
    print(df['sequence'])
    out_path = Path(data_dir, dataset + '_if')
    out_path.mkdir(exist_ok=True, parents=True)
    [Path(out_path,  f"{index+1}{out_extension}").resolve().write_text(f"{index+1}\t{row[0]}\t{row[1]}\t{row[2]}\t{row[3]}") for index, row in enumerate(zip(df['structure'], df['sequence'], df['gc_content'], df['mfe']))]

def generate_if_baseline_data(data_dir, dataset, in_extension='.interim', out_extension='.rna'):
    path = Path(data_dir, dataset, f"{str(dataset).split('_')[-1]}{in_extension}").resolve()
    print('Read interim data')
    df = pd.read_csv(path, sep='\t')
    print('Write single files')
    out_path = Path(data_dir, dataset + '_bl')
    out_path.mkdir(exist_ok=True, parents=True)
    [Path(out_path,  f"{index+1}{out_extension}").resolve().write_text(f"{row}") for index, row in enumerate(df['structure'])]

def generate_data_4_anta_sc_run(data_dir, dataset, in_extension='.interim', out_extension='.anta'):
    path = Path(data_dir, dataset, f"{str(dataset)}{in_extension}").resolve()
    print('Write single files')
    with open(path) as f:
        for l in f:
            id = l.split()[0].rstrip()
            omega = l.split()[1].rstrip()
            psi = l.split()[2].rstrip().upper().replace('T', 'U')
            gc = id
            mfe = id
            out_path = Path(data_dir, dataset + '_anta')
            out_path.mkdir(exist_ok=True, parents=True)
            Path(out_path, f"{id}{out_extension}").resolve().write_text(f"{omega}\t{psi}")

def generate_gap_data(data_dir, dataset, in_extension='.interim', out_extension='.rna'):
    path = Path(data_dir, dataset, f"{str(dataset).split('_')[-1]}{in_extension}").resolve()
    print('Read interim data')
    df = pd.read_csv(path, sep='\t')
    print('Introducing gaps')
    targets = []
    for tau in df['local_random']:
        tau_list = list(tau)
        iterations = np.random.randint(1, 10)
        current_index = 0
        for iteration in range(iterations):
            try:
                length = np.random.randint(1, int(0.1 * len(tau[current_index:])))
                start = np.random.randint(current_index + 1, len(tau) - length)
                tau_list[start:start + length] = 'N' * length
                current_index = start + 1
            except:
                break
        assert len(''.join(tau_list)) == len(tau)
        targets.append(''.join(tau_list))

    df['local_random'] = pd.DataFrame(targets, columns=['local_random'])

    out_dir = Path(data_dir, dataset + '_gaps')
    out_dir.mkdir(exist_ok=True, parents=True)

    print('Write single files')
    [Path(out_dir, f"{index+1}{out_extension}").resolve().write_text(f"{index+1}\t{row[0]}\t{row[1]}\t{row[2]}\t{row[3]}\t{row[4]}\t{row[5]}") for index, row in enumerate(zip(df['structure'], df['sequence'], df['local_random'], df['local_motif'], df['gc_content'], df['mfe']))]




if __name__ == '__main__':
    # split_local_data_to_single_files('data/rfam_learn_local/test')
    # split_local_data_to_single_files('data/rfam_learn_local/train')
    # split_local_data_to_single_files('data/rfam_learn_local/validation')
    # print("split test data")
    # split_via_dataframe('data/', 'rfam_local_short_train')
    # print('split training data')
    # split_via_dataframe('data/', 'rfam_local_long_train')
    # print('split validation data')
    # split_via_dataframe('data/rfam_learn_local', 'validation')
    # split_via_dataframe('data', 'eterna_local_test')
    # split_via_dataframe('data', 'rfam_taneda_local_test')
    # split_via_dataframe('data', 'rfam_local_test')
    # split_via_dataframe('data', 'rfam_local_min_400_max_1000_test')
    # split_via_dataframe('data', 'rfam_local_min_1000_test')
    # split_rfam_anta('data', 'rfam_anta_sc')
    # generate_sc_data('data', 'rfam_local_min_1000_test')
    # generate_inverse_folding_data('data', 'rfam_local_min_1000_test')
    # generate_if_baseline_data('data', 'rfam_local_min_1000_test')
    # generate_data_4_anta_sc_run('data', 'rfam_anta_sc')
	generate_gap_data('data', 'rfam_local_min_1000_test')
