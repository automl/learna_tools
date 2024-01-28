from pathlib import Path
import pandas as pd

def read_data(path):
    return path.glob('*.rna')

def generate_local_data(data):
    data = [(path.stem, path.read_text().rstrip()) for path in data]
    df = pd.DataFrame(data, columns=['id', 'structure'])
    df['sequence'] = df['structure']
    df['local_random'] = df['structure']
    df['local_motif'] = df['structure']
    df['gc_content'] = df['id']
    df['mfe'] = df['id']
    return df

def write_data(df, out_path):
    out_path.mkdir(parents=True, exist_ok=True)
    df.to_csv(path_or_buf=Path(out_path, 'test.interim'), index=False, sep='\t')

if __name__ == '__main__':
    path = Path('data', 'eterna')
    out_path = Path('data', 'eterna_local_test')

    data = read_data(path)

    df = generate_local_data(data)

    write_data(df, out_path)
