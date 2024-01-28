from pathlib import Path


def get_files(data_dir):
    return Path(data_dir).glob('*.rna')


def convert_structure(secondary):
    stack = []
    new_struc = []
    for index, value in enumerate(secondary):
        if value == '(':
            stack.append(index)
            new_struc.append(f"{value}{index}")
        elif value == ')':
            pairing_partner = stack.pop()
            new_struc.append(f"{value}{pairing_partner}")
        else:
            new_struc.append(value)
    return ''.join(new_struc)


def convert_dataset(data_dir, out_dir):
    files = get_files(data_dir)
    for f in files:
        id = f.read_text().split('\t')[0]
        struc = f.read_text().split('\t')[1]
        seq = f.read_text().split('\t')[2]
        gc = f.read_text().split('\t')[3]
        mfe = f.read_text().split('\t')[4]

        struc = convert_structure(struc)

        print(id)
        print(struc)
        print(seq)
        print(gc)
        print(mfe)
        print()

        out = Path(out_dir)
        out.mkdir(exist_ok=True, parents=True)
        with open(Path(out, f"{id}.rna"), 'w+') as out_file:
            out_file.write(id+'\t'+struc+'\t'+seq+'\t'+gc+'\t'+mfe)


if __name__ == '__main__':
    data_dir = 'data/rfam_LD_balanced'
    out_dir = 'data/rfam_LD_balanced_learna'

    convert_dataset(data_dir, out_dir)
