from pathlib import Path

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        default="data/rfam_local_test",
        help="Path to data directory",
    )
    parser.add_argument(
        "--dataset",
        default="rfam_local_test",
        help="Path to data directory",
    )

    args = parser.parse_args()

    dataset_path = Path(args.data_path, args.dataset)
    # out_dir = 'data/rfam_local_test_anta/'
    # out_dir_path = Path(out_dir)
    # out_dir_path.mkdir(exist_ok=True, parents=True)
    for path in dataset_path.glob('*.rna'):
        local = path.read_text().split('\t')[3]
        struc = path.read_text().split('\t')[1]
        # print(struc)
        seq = local.replace('.', 'N').replace('(', 'N').replace(')', 'N')
        out_path = Path(dataset_path, f"{path.stem}.anta")
        out_path.write_text(f"{struc}\t{seq}")
