from pathlib import Path

datasets = ['rfam_local_test', 'rfam_local_validation', 'rfam_local_train', 'rfam_local_min_400_max_1000_test', 'rfam_local_min_1000_test', 'rfam_local_short_train', 'rfam_local_long_train', 'rfam_learn_local_min_500_test', 'rfam_learn_local_min_100_max_500_test']

def get_dataset(data_dir, dataset):
    return Path(data_dir, dataset).glob('*.rna')

def validate(datasets):
    print('Validating all datasets')
    valid = [dataset.intersection(others) for dataset, others in zip(datasets.values(), datasets.values()) if not others == dataset]
    # if not all(valid
    validation_result = 'There are no intersections between datasets' if not valid else f"Datasets share targets: \n {valid}"
    print(validation_result)


def get_datasets(data_dir):
    print('reading all datasets')
    data = {}
    for dataset in datasets:
        dataset_targets = set([target.read_text().split('\t')[3] for target in get_dataset(data_dir, dataset)])
        data[dataset] = dataset_targets
    return data

def check(data_dir):
    for dataset in datasets:
        for index, target in enumerate(get_dataset(data_dir, dataset)):
            try:
                target.read_text().split('\t')[3]
            except:
                print(dataset, index)
                print(target.read_text().split('\t'))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", type=str, default='data', help="The data dir"
    )

    args = parser.parse_args()

    validate(get_datasets(args.data_dir))
    # check(args.data_dir)
