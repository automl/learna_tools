import time
from pathlib import Path
from itertools import islice
from RNA import fold


class FamilyObject(object):
    def __init__(self, family_path):
        print(f"Create family object of rfam family {family_path.stem}")
        self._path = family_path
        self._id = family_path.stem
        self._content = family_path.read_text().splitlines()

    def __iter__(self):
        targets = self.targets
        return iter(targets)

    def _get_targets(self):
        return [(self.id, len(sequence), sequence.upper().replace('T', 'U'), fold(sequence), ((sequence.upper().count('G') + sequence.upper().count('C')) / len(sequence))) for sequence in islice(self._content, 1, len(self._content), 2)]

    @property
    def id(self):
        return self._id

    @property
    def targets(self):
        return self._get_targets()


def read_family_targets(database_dir, family):
    if family:
        try:
            return [FamilyObject(Path(database_dir, f"RF{family}.fa"))]
        except:
            print(f"family {family} doesn't exist")
    families = [FamilyObject(family_path) for family_path in Path(database_dir).glob('*.fa')]
    return iter(families)

def database_to_interim(database_dir, family, out_dir):
    out_path = Path(out_dir, 'rfam_database.tsv')
    write_database(database_dir, family, out_path)

def write_database(database_dir, family, out_path):
    print("Start writing database file")
    if family:
        out_path = Path(out_path.parent, f"targets_{family}.tsv")
    database_file = open(out_path, "a+")
    database_file.write("family\tlength\tsequence\tstructure_and_mfe\tgc_content\n")
    for family in read_family_targets(database_dir, family):
        print(f"Writing targets of family {family.id} into database")
        for target in family:
            database_file.write(f"{target[0]}\t{target[1]}\t{target[2]}\t{target[3]}\t{target[4]}\n")
    database_file.close()
    print(f"Database written to {out_path.resolve()}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", default="data/rfam_raw/", type=Path, help="Output path")
    parser.add_argument("--out_dir", default="data/test/", help="Output path")
    parser.add_argument(
        "--family",
        default=None,
        required=False,
        type=str,
        help="Rfam family id to use for dataset. Requires all ",
    )

    args = parser.parse_args()

    out_dir = args.out_dir
    data = args.data
    family = ("0" * (5 - len(args.family))) + args.family


    start = time.time()
    database_to_interim(data, family, out_dir)
    end = time.time()
    print(f"dataset generated in {end - start} seconds")

    # print(pd.read_csv(Path(out_dir, 'rfam_database.tsv'), sep='\t'))
