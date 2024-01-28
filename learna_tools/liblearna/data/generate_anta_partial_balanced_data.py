from pathlib import Path
import numpy as np

def _encode_pairing(secondary):
    """
    Encode the base pairing scheme of the design task.

    Args:
        secondary: the secondary structure (domain) that has to get encoded.

    Returns:
        List that contains the paired site for each site if exists, or None otherwise.
    """
    pairing_encoding = [None] * len(secondary)
    stack = []
    for index, symbol in enumerate(secondary, 0):
        if symbol == "(":
            stack.append(index)
        elif symbol == ")":
            if not stack:
                continue
            else:
                paired_site = stack.pop()
                pairing_encoding[paired_site] = index
                pairing_encoding[index] = paired_site
    return pairing_encoding

def random_insert_n(string):
    string_list = list(string)

    if string_list[0] in ['A', 'C', 'G', 'U']:
        insert_times = np.random.randint(1, 8)
    else:
        insert_times = np.random.randint(0, 3)

    percentage = 0.3
    length_change = np.random.random_sample((insert_times, )) * percentage
    inserts = ['N' * (int(len(string) * l)) for l in length_change]
    if insert_times != 0:
        max_insert_position = len(string) - np.max([(int(len(string) * l)) for l in length_change])
        insert_positions = np.random.choice(max_insert_position, insert_times)
        for insert, position in zip(inserts, insert_positions):
            string_list[position:position + len(insert)] = insert

    return ''.join(string_list)


def remove_unbalanced_brackets(secondary, mode):
    secondary_list = list(secondary.upper())
    stack = []
    for index, symbol in enumerate(secondary, 0):
        if symbol == "(":
            stack.append(index)
        elif symbol == ")":
            if not stack:
                secondary_list[index] = 'N'
            else:
                stack.pop()
        elif symbol in ['A', 'C', 'G', 'U']:
            if mode == 'anta':
                secondary_list[index] = 'N'
    for index in stack:
        secondary_list[index] = 'N'
    return ''.join(secondary_list)

def extend_unbalanced2balanced(secondary, pairing_encoding, mode):
    # unbalanced_closing = []
    # stack = []
    # for index, symbol in enumerate(secondary, 0):
    #     if symbol == "(":
    #         stack.append(index)
    #     elif symbol == ")":
    #         if not stack:
    #             unbalanced_closing.append(index)
    #             # secondary_list[index] = 'N'
    #         else:
    #             stack.pop()
    #     elif symbol in ['A', 'C', 'G', 'U']:
    #         secondary_list[index] = 'N'
    # for index in stack:
    #     secondary_list[pairing_encoding[index]] = ')'
    # for index in unbalanced_closing:
    #     secondary_list[pairing_encoding[index]] = '('
    secondary_list = list(secondary.upper())
    for index, symbol in enumerate(secondary_list, 0):
        if symbol == "(":
            secondary_list[pairing_encoding[index]] = ')'
        elif symbol == ")":
            secondary_list[pairing_encoding[index]] = '('
        elif symbol in ['A', 'C', 'G', 'U']:
            if mode == 'anta':
                secondary_list[index] = 'N'
    return ''.join(secondary_list)

def verify_balance(secondary):
    secondary_list = list(secondary.upper())
    stack = []
    for index, symbol in enumerate(secondary, 0):
        if symbol == "(":
            stack.append(index)
        elif symbol == ")":
            if not stack:
                return False
            else:
                stack.pop()
        # elif symbol in ['A', 'C', 'G', 'U']:
        #     secondary_list[index] = 'N'
    # for index in stack:
    #     secondary_list[index] = 'N'
    return not bool(stack)

def get_unbalanced_brackets(secondary):
    secondary_list = list(secondary.upper())
    unbalanced_closing = []
    stack = []
    for index, symbol in enumerate(secondary, 0):
        if symbol == "(":
            stack.append(index)
        elif symbol == ")":
            if not stack:
                unbalanced_closing.append(index)
            else:
                stack.pop()
        # elif symbol in ['A', 'C', 'G', 'U']:
        #     secondary_list[index] = 'N'
    # for index in stack:
    #     secondary_list[index] = 'N'
    for index in stack:
        secondary_list[index] = 'L'
    for index in unbalanced_closing:
        secondary_list[index] = 'R'
    return ''.join(secondary_list)

def make_local_data(secondary, primary):
    primary_list = list(primary)
    for index, symbol in enumerate(secondary, 0):
        if symbol in ['.', '(', ')']:
            primary_list[index] = 'N'
    return ''.join(primary_list)



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
    # counter = 0
    # length = []
    strucs = []
    counter = 0
    for path in dataset_path.glob('*.rna'):
        # id = path.read_text().split('\t')[0]
        # struc = path.read_text().split('\t')[1]
        # seq = path.read_text().split('\t')[2]
        # local = path.read_text().split('\t')[3]
        # gc = path.read_text().split('\t')[5]
        # mfe = path.read_text().split('\t')[6]
        # print(path.read_text().split())
        seq = path.read_text().split()[2].strip()
        struc = path.read_text().split()[1].strip()
        gc = path.read_text().split()[3].strip()
        id = path.stem
        # gc = path.stem
        # mfe = path.stem
        # seq = 'N' * len(struc)

        # print(id)
        # struc = local.replace('A', 'N').replace('C', 'N').replace('G', 'N').replace('U', 'N')
        # seq = local.replace('.', 'N').replace('(', 'N').replace(')', 'N')
        print(struc)
        print(seq)
        print(gc)
        # print(gc)
        # print(mfe)
        # print()

        out_dir = Path('archiveII_thirdparty')
        out_dir.mkdir(exist_ok=True, parents=True)

        out = Path(out_dir, f"{id}.rna")

        with open(out, 'w+') as f:
            f.write(struc + '\t' + seq + '\t' + gc)


        # motif_based = path.read_text().split('\t')[3]
        # gc = path.read_text().split('\t')[4]
        # assert len(seq) == len(struc)

        # length.append(len(seq))
        # strucs.append(struc)

        # if int(path.stem)%3 == 0:
        #     struc_c = random_insert_n(struc)
        #     seq_c = random_insert_n(seq)
        #     while seq_c.count('N') / len(seq_c) < 0.4:
        #         seq_c = random_insert_n(seq_c)
        #     # print(path.stem)
        #     # print(struc_c)
        #     # print(seq_c)
        # else:
        #     struc_c = local.replace('A', 'N').replace('C', 'N').replace('G', 'N').replace('U', 'N')
        #     seq_c = local.replace('.', 'N').replace('(', 'N').replace(')', 'N')
        #     # print(path.stem)
        #     # print(struc_c)
        #     # print(seq_c)
        # if not 'N' in seq_c:
        #     seq_c = random_insert_n(seq_c)
        # print(path.stem)
        # print(struc_c)
        # print(seq_c)
        # out_path = Path(args.data_path + '/' + args.dataset + '_new/', f"{path.stem}.rna")
        # out_path.parent.mkdir(exist_ok=True)
        # out_path.write_text(f"{path.stem}\t{struc_c}\t{seq_c}\t{gc}\t{mfe}")



        # seq_i = random_insert_n(seq)
        # struc_i = random_insert_n(struc)
        # if struc_i[-1] == 'N' or seq_i[-1] == 'N':
        #     print(seq_i)
        #     print(struc_i)




        # counter += 1
        # if counter == 20:
        #     break
        # counter += 1
    # avg_length = length / counter
    # length = np.asarray(length)
    # strucs = np.asarray(strucs)
    # length = np.array(list(map(len, strucs)))
    # avg_length = np.mean(length)

    # print(f"average length: {avg_length}")
    # print(f"max length: {np.max(length)}")
    # print(f"min length: {np.min(length)}")
    # print(strucs[np.where(length == 104)])
    # print('N' * 104)




        # print(struc)
        # balanced_struc_remove = remove_unbalanced_brackets(local, 'anta')
        # balanced_struc_remove_learna = remove_unbalanced_brackets(local, 'learna')
        # pairing_encoding = _encode_pairing(struc)
        # balanced_struc_add = extend_unbalanced2balanced(local, pairing_encoding, 'anta')
        # balanced_struc_add_learna = extend_unbalanced2balanced(local, pairing_encoding, 'learna')

        # print(local)
        # print(balanced_struc_remove)
        # print(balanced_struc_add)
        # print(balanced_struc_remove_learna)
        # print(balanced_struc_add_learna)
        # print(struc)
        # seq_c = local.replace('.', 'N').replace('(', 'N').replace(')', 'N')
        # seq_c = make_local_data(balanced_struc_add, seq_c)
        # print('\n')
        # print(balanced_struc_remove_learna)
        # print(balanced_struc_remove)
        # print(balanced_struc_add)
        # print(struc)
        # print(seq)
        # print(seq_c)
        # print(local)
        # print(gc)
        # if not verify_balance(balanced_struc_add) or not verify_balance(balanced_struc_remove):
        #     print(get_unbalanced_brackets(balanced_struc_add))
        #     print(get_unbalanced_brackets(balanced_struc_remove))
        # print(seq_c)
        # print(verify_balance(balanced_struc_add), verify_balance(balanced_struc_remove))
        # out_path_anta = Path(args.data_path + '/' + args.dataset + '_balanced/', f"{path.stem}.anta")
        # out_path_learna = Path(args.data_path + '/' + args.dataset + '_balanced/', f"{path.stem}.rna")
        # out_path_anta.parent.mkdir(exist_ok=True)
        # out_path_learna.parent.mkdir(exist_ok=True)
        # print(out_path_anta.resolve())

        # out_path_anta_add = Path( + '_balanced', f"{path.stem}.anta")
        #out_path_anta.write_text(f"{balanced_struc_add}\t{seq_c}")
        #out_path_learna.write_text(f"{path.stem}\t{balanced_struc_add}\t{seq_c}\t{path.stem}\t{path.stem}")

        # out_path_anta_remove = Path(dataset_path + '_balanced_remove', f"{path.stem}.anta")
        # out_path_anta_remove.write_text(f"{balanced_struc_remove}\t{seq_c}")


        # out_path_learna_add = Path(dataset_path + '_balanced_add', f"{path.stem}.anta")
        # out_path_learna_add.write_text(f"{balanced_struc_add}\t{seq}")

        # out_path_learna_remove = Path(dataset_path + '_balanced_remove', f"{path.stem}.anta")
        # out_path_learna_remove.write_text(f"{path.stem}\t{struc}\t{seq}{balanced_struc_remove}\t{seq}")
