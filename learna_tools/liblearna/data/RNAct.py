from typing import List, Union, Dict
import os, pickle, sys
import pathlib
from collections import deque
import collections
import pandas as pd
import numpy as np
from io import StringIO
import ast
import multiprocessing
import torch
from torchtext import data
import torchtext


from src.data.basis import BasisDataSet

"""
Loads data from:
archiveII (e2efold)
RNAStrAlign (e2efold)
RNA_STAND (http://www.rnasoft.ca/strand/)
bpRNA (http://bprna.cgrb.oregonstate.edu/download.php#bpRNA)

"""
RNA_SS_data = collections.namedtuple('RNA_SS_data', 'seq ss_label length name pairs')

class RNAct(BasisDataSet):

    def __init__(self, data_dir, max_len, min_len, source, include_N, design=False, partial=False):

        super().__init__(data_dir=data_dir)

        self.rng = np.random.RandomState(seed=100)

        self.samples = {"train": [], "valid": [], "test": []}
        self.split = {"train": 0.8, "valid": 0.1, "test": 0.1}

        dump = True

        self.max_len = max_len
        self.min_len = min_len
        self.include_N = include_N
        self.design  = design

        if not isinstance(partial, int):
            partial = 10
        self.partial  = partial


        self.data_dir = self.data_dir


        config_str = f"{max_len}-{min_len}-{include_N}-{''.join(source) }-design-{design}-partial-{partial}"
        data_file = f".data/RNAct_hash-{config_str}.plk"
        if os.path.isfile(data_file):
            with open(data_file, "rb") as f:
                load_out = pickle.load(f)

                if self.partial:
                    train_list, valid_list, test_list, self.SRC, self.TRG, self.POS1ID, self.POS2ID, self.PK, self.MASK = load_out
                    fields = [("src", self.SRC), ("trg", self.TRG), ("pos1id", self.POS1ID), ("pos2id", self.POS2ID), ("pk", self.PK), ("mask", self.MASK)]
                else:
                    train_list, valid_list, test_list, self.SRC, self.TRG, self.POS1ID, self.POS2ID, self.PK = load_out
                    fields = [("src", self.SRC), ("trg", self.TRG), ("pos1id", self.POS1ID),("pos2id", self.POS2ID),("pk", self.PK)]

                self.train = torchtext.data.Dataset(examples=train_list, fields=fields)
                self.valid = torchtext.data.Dataset(examples=valid_list, fields=fields)
                self.test = torchtext.data.Dataset(examples=test_list, fields=fields)
            print(f"RNA ct file: reload existing data dump {data_file}")
        else:

            file_list = self.get_file_list(source)
            sample_df = self.process_ct_files(file_list)

            self.create_datasets(sample_df)

            self.SRC.build_vocab(self.train.src, min_freq=2)
            self.TRG.build_vocab(self.train.trg, min_freq=2)

            if dump:
                with open(data_file, "wb") as f:
                    if self.partial:
                        work_load = [self.train.examples, self.valid.examples, self.test.examples, self.SRC, self.TRG, self.POS1ID, self.POS2ID, self.PK, self.MASK]
                    else:
                        work_load = [self.train.examples, self.valid.examples, self.test.examples, self.SRC, self.TRG, self.POS1ID, self.POS2ID, self.PK ]
                    pickle.dump(work_load, f)
                print("# RNAct dump successful")


    def get_file_list(self, source):
        file_list = []
        if "RNAStrAlign" in source:
            rna_types = ['tmRNA', 'tRNA', 'telomerase', 'RNaseP', 'SRP', '16S_rRNA', '5S_rRNA', 'group_I_intron']
            for rna_type in rna_types:
                file_list.extend(self.find_ct_files(dir=self.data_dir / "RNAStrAlign" / (rna_type + '_database'), rna_type=rna_type))
        if "archiveII" in source:
            file_list.extend(self.find_ct_files(dir=self.data_dir / "archiveII"))
        if "RNA_Stand" in source:
            file_list.extend(self.find_ct_files(dir=self.data_dir / "RNA_STRAND_data"))
        if "bpRNA" in source:
            file_list.extend(self.find_ct_files(dir=self.data_dir / "bpRNA_dataset/ctFiles", type_segment=1))
        return file_list


    def find_ct_files(self, dir, rna_type=None, type_segment=0):
        file_list = []
        for r, d, f in os.walk(dir):
            for file in f:
                if file.endswith(".ct"):
                    if rna_type:
                        file_list.append({"dir": os.path.join(r, file), "file": file, "type": rna_type, "data_set": dir.parts[-2]})
                    else:
                        file_list.append({"dir": os.path.join(r, file), "file": file, "type": file.split("_")[type_segment], "data_set": dir.parts[-1]})
        return file_list


    def _read_ct_file(self, file_dir:str):

        try:
            lines = "".join([line for line in open(file_dir)
                             if not line.startswith("#")])

            sample = pd.read_csv(StringIO(lines), sep='\s+', skiprows=1, header=None)
        except pd.errors.ParserError:
            print("ParserError", file_dir)
            return False

        sequence = sample.loc[:, 1].to_numpy().tolist()
        if  not self.min_len <= len(sequence) <= self.max_len:
            return False


        pos1id_list = list(sample.loc[:, 0].values)
        pos2id_list = list(sample.loc[:, 4].values)
        pairs_list = list(zip(pos1id_list, pos2id_list))
        pairs_list = list(filter(lambda x: x[1] > 0, pairs_list))
        # pairs_list = list(filter(lambda x: x[1] > x[0], pairs_list))


        if len(pairs_list) < 2:             # TODO include no pair sequcneces
            print("len(paris) < 2", file_dir)
            return False

        if len(pairs_list) % 2 == 1:
            print("len(paris) % 2 = 1", file_dir)
            return False

        norm_numb = pos1id_list[0]
        pairs = np.array(pairs_list) - norm_numb

        if len(sequence) <= np.max(pairs):
            print("len(sequence) <= max(posID)", file_dir)
            return False

        pairs = list(pairs)
        structure, pk_pairs = self.pairs_to_dot_point(pairs, len(sequence))

        pk_pairs = np.asarray(pk_pairs)
        if len(pk_pairs.shape) <= 1:
            print("len(pk_pairs.shape) <= 1", file_dir)
            return False
        pk_pairs = pk_pairs[np.lexsort((pk_pairs[:, 0], pk_pairs[:, 2]))]
        pos1id = pk_pairs[:, 0].tolist()
        pos2id = pk_pairs[:, 1].tolist()
        pseudoknot = pk_pairs[:, 2].tolist()

        return {"structure": structure, "sequence":sequence, "pos1id":pos1id, "pos2id":pos2id, "pk": pseudoknot}


    @staticmethod # also used in RNAspotrna
    def pairs_to_dot_point(pairs, length, inverse=False):
        structure = ['.'] * length
        pk_pairs = []

        if len(pairs) == 0:
            return structure, pk_pairs

        pairs = np.stack(pairs, axis=0)
        if inverse:
            pairs = pairs[pairs[:, 0].argsort()[::-1]]
        else:
            pairs = pairs[pairs[:, 0].argsort()]
        pairs = [pairs[i, :] for i in range(pairs.shape[0])]
        pair_buffer = [deque(), deque(), deque(), deque()]

        for i, j in pairs:

            in_level = -1
            for level, buffer in enumerate(pair_buffer):
                if (i, j) in buffer:
                    in_level = level
                    break
            if in_level == -1:
                pair_buffer[0].append((j, i))
            else:
                while True:
                    if (i, j) == pair_buffer[in_level][-1]:
                        pair_buffer[in_level].pop()
                        if inverse:
                            structure[j] = f"){in_level}"
                            structure[i] = f"({in_level}"
                            pk_pairs.append([i, j, in_level])
                        else:
                            structure[j] = f"({in_level}"
                            structure[i] = f"){in_level}"
                            pk_pairs.append([j, i, in_level])
                        break
                    else:
                        pair_buffer[min(in_level + 1, len(pair_buffer) - 1)].insert(0, pair_buffer[in_level].pop())
        return structure, pk_pairs





    def process_ct_files(self, file_list):

        print("file list len: ", len(file_list))

        sample_list = []
        for file_dict in file_list:
            sample_dict = self._read_ct_file(file_dict['dir'])
            if sample_dict:
                sample_dict.update(file_dict)
                sample_list.append(sample_dict)

        sample_df = pd.DataFrame(sample_list)

        sample_df = sample_df[sample_df['sequence'].apply(lambda x: self.min_len <= len(x) <= self.max_len)]
        print("samples after length drop",  sample_df.shape[0])

        if self.include_N:
            bases = ["A", "C", "G", "U", "N"]
        else:
            bases = ["A", "C", "G", "U"]
        sample_df = sample_df[sample_df['sequence'].apply(lambda x: all([s in bases for s in x])   )]
        print("samples after wrong bases drop", sample_df.shape[0])

        sample_df['seq_struc_str'] = sample_df['sequence'].apply(lambda x: ''.join(x)) + sample_df['structure'].apply(lambda x: ''.join(x))
        # drop equal sequences
        sample_df = sample_df.loc[~sample_df['seq_struc_str'].duplicated()]
        sample_df = sample_df.drop(columns=['seq_struc_str', 'dir'])
        print("samples after drop duplicated", sample_df.shape[0])

        sample_df = sample_df.reset_index()

        return sample_df

    def save_to_file(self, sample_df:pd.DataFrame):
        for idx in range(sample_df.shape[0]):
            with open(f'./rnajoined/rnajoined_{idx//500:06}.fasta', 'a') as f:
                seq = ''.join(sample_df['sequence'][idx])
                name = f">{sample_df['data_set'][idx]}_{sample_df['type'][idx]}_{idx:06}"
                f.write(name + "\n")
                f.write(seq + "\n")


    def split_dataframe(self, samples_df:pd.DataFrame ):

        train_df = pd.DataFrame()
        valid_df = pd.DataFrame()
        test_df = pd.DataFrame()

        rna_types = samples_df['type'].unique().tolist()
        for rtype in rna_types:
            sub_df = samples_df.loc[samples_df['type'] == rtype]

            rand_idx = self.rng.permutation(sub_df.shape[0])

            train_idx = int(sub_df.shape[0] * self.split['train'])
            valid_idx = int(sub_df.shape[0] * self.split['valid']) + train_idx

            sub_train_df = sub_df.iloc[rand_idx[:train_idx]]
            sub_valid_df = sub_df.iloc[rand_idx[train_idx:valid_idx]]
            sub_test_df = sub_df.iloc[rand_idx[valid_idx:]]

            train_df = pd.concat([train_df, sub_train_df]) #train_df.append(sub_train_df, ignore_index=True)
            valid_df = pd.concat([valid_df, sub_valid_df]) #valid_df.append(sub_valid_df, ignore_index=True)
            test_df = pd.concat([test_df, sub_test_df]) #test_df.append(sub_test_df, ignore_index=True )

            # print(f"family: {rtype} rand_idx", rand_idx.shape)
            # print(f"family: {rtype} train_idx", train_idx)
            print(f"family: {rtype} sub_train_df", sub_train_df.shape)
            print(f"family: {rtype} train_df", train_df.shape)

        train_df = train_df.reset_index()
        valid_df = valid_df.reset_index()
        test_df = test_df.reset_index()

        return train_df,  valid_df,  test_df


    def create_sample_list(self, sample_df):

        sample_list = []
        for idx, sample in sample_df.iterrows():
            sample_list.append({"sequence": sample["sequence"], "structure": sample["structure"], "pos1id": sample["pos1id"]
                                   , "pos2id": sample["pos2id"], "pk": sample["pk"]})

        return sample_list




    def create_datasets(self, sample_df):

        train_df,  valid_df,  test_df = self.split_dataframe(sample_df)

        print("train_df", train_df.shape)
        print("valid_df", valid_df.shape)
        print("test_df ", test_df.shape)

        samples = {}
        train_sample_list = self.create_sample_list(train_df)
        train_sample_list = self.rng.permutation(train_sample_list)
        samples['train'] = train_sample_list
        samples['valid'] = self.create_sample_list(valid_df)
        samples['test'] = self.create_sample_list(test_df)

        self.SRC = data.Field(pad_token=self.blank_word, unk_token=None, batch_first=True, include_lengths=True)
        self.TRG = data.Field(pad_token=self.blank_word, unk_token=None, batch_first=True, include_lengths=True)
        self.POS1ID = data.Field(pad_token=-1, batch_first=True, include_lengths=True, use_vocab=False)
        self.POS2ID = data.Field(pad_token=-1, batch_first=True, include_lengths=True, use_vocab=False)
        self.PK = data.Field(pad_token=-1, batch_first=True, include_lengths=True, use_vocab=False)

        fields = {"src": ("src", self.SRC), "trg": ("trg", self.TRG), "pos1id": ("pos1id", self.POS1ID),
                  "pos2id": ("pos2id", self.POS2ID),"pk": ("pk", self.PK)}

        if self.partial:
            self.MASK = data.Field(pad_token=-1, batch_first=True, include_lengths=True, use_vocab=False)
            fields = {"src": ("src", self.SRC), "trg": ("trg", self.TRG), "pos1id": ("pos1id", self.POS1ID),
                      "pos2id": ("pos2id", self.POS2ID), "pk": ("pk", self.PK), "mask": ("mask", self.MASK)}



        for set, sample_list in samples.items():
            example_list = []
            for sample in sample_list:

                if self.partial:
                    structure = sample.pop("structure")
                    sequence = sample.pop("sequence")

                    if set == "train":
                        for i in range(self.partial):
                            src, trg, mask = self.make_partial(sequence=sequence, structure=structure)
                            sample["src"] = src
                            sample["trg"] = trg
                            sample["mask"] = mask
                            example_list.append(torchtext.data.Example().fromdict(sample, fields=fields))
                    else:
                        # src, trg, mask = self.make_partial(sequence=sequence, structure=structure)
                        # sample["src"] = src
                        # sample["trg"] = trg
                        # sample["mask"] = mask

                        sample["src"] = structure
                        sample["trg"] = sequence + [self.blank_word] * len(sample["src"])
                        sample["mask"] = [1]*len(sample["src"]) + [0]*len(sample["src"])
                        example_list.append(torchtext.data.Example().fromdict(sample, fields=fields))


                elif self.design:
                    sample["src"] = sample.pop("structure")
                    sample["trg"] = sample.pop("sequence")
                    example_list.append(torchtext.data.Example().fromdict(sample, fields=fields))
                else:
                    sample["src"] = sample.pop("sequence")
                    sample["trg"] = sample.pop("structure")
                    example_list.append(torchtext.data.Example().fromdict(sample, fields=fields))



            setattr(self, set, torchtext.data.Dataset(example_list, fields=list(fields.values())))


    def make_partial(self, sequence, structure):
        trg_structure = []
        trg_sequence = []
        src = []
        mask_structure = []
        mask_sequence = []

        for seq, struct in zip(sequence, structure):
            choice = self.rng.choice(["seq", "struct", "x"], p=[0.4, 0.4, 0.2])

            if choice == "seq":
                trg_structure.append(struct)
                trg_sequence.append(self.blank_word)
                src.append(seq)
                mask_structure.append(1)
                mask_sequence.append(0)
            elif choice == "struct":
                trg_structure.append(self.blank_word)
                trg_sequence.append(seq)
                src.append(struct)
                mask_structure.append(0)
                mask_sequence.append(1)
            elif choice == "x":
                trg_structure.append(struct)
                trg_sequence.append(seq)
                src.append("X")
                mask_structure.append(1)
                mask_sequence.append(1)

        if self.rng.choice([True, False]):
            trg_joined = trg_structure + trg_sequence
            mask_joined = mask_structure + mask_sequence
        else:
            trg_joined = trg_sequence + trg_structure
            mask_joined = mask_sequence + mask_structure

        return src, trg_joined, mask_joined




if __name__ == "__main__":
    device = torch.device("cpu")
    batch_size = 12000

    data_dir = "/home/joerg/workspace/data/rna"

    # data_set = RNAct(data_dir=data_dir, max_len=200, min_len=10, source=[ "RNAStrAlign" ], include_N=True, design=False, partial=20)
    data_set = RNAct(data_dir=data_dir, max_len=1800, min_len=10, source=["RNA_Stand", "archiveII", "RNAStrAlign" , "bpRNA" ],
                     include_N=True, design=False, partial=False)
    # data_set = RNAct(data_dir=data_dir, max_len=2000, min_len=1, source=["RNA_Stand", "archiveII", "RNAStrAlign" ], include_N=False, design=False)



    train_iter = data_set.get_iterator("train", batch_size=batch_size, device=device, train=True)
    valid_iter = data_set.get_iterator("valid", batch_size=batch_size, device=device, train=False)
    test_iter = data_set.get_iterator("test", batch_size=batch_size, device=device, train=False)

    batch = next(train_iter.__iter__())
    src_data, src_length = batch.src
    trg_data, trg_length = batch.trg
    pos1id_data, pos1id_len = batch.pos1id

    # src_data  [batch size, sequencel length] [0,1,2,3,1,2,3,1] (int list)
    # stoi = data_set.SRC.vocab.stoi -> stoi == string to int is a dict {"<blank>":0, "A":1, "C":2, ... "N":}    stoi["A"]
    # itos = data_set.SRC.vocab.itos -> itos == int to string  is a list ["<blank>","A",...]    itos[0]
    # itos = data_set.TRG.vocab.itos -> itos == int to string  is a list ["<blank>",".", "(0",")0","(1",")1",...,")3"]    itos[0]

    print(f"# train set size: {data_set.set_size('train')}")
    seq_length = [batch.src[1].detach().cpu().numpy().tolist() for batch in train_iter]
    seq_length = [item for sublist in seq_length for item in sublist]
    # print("seq_length", [s.shape for s in seq_length if len(s.shape) > 1])

    print(f"# train seq length min: {np.min(seq_length)}")
    print(f"# train seq length mean: {np.mean(seq_length)}")
    print(f"# train seq length max: {np.max(seq_length)}")

    print(f"# valid set size: {data_set.set_size('valid')}")
    seq_length = [batch.src[1].detach().cpu().numpy().squeeze() for batch in valid_iter]
    seq_length = np.concatenate(seq_length, axis=0)
    print(f"# valid seq length min: {np.min(seq_length)}")
    print(f"# valid seq length mean: {np.mean(seq_length)}")
    print(f"# valid seq length max: {np.max(seq_length)}")

    print(f"# test set size: {data_set.set_size('test')}")

    print(f"src vocab: {data_set.src_vocab}")
    print(f"src vocab size: {data_set.src_vocab_size}")
    print(f"trg vocab: {data_set.trg_vocab}")
    print(f"trg vocab size: {data_set.trg_vocab_size}")



    # start_symbol = data_set.stoi(data_set.bos_word, origin='trg')
    # print("start_symbol: ", start_symbol)
    dist_list = []
    len_list = []
    knots_start_list = []
    for batch in train_iter:
        batch_pos1, len_pos1 = batch.pos1id
        batch_pos2, len_pos2 = batch.pos2id
        batch_src, len_src = batch.src

        assert torch.all(len_pos1.eq(len_pos2))

        for i in range(batch_pos1.size(0)):
            pos_id1 = batch_pos1[i,:len_pos1[i]].cpu().numpy()
            pos_id2 = batch_pos2[i,:len_pos2[i]].cpu().numpy()
            distance = np.abs(pos_id1-pos_id2)
            dist_list.append(distance)
            for dist, pos1, pos2 in zip(distance, pos_id1, pos_id2):
                knots_start_list.append([np.min([pos1, pos2]), dist])
            len_list.append(len_src[i])

    dist_list = np.concatenate(dist_list, 0)
    knots_start_list = np.asarray(knots_start_list)
    print("long_knots_start_list", knots_start_list.shape)
    import matplotlib.pyplot as plt
    plt.subplot(4, 1, 1)
    plt.hist(len_list, bins=40, label="sequence length")
    plt.ylabel("sequence length")
    plt.subplot(4, 1, 2)
    plt.hist(dist_list, bins=40, label="knot width")
    plt.ylabel("knot width")
    plt.subplot(4, 1, 3)
    plt.hist(knots_start_list[:,0], bins=40)
    plt.ylabel("knots_start")
    plt.subplot(4, 1, 4)
    plt.hist2d(knots_start_list[:, 1], knots_start_list[:, 0], bins=40)
    plt.ylabel("knots_start")
    plt.xlabel("knot width")
    plt.show()
    print("min pos dist: ", np.min(dist_list))
    print("mean pos dist: ", np.mean(dist_list))
    print("max pos dist: ", np.max(dist_list))

    # for batch in train_iter:
    #     target = batch.trg[0]
    #     target = target.cpu().numpy()
    #     target = target.flatten()
    #     uniques = np.unique(target).tolist()
    #     print(data_set.itos(uniques, origin="trg"))

    sample = train_iter.__iter__().__next__()
    #
    # print("src mat", sample.src[0].size())
    # print("src len", sample.src[1].size())
    #
    # print("trg mat", sample.trg[0].size())
    # print("trg len", sample.trg[1].size())
    #
    print("src sentence int", sample.src[0][0, :])
    print("src sentence int", sample.src[1][0])
    print("src sentence str", data_set.itos(sample.src[0][0, :].cpu().numpy().tolist(), origin="src"))

    print("trg sentence int", sample.trg[0][0, :])
    print("trg sentence int", sample.trg[1][0])
    print("trg sentence str", data_set.itos(sample.trg[0][0, :].cpu().numpy().tolist(), origin="trg"))
    #
    # print("pos1id", sample.pos1id[0][0, :])
    # print("pos2id", sample.pos2id[0][0, :])
