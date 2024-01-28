from typing import List, Union, Dict
import pathlib
import time, os, pickle
import numpy as np

from torchtext import data, datasets
from torchtext.data import BucketIterator, BPTTIterator

global max_src_in_batch, max_tgt_in_batch


def batch_size_fn(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0

    max_src_in_batch = max(max_src_in_batch, len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch, len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)


class SubSortedBucketIterator(data.Iterator):
    def create_batches(self):
        if self.train:
            def pool(d, random_shuffler):
                for p in data.batch(d, self.batch_size * 100):
                    p_batch = data.batch(sorted(p, key=self.sort_key), self.batch_size, self.batch_size_fn)
                    for b in random_shuffler(list(p_batch)):
                        yield b

            self.batches = pool(self.data(), self.random_shuffler)
        else:
            self.batches = []
            for b in data.batch(self.data(), self.batch_size, self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))


class BasisDataSet():

    def __init__(self, data_dir=None):

        self.blank_word = '<blank>'
        self.bos_word = '<s>'
        self.eos_word = '</s>'

        if data_dir is not None:
            self.data_dir = pathlib.Path(data_dir)
            if not os.path.isdir(".data"):
                os.mkdir(".data")

    def get_iterator(self, set_name: str, batch_size: int = None, train=False, sorted=True, bptt=False, device='cpu'):

        if batch_size is None:
            batch_size = self.batch_size
        data_set = getattr(self, set_name)

        if sorted:
            data_iter = SubSortedBucketIterator(data_set, batch_size=batch_size, device=device, repeat=False,
                                                sort_key=lambda x: (len(x.src), len(x.trg)), batch_size_fn=batch_size_fn, train=train)
        elif bptt:
            data_iter = BPTTIterator(data_set, batch_size=batch_size, device=device, repeat=False, bptt_len=bptt, train=train)
        else:
            data_iter = BucketIterator(data_set, batch_size=batch_size, device=device, repeat=False,
                                       sort_key=lambda x: (len(x.src), len(x.trg)), batch_size_fn=batch_size_fn, train=train)
        return data_iter

    def set_size(self, set_name):
        return len(getattr(self, set_name))

    @property
    def src_vocab(self):
        return self.SRC.vocab.freqs

    @property
    def trg_vocab(self):
        return self.TRG.vocab.freqs

    @property
    def src_vocab_size(self):
        return len(self.SRC.vocab)

    @property
    def trg_vocab_size(self):
        return len(self.TRG.vocab)

    def _convert_list(self, l, dictionary):
        result = []
        for t in l:
            result.append(dictionary[t])
        return result

    def _get_vocab(self, origin):
        if origin == "src":
            return self.SRC.vocab
        elif origin == "trg":
            return self.TRG.vocab
        elif origin == "dataset":
            return self.DATASET.vocab
        else:
            raise UserWarning(f"Wrong origin {origin}, expect src or trg")

    def _get_stoi(self, origin):
        if origin == "src":
            return self.SRC.vocab.stoi
        elif origin == "trg":
            return self.TRG.vocab.stoi
        elif origin == "dataset":
            return self.DATASET.vocab.stoi
        else:
            raise UserWarning(f"Wrong origin {origin}, expect src or trg")

    def stoi(self, s: Union[List[str], str], origin: str) -> Union[List[int], int]:
        vocab = self._get_vocab(origin)
        if isinstance(s, str):
            return vocab.stoi[s]
        elif isinstance(s, List):
            return self._convert_list(s, vocab.stoi)
        else:
            raise UserWarning(f"Wrong data type: {type(s)}, expect int or List")

    def itos(self, i: Union[List[int], int], origin: str) -> Union[List[str], str]:
        vocab = self._get_vocab(origin)
        if isinstance(i, int):
            return vocab.itos[i]
        elif isinstance(i, List):
            return self._convert_list(i, vocab.itos)
        else:
            raise UserWarning(f"Wrong data type: {type(i)}, expect int or List")

    def add_bos_eos(self, l: List[Union[int, str]], origin: str) -> List[Union[int, str]]:
        if isinstance(l[0], int):
            bos_word = self.stoi(self.bos_word, origin)
            eos_word = self.stoi(self.eos_word, origin)
            blank_word = self.stoi(self.blank_word, origin)
        elif isinstance(l[0], str):
            bos_word = self.bos_word
            eos_word = self.eos_word
            blank_word = self.blank_word
        else:
            raise UserWarning(f"Wrong data type in list: {type(l[0])}, expect list elements are int or str")

        if l[0] != bos_word:
            l = [bos_word] + l
        if not eos_word in l:
            if blank_word in l:
                blank_idx = l.index(blank_word)
                l[blank_idx] = eos_word
            else:
                l = l + [eos_word, ]
        return l

    def remove_bos_eos(self, l: List[Union[int, str]], origin: str) -> List[Union[int, str]]:
        if isinstance(l[0], int):
            bos_word = self.stoi(self.bos_word, origin)
            eos_word = self.stoi(self.eos_word, origin)
            blank_word = self.stoi(self.blank_word, origin)
        elif isinstance(l[0], str):
            bos_word = self.bos_word
            eos_word = self.eos_word
            blank_word = self.blank_word
        else:
            raise UserWarning(f"Wrong data type in list: {type(l[0])}, expect list elements are int or str")

        if l[0] == bos_word:
            l = l[1:]
        if eos_word in l:
            eos_idx = l.index(eos_word)
            l = l[:eos_idx]
        if blank_word in l:
            blank_idx = l.index(blank_word)
            l = l[:blank_idx]
        return l
