import time
import re
import subprocess

import numpy as np
import pandas as pd

from itertools import product
from dataclasses import dataclass
from distance import hamming
from io import StringIO
from pathlib import Path
from Bio import AlignIO

from tensorforce.environments import Environment
from RNA import fold

# from src.e2efold.e2efold_productive_short import e2efold_main_short
# from src.e2efold.e2efold_productive_long import e2efold_main_long
# from src.e2efold.utils import ct2db


@dataclass
class RnaDesignEnvironmentConfig:
    """
    Dataclass for the configuration of the environment.

    Default values describe:
        mutation_threshold: Defines the minimum distance needed before applying the local
            improvement step.
        reward_exponent: A parameter to shape the reward function.
        state_radius: The state representation is a (2*<state_radius> + 1)-gram
            at each position.
        use_conv: Bool to state if a convolutional network is used or not.
        use_embedding: Bool to state if embedding is used or not.
        gc_improvement_step: Bool to decide if gc_improvement step is used.
        gc_tolerance: The tolerance for gc content.
        desired_gc: The gc_content that is desired.
        gc_weight: Determines how much weight is set on the gc content control.
        gc_reward: Bool to decide if the gc_content is included in the reward function.
        num_actions: The number of actions the agent can choose from.
    """
    mutation_threshold: int = 5
    reward_exponent: float = 1.0
    state_radius: int = 5
    use_conv: bool = False
    use_embedding: bool = False
    local_design: bool = False
    predict_pairs: bool = False
    reward_function: str = 'structure'
    state_representation: str = 'sequence-progress'
    data_type: str = 'random'
    sequence_constraints: str = None
    encoding: str = 'sequence_bias'
    min_length: int = 0
    max_length: int = 0
    min_dot_extension: int = 0
    variable_length: bool = False
    control_gc: bool = False
    desired_gc: float = None
    gc_tolerance: float = 0.01
    algorithm: str = 'rnafold'
    folding_counter: int = 0
    interactions: int = 0
    seed: int = 42  # 1  # 123 # 0  # 137  # 42
    cm_design: bool = False
    cm_name: str = 'RF00008'
    cm_path: str = 'rfam_cms/Rfam.cm'
    working_dir: str = 'working_dir'
    rri_design: bool = False
    rri_target: str = 'UUUAAAUUAAAAAAUCAUAGAAAAAGUAUCGUUUGAUACUUGUGAUUAUACUCAGUUAUACAGUAUCUUAAGGUGUUAUUAAUAGUGGUGAGGAGAAUUUAUGAAGCUUUUCAAAAGCUUGCUUGUGGCACCUGCAACUCUUGGUCUUUUAGCACCAAUGACCGCUACUGCUAAU'  # mRNA1 of intarna example



tuple_encoding = {
    ('=', '='): 0,
    ('A', '.'): 1,
    ('A', '('): 2,
    ('A', ')'): 3,
    ('A', 'N'): 4,
    ('C', '.'): 5,
    ('C', '('): 6,
    ('C', ')'): 7,
    ('C', 'N'): 8,
    ('G', '.'): 9,
    ('G', '('): 10,
    ('G', ')'): 11,
    ('G', 'N'): 12,
    ('U', '.'): 13,
    ('U', '('): 14,
    ('U', ')'): 15,
    ('U', 'N'): 16,
    ('N', '.'): 17,
    ('N', '('): 18,
    ('N', ')'): 19,
    ('N', 'N'): 20,
}

# tuple_encoding = {
#     ('=', '='): 0,
#     ('A', '.'): 1,
#     ('A', '('): 2,
#     ('A', ')'): 3,
#     # ('A', '|'): 4,
#     ('A', 'N'): 5,
#     ('C', '.'): 6,
#     ('C', '('): 7,
#     ('C', ')'): 8,
#     # ('C', '|'): 8,
#     ('C', 'N'): 9,
#     ('G', '.'): 10,
#     ('G', '('): 11,
#     ('G', ')'): 12,
#     # ('G', '|'): 13,
#     ('G', 'N'): 14,
#     ('U', '.'): 15,
#     ('U', '('): 16,
#     ('U', ')'): 17,
#     # ('U', '|'): 18,
#     ('U', 'N'): 19,
#     ('N', '.'): 20,
#     ('N', '('): 21,
#     ('N', ')'): 22,
#     # ('N', '|'): 23,
#     ('N', 'N'): 24,
# }



def _fold_primary(primary, env_config):
    # increment folding counter
    env_config.folding_counter += 1
    # fold with rnafold
    if env_config.algorithm == 'rnafold':
        folding = fold(primary)[0]
    elif env_config.algorithm == 'rnafold_mea':
        pathname = ''.join([str(env_config.desired_gc)])+'_rnafold.fasta'
        rnafold_path = Path(pathname)
        with open(rnafold_path, 'w+') as f:
            f.write('>current' + '\n')
            f.write(primary)
        folding = subprocess.Popen(["RNAfold", "--MEA", "-i",  str(rnafold_path.resolve())], stdout=subprocess.PIPE).communicate()[0]
        folding = folding.decode("utf-8").split('\n')[5].split()[0]  # 5 is MEA structure output, 0 is structure in DB notation
        rnafold_path.unlink()
    # fold with e2efold
    elif env_config.algorithm == 'e2efold':
        if len(primary) > 600:
            ct_df = pd.read_csv(StringIO(e2efold_main_long(primary, 'src/e2efold/config_long.json')), sep='\s+', header=None)
            folding = ''.join(change_encoding(ct2db(ct_df)))
        else:
            ct_df = pd.read_csv(StringIO(e2efold_main_short(primary, 'src/e2efold/config.json')), sep='\s+', header=None)
            folding = ''.join(change_encoding(ct2db(ct_df)))
    # fold with linearfold
    elif env_config.algorithm == 'linearfold':
        echo_ps = subprocess.Popen(['echo', f"{primary}"], stdout=subprocess.PIPE)
        folding = subprocess.Popen(["./thirdparty/linearfold/LinearFold/linearfold"], stdin=echo_ps.stdout, stdout=subprocess.PIPE).communicate()[0].decode("utf-8").split()[1]
    elif env_config.algorithm == 'linearfoldV':
        echo_ps = subprocess.Popen(['echo', f"{primary}"], stdout=subprocess.PIPE)
        folding = subprocess.Popen(["./thirdparty/linearfold/LinearFold/linearfold", "-V"], stdin=echo_ps.stdout, stdout=subprocess.PIPE).communicate()[0].decode("utf-8").split()[1]

    elif env_config.algorithm == 'contrafold':
        pathname = ''.join([str(env_config.desired_gc)])+'_contrafold.fasta'
        contrafold_path = Path(pathname)
        with open(contrafold_path, 'w+') as f:
            f.write('>current' + '\n')
            f.write(primary)
            # echo_ps = subprocess.Popen(['echo', ">current" + '\n' + f"{primary}"], stdout=f)
        folding = subprocess.Popen(["contrafold", "predict", "--noncomplementary", str(contrafold_path.resolve())], stdin=contrafold_path.open(), stdout=subprocess.PIPE).communicate()[0].decode("utf-8").split()[-1]  # , stdin=echo_ps.stdout
        contrafold_path.unlink()
    return folding


def _encode_tuple(secondary, primary, env_config):
    """
    Encodes the sequence and structure constraints into a joint represenation from tuples of sequence and structure.

    Args:
        secondary <str>: input structure constraints.
        primary <str>: input sequence constraints.
        env_config <RnaDesignEnvironmentConfig>: the environment config file.

    Returns:
        A list/list of lists that encodes each site with a natural number.
    """
    padding = "=" * env_config.state_radius
    padded_secondary = padding + secondary + padding
    padded_primary = padding + primary + padding

    p_primary_list = list(padded_primary)
    p_secondary_list = list(padded_secondary)

    p_encoding = [tuple_encoding[(x, y)] for x, y in zip(p_primary_list, p_secondary_list)]

    if env_config.use_conv and not env_config.use_embedding:
        return [[site] for site in p_encoding]
    return [site for site in p_encoding]


def multihot(secondary, primary, env_config):
    """TODO
    """
    # one_hot = {
    #     'A': np.array([[1], [0], [0], [0], [0], [0], [0], [0], [0], [0]]),  # [A, C, G, U, N, ., (, ), |, =]
    #     'C': np.array([[0], [1], [0], [0], [0], [0], [0], [0], [0], [0]]),
    #     'G': np.array([[0], [0], [1], [0], [0], [0], [0], [0], [0], [0]]),
    #     'U': np.array([[0], [0], [0], [1], [0], [0], [0], [0], [0], [0]]),
    #     'N': np.array([[0], [0], [0], [0], [1], [0], [0], [0], [0], [0]]),
    #     '.': np.array([[0], [0], [0], [0], [0], [1], [0], [0], [0], [0]]),
    #     '(': np.array([[0], [0], [0], [0], [0], [0], [1], [0], [0], [0]]),
    #     ')': np.array([[0], [0], [0], [0], [0], [0], [0], [1], [0], [0]]),
    #     '|': np.array([[0], [0], [0], [0], [0], [0], [0], [0], [1], [0]]),
    #     '=': np.array([[0], [0], [0], [0], [0], [0], [0], [0], [0], [1]]),
    # }

    one_hot = {
        'A': np.array([[1], [0], [0], [0], [0], [0], [0], [0], [0]]),  # [A, C, G, U, N, ., (, ), |, =]
        'C': np.array([[0], [1], [0], [0], [0], [0], [0], [0], [0]]),
        'G': np.array([[0], [0], [1], [0], [0], [0], [0], [0], [0]]),
        'U': np.array([[0], [0], [0], [1], [0], [0], [0], [0], [0]]),
        'N': np.array([[0], [0], [0], [0], [1], [0], [0], [0], [0]]),
        '.': np.array([[0], [0], [0], [0], [0], [1], [0], [0], [0]]),
        '(': np.array([[0], [0], [0], [0], [0], [0], [1], [0], [0]]),
        ')': np.array([[0], [0], [0], [0], [0], [0], [0], [1], [0]]),
        '=': np.array([[0], [0], [0], [0], [0], [0], [0], [0], [1]]),
    }


    padding = "=" * env_config.state_radius
    padded_secondary = padding + secondary + padding
    padded_primary = padding + primary + padding

    p_primary_list = list(padded_primary)
    p_secondary_list = list(padded_secondary)

    p_primary_encoding = list(map(lambda x: one_hot[x], p_primary_list))
    p_secondary_encoding = list(map(lambda x: one_hot[x], p_secondary_list))

    p_primary_encoding = np.stack(p_primary_encoding, axis=0)
    p_secondary_encoding = np.stack(p_secondary_encoding, axis=0)

    encoding = p_primary_encoding + p_secondary_encoding
    encoding[encoding > 1] = 1

    return encoding


def _string_difference_indices(s1, s2):
    """
    Returns all indices where s1 and s2 differ.

    Args:
        s1: The first sequence.
        s2: The second sequence.

    Returns:
        List of indices where s1 and s2 differ.
    """
    return [index for index in range(len(s1)) if s1[index] != s2[index]]


def _encode_dot_bracket(secondary, env_config):
    """
    Encode the dot_bracket notated target structure. The encoding can either be binary
    or by the embedding layer.

    Args:
        secondary: The target structure in dot_bracket notation.
        env_config: The configuration of the environment.

    Returns:
        List of encoding for each site of the padded target structure.
    """
    padding = "=" * env_config.state_radius
    padded_secondary = padding + secondary + padding

    if env_config.use_embedding:
        # site_encoding = {".": 0, "(": 1, ")": 2, "|": 3, "=": 4, "A": 5, "G": 6, "C": 7, "U": 8, "N": 9}
        site_encoding = {".": 0, "(": 1, ")": 2, "=": 3, "A": 4, "G": 5, "C": 6, "U": 7, "N": 8}
    else:
        site_encoding = {".": 0, "(": 0, ")": 0, "|": 0, "=": 1, "A": 2, "G": 2, "C": 2, "U": 2, "N": 1}

    # Sites corresponds to 1 pixel with 1 channel if convs are applied directly
    if env_config.use_conv and not env_config.use_embedding:
        return [[site_encoding[site]] for site in padded_secondary]
    return [site_encoding[site] for site in padded_secondary]


def _encode_pairing(secondary):
    """TODO
    """
    # initialize pairing encoding with None everywhere
    pairing_encoding = [None] * len(secondary)
    # initialize stack
    stack = []
    # loop over input secondary
    for index, symbol in enumerate(secondary, 0):
        # if opening bracket pu on stack; note opening brackets correspond to next closing bracket, therefore opening brackets on the stack don't matter for unbalanced brackets
        if symbol == "(":
            stack.append(index)
        # if closing bracket, pair with last opening bracket on stack; if there is no stack, closing bracket is unbalanced bracket and we continue
        elif symbol == ")":
            if not stack:
                continue
            # if there is a stack pair with last opening bracket
            else:
                paired_site = stack.pop()
                pairing_encoding[paired_site] = index
                pairing_encoding[index] = paired_site
    return pairing_encoding


def _encode_structure_parts(local_target):
    """TODO
    """
    encoding = [None] * len(local_target)
    for index, site in enumerate(local_target):
        if site in ['.', '(', ')', 'N']:
            encoding[index] = site
    return encoding


class GcControler(object):
    """TODO
    Class for local improvement of the GC content of the designed sequence.
    """
    def __init__(self, design, target, gc_tolerance, env_config):
        self._env_config = env_config
        self._design = design
        self._target = target
        self._gc_tolerance = gc_tolerance
        if self._target.gc is None:
            self._desired_gc = 0.5
        else:
            self._desired_gc = float(self._target.gc)

    def gc_content_primary(self, primary):
        return (primary.upper().count('G') + primary.upper().count('C')) / len(primary)

    def gc_content(self, design):
        return (design.primary.upper().count('G') + design.primary.upper().count('C')) / len(design.primary)

    def gc_diff(self, design):
        return self.gc_content(design) - self.desired_gc

    def gc_diff_abs(self, design):
        return abs(self.gc_diff(design))

    def gc_diff_abs_primary(self, primary):
        return abs(self.gc_content_primary(primary) - self.desired_gc)

    def gc_satisfied(self, design):
        return self.gc_diff_abs(design) <= self._gc_tolerance

    @property
    def gc_tolerance(self):
        return self._gc_tolerance

    @property
    def desired_gc(self):
        return self._desired_gc

    def gc_improvement_step(self, hamming_distance=0, site=0):
        """TODO
        """
        # if gc satisfied just return the design
        if self.gc_satisfied(self._design):
            return self._design

        # start with paired sites, save single sites for second gc improvement step if necessary
        single_sites = []

        sites = list(range(0, len(self._design.primary)))

        # go through all sites of the current best candidate solution (updated during improvement)
        for i in range(len(self._design.primary)):

            site = np.random.choice(sites)
            sites.remove(site)

            # if gc already satisfied return current design
            if self.gc_satisfied(self._design):
                return self._design

            # if site is constrained in sequence space, don't change it
            if self._target.sequence_constraints[site] != 'N':
                continue

            # if site is single site, ignore now, use later
            if not self._target.get_paired_site(site):
                single_sites.append(site)
                continue

            # if gc content is too low (gc_current - gc_desired) < 0, increas gc content
            if self.gc_diff(self._design) < 0:
                self._increase_gc(site, self._target.get_paired_site(site), hamming_distance)

            # else decreas gc content
            else:
                self._decrease_gc(site, self._target.get_paired_site(site), hamming_distance)

        # if improving paired sites wasn't sufficient, start improving single sites
        if not self.gc_satisfied(self._design):
            self._gc_improve_single_sites(single_sites, hamming_distance=hamming_distance)

        # return the current (improved design)
        return self._design


    def _gc_improve_single_sites(self, single_sites, hamming_distance=0):
        """TODO
        """
        sites = single_sites
        # walk over all single sites
        for i in range(len(single_sites)):

            site = np.random.choice(sites)
            sites.remove(site)

            # if gc content satisfied, break
            if self.gc_satisfied(self._design):
                break
            # should already not happen because checked in outer loop already...but...
            if self._target.sequence_constraints[site] != 'N':
                continue
            # else, start improving
            else:
                # if gc content is too low (gc_current - gc_desired) < 0, increas gc content
                if self.gc_diff(self._design) < 0:
                    self._increase_gc(site, self._target.get_paired_site(site), hamming_distance)
                # else decreas gc content
                else:
                    self._decrease_gc(site, self._target.get_paired_site(site), hamming_distance)

    def _increase_gc(self, site, paired_site=None, hamming_distance=0):
        """TODO
        """
        # create two temporary primaries to check for gc-improvement (two because we swap A/U with G and C)
        primary_tmp_1 = list(self._design.primary)
        primary_tmp_2 = list(self._design.primary)
        # if current site is A or U replace with G and C (and watson-crick pair base if site is paired)
        if self._design.primary[site] in ['A', 'U']:
            # assign G and C separately
            primary_tmp_1[site] = 'G'
            primary_tmp_2[site] = 'C'
            # and assign paired site if nucleotide is paired; paired site is not restricted in sequence space, otherwise site would already have been restricted
            if paired_site:
                primary_tmp_1[paired_site] = 'C'
                primary_tmp_2[paired_site] = 'G'

            # fold first tmp candidate
            folding_tmp_1 = _fold_primary(''.join(primary_tmp_1), self._env_config)

            # check first if hamming distance is at least the same for tmp candidate 1 (we do not want to increase hamming distance)
            if hamming_with_n(folding_tmp_1, self._target.dot_bracket) <= hamming_distance:
                # if hamming is ok, check if gc content would be better (abs diff)
                if self.gc_diff_abs_primary(''.join(primary_tmp_1)) < self.gc_diff_abs(self._design):
                    # if gc is better, asssign site to the design
                    self._design.assign_sites(0, site, self._target.get_paired_site(site))  # assign G/GC <=> action 0
            else:
                # do the same for candidate 2 if candidate 1 failed
                folding_tmp_2 = _fold_primary(''.join(primary_tmp_1), self._env_config)
                if hamming_with_n(folding_tmp_2, self._target.dot_bracket) <= hamming_distance:
                    if self.gc_diff_abs_primary(''.join(primary_tmp_2)) < self.gc_diff_abs(self._design):
                        self._design.assign_sites(3, site, self._target.get_paired_site(site))  # assign C/CG <=> action 3


    def _decrease_gc(self, site, paired_site=None, hamming_distance=0):
        """TODO
        """
        # create two temporary primaries to check for gc-improvement (two because we swap G/C with A and U)
        primary_tmp_1 = list(self._design.primary)
        primary_tmp_2 = list(self._design.primary)
        # if current site is A or U replace with G and C (and watson-crick pair base if site is paired)
        if self._design.primary[site] in ['G', 'C']:
            # assign G and C separately
            primary_tmp_1[site] = 'A'
            primary_tmp_2[site] = 'U'
            # and assign paired site if nucleotide is paired; paired site is not restricted in sequence space, otherwise site would already be restricted
            if paired_site:
                primary_tmp_1[paired_site] = 'U'
                primary_tmp_2[paired_site] = 'A'

            # fold first tmp candidate
            folding_tmp_1 = _fold_primary(''.join(primary_tmp_1), self._env_config)

            # check first if hamming distance is at least the same for tmp candidate 1 (we do not want to increase hamming distance)
            if hamming_with_n(folding_tmp_1, self._target.dot_bracket) <= hamming_distance:
                # if hamming is ok, check if gc content would be better (abs diff)
                if self.gc_diff_abs_primary(''.join(primary_tmp_1)) < self.gc_diff_abs(self._design):
                    # if gc is better, asssign site to the design
                    self._design.assign_sites(1, site, self._target.get_paired_site(site))  # assign A/AU <=> action 1
            else:
            # do the same for candidate 2 if candidate 1 failed
                folding_tmp_2 = _fold_primary(''.join(primary_tmp_1), self._env_config)
                if hamming_with_n(folding_tmp_2, self._target.dot_bracket) <= hamming_distance:
                    if self.gc_diff_abs_primary(''.join(primary_tmp_2)) < self.gc_diff_abs(self._design):
                        self._design.assign_sites(2, site, self._target.get_paired_site(site))  # assign U/UA <=> action 2


class _Target(object):
    """TODO
    Class of the target structure. Provides encodings and id.
    """

    _id_counter = 0

    def __init__(self, input, env_config):
        """TODO: rename dot_bracket input
        Initialize a target structure.

        Args:
             dot_bracket: dot_bracket encoded target structure.
             env_config: The environment configuration.
        """
        _Target._id_counter += 1
        self._env_config = env_config
        self.rng = np.random.default_rng(seed=env_config.seed)

        assert len(input) == 5, "Input does not contain enough information, require: (id, structure constraints, sequence constraints, GC content, desired energy)"

        self.id = input[0]
        # dot bracket may include extension symbols
        self.dot_bracket_X_int = input[1]

        # get all explicit X and O extensions and replace by X/O to get correct sequence constraints later
        self.X_plicits = {}

        # if there are any specific extensions given
        if re.search(r'X\d+|O\d+', self.dot_bracket_X_int):
            # loop until none are left
            while re.search(r'X\d+|O\d+', self.dot_bracket_X_int):
                # find first occurence of extension
                m = re.search(r'X\d+|O\d+', self.dot_bracket_X_int)
                # get the length of the extension
                r = int(m.group(0)[1:])
                # save the substitution and the index to insert later
                if self.dot_bracket_X_int[m.start()] == 'X':
                    sub = 'N' * r
                    self.X_plicits[m.start()] = sub
                    # replace the Xinteger match with X only to get correct sequence constraints
                    self.dot_bracket_X_int = re.sub('X\d+', 'X', self.dot_bracket_X_int, count=1)
                else:
                    sub = '.' * r
                    self.X_plicits[m.start()] = sub
                    # replace the Xinteger match with X only to get correct sequence constraints
                    self.dot_bracket_X_int = re.sub('O\d+', 'O', self.dot_bracket_X_int, count=1)

        # sequence constraints (if given) should not contain extension symbol
        seq = list(input[2]) if input[2] != '-' else []
        # gc content required if provided with file (TODO: set control_gc to true if provided and capture if not provided)
        self.gc = input[3] if not self._env_config.desired_gc else self._env_config.desired_gc
        # mfe currently not used
        self.mfe = input[4]
        # generate dot_bracket by removing the Xs and Os (dot bracket still contains int of enforced pairs)
        self.dot_bracket = self.dot_bracket_X_int.upper().replace('X', '').replace('O', '')
        # remove integers from dot_bracket_XO to find indices for variable length insertions
        self.dot_bracket_X = re.sub('\d+', '', self.dot_bracket_X_int)
        # but get all Xs (extend with N) and Os (extend with dots) to insert into sequence constraints
        Xs = [(i, self.dot_bracket_X[i]) for i in self.findall_X('X', self.dot_bracket_X.upper())]
        Os = [(i, self.dot_bracket_X[i]) for i in self.findall_X('O', self.dot_bracket_X.upper())]

        extensions = Xs + Os

        extensions = sorted(extensions, key=lambda x: x[0])

        from collections import defaultdict

        # start getting enforced pairs
        self.enforced_pairs = defaultdict(list)
        # if there are any ints left in the dot bracket (=enforced pairs; matching integer numbers correspond to matching brackets)
        if re.search(r'\d+', self.dot_bracket):
            tmp_db = list(self.dot_bracket)
            # while there are any ints left
            while re.search(r'\d+', self.dot_bracket):
                # find first occurence
                m = re.search(r'\d+', self.dot_bracket)
                # append them matching their int value
                self.enforced_pairs[m[0]].append(m.start() - 1)
                # but remove them from the list to get correct indices for further search
                del tmp_db[m.start():m.end()]
                # and also remove them from dot bracket
                self.dot_bracket = ''.join(tmp_db)

        # at the end check that all enforced pairs are balanced
        assert all([len(x) == 2 for x in self.enforced_pairs.values()]), "Found unbalanced explicit pairs"

        if not seq:
            self.sequence_constraints_X = None
            self.sequence_constraints = None
        else:
            self.sequence_constraints = ''.join(seq)

        if not self.sequence_constraints:
            sequence_constraints = ['N' for _ in self.dot_bracket]
            dot_bracket = [c for c in self.dot_bracket]
            for index, character in enumerate(self.dot_bracket):
                if character in ['A', 'C', 'G', 'U']:
                    sequence_constraints[index] = character
                    dot_bracket[index] = 'N'
            self.sequence_constraints = ''.join(sequence_constraints)
            self.dot_bracket = ''.join(dot_bracket)

        # if there are any extensions in the provided dot bracket
        for index, extension in extensions:
            # insert them into the sequence constraints
            seq[index:index] = extension
        self.sequence_constraints_X = ''.join(seq)
        # but remove them for sequence constraints only string
        self.sequence_constraints = self.sequence_constraints_X.upper().replace('O', '').replace('X', '')

        self._current_site = 0
        if not self.sequence_contains_iupac():

            self._partition = self.get_partition()
            self.encode_partial_pairing()
            self.local_target = self.assign_sequence_constraints()

            self.sequence_progress = self.local_target
            if self._env_config.encoding == 'tuple':
                self.padded_encoding = _encode_tuple(self.dot_bracket, self.sequence_constraints, self._env_config)
            elif self._env_config.encoding == 'multihot' and self._env_config.use_conv:
                self.padded_encoding = multihot(self.dot_bracket, self.sequence_constraints, self._env_config)
            else:
                self.padded_encoding = _encode_dot_bracket(self.local_target, self._env_config)

            if env_config.reward_function == 'structure_only':
                self.structure_parts_encoding = _encode_structure_parts(self.local_target)

        
    def sequence_contains_iupac(self):
        iupac = {
            'R': ['A', 'G'],
            'Y': ['C', 'U'],
            'S': ['G', 'C'],
            'W': ['A', 'U'],
            'K': ['G', 'U'],
            'M': ['A', 'C'],
            'B': ['C', 'G', 'U'],
            'D': ['A', 'G', 'U'],
            'H': ['A', 'C', 'U'],
            'V': ['A', 'C', 'G'],
        }
        for nuc in self.sequence_constraints:
            if nuc in iupac.keys():
                return True
        return False




    def sample_iupac(self, seq):
        if self._env_config.control_gc:
            p_GC = float(self.gc)
            p_AU = 1 - float(self.gc)
        else:
            p_GC = 0.5
            p_AU = 0.5
        

        iupac = {
            'R': [['A', 'G'], [p_AU, p_GC]],
            'Y': [['C', 'U'], [p_GC, p_AU]],
            'S': [['G', 'C'], [.5, .5]],
            'W': [['A', 'U'], [.5, .5]],
            'K': [['G', 'U'], [p_GC, p_AU]],
            'M': [['A', 'C'], [p_AU, p_GC]],
            'B': [['C', 'G', 'U'], [p_GC / 2, p_GC / 2, p_AU]],
            'D': [['A', 'G', 'U'], [p_AU / 2, p_GC, p_AU / 2]],
            'H': [['A', 'C', 'U'], [p_AU / 2, p_GC, p_AU / 2]],
            'V': [['A', 'C', 'G'], [p_AU, p_GC / 2, p_GC / 2]],
        }

        seq_copy = list(seq)
        # print(seq_copy)
        for i, sym in enumerate(seq):
            if sym in iupac.keys():
                # print(sym)
                # print(i)
                seq_copy[i] = np.random.choice(iupac[sym][0], p=iupac[sym][1])  # sampling without seed
                seq_copy[i] = self.rng.choice(iupac[sym][0], p=iupac[sym][1])  # sampling with seed
                # print(seq_copy[i])
        # print(''.join(seq_copy))
        return ''.join(seq_copy)


    def findall_X(self, pattern, s):
        """TODO
        """
        i = s.find(pattern)
        while i != -1:
            yield i
            i = s.find(pattern, i+1)

    def sample_random_target(self, min_length=0, max_length=0, min_dot_extension=0):
        """TODO
        """
        if not 'X' in self.dot_bracket_X and not 'O' in self.dot_bracket_X:  # no X to extend
            return _Target((self.id, self.dot_bracket_X, self.sample_iupac(self.sequence_constraints), self.gc, self.mfe), self._env_config)

        # make lists from dot-bracket inclusing Xs, Os, and enforced pairs to extend the strings
        db_list = list(self.dot_bracket_X_int)
        seq = list(self.sequence_constraints_X)

        # insert explicit length inserts
        x_plicit_extensions = []

        for index, value in enumerate(db_list):
            if value in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                seq[index:index] = value

        for index, sub in self.X_plicits.items():
            index += int(np.sum(x_plicit_extensions))
            seq_sub = 'N' * len(sub)
            db_list[index:index] = sub
            seq[index:index] = seq_sub
            x_plicit_extensions.append(len(sub))

        seq = re.sub('\d+', '', ''.join(seq))
        seq = list(seq)

        xplicit_extensions_length = np.sum(x_plicit_extensions)

        num_Os = self.dot_bracket_X_int.upper().count('O')
        num_Xs_db = self.dot_bracket_X_int.upper().count('X')
        num_Xs_seq = self.sequence_constraints_X.upper().count('X')

        # insert minimum dots at each extension site (Os)
        min_O = ["."] * min_dot_extension
        min_O_N = ["N"] * min_dot_extension

        for i in range(num_Os):
            all_O = [i for i in self.findall_X('O', ''.join(db_list))]
            all_O_seq = [i for i in self.findall_X('O', ''.join(seq))]
            db_list[all_O[i]:all_O[i]] = min_O
            seq[all_O_seq[i]:all_O_seq[i]] = min_O_N


        # compute borders for the extension with Ns
        max_extension = max_length - (num_Os * min_dot_extension) - len(self.dot_bracket) - xplicit_extensions_length
        min_extension = min_length - (len(self.dot_bracket) + (num_Os * min_dot_extension) + xplicit_extensions_length)

        # if minimal length is already covered by current dot bracket length and minimum extensions by dots, set lower bound to zero
        if min_extension < 0:
            min_extension = 0

        if max_extension <= 0:
            # print("Reached extension limit before extension of Xs.")
            dot_bracket = (self.id, ''.join(db_list), self.sample_iupac(''.join(seq).upper().replace('X', '').replace('O', '')), self.gc, self.mfe)
            return _Target(dot_bracket, self._env_config)

        if 'X' in self.dot_bracket_X:
            # select random insert length for Xs
            # random_insert_length = np.random.randint(min_extension, max_extension)  # sampling without seed
            random_insert_length = self.rng.integers(min_extension, max_extension, endpoint=True)  # sampling with seed

            # and create list of Ns of length random_insert length to insert into dot_bracket
            substitutions_N = ["N"] * random_insert_length

            # iterate over the possible inserts
            while substitutions_N:
                # find all extension positions in the dot_bracket
                all_X = [i for i in self.findall_X('X', ''.join(db_list))]
                all_X_seq = [i for i in self.findall_X('X', ''.join(seq))]

                # get an insertion position
                # insertion_index = np.random.randint(0, len(all_X))  # sampling without seed
                insertion_index = self.rng.integers(0, len(all_X))  # sampling with seed
                insertion_position_db = all_X[insertion_index]
                insertion_position_seq = all_X_seq[insertion_index]

                # get a length of the substitution to insert
                # substitution_len = np.random.randint(0, len(substitutions_N) + 1)  # sampling without a seed
                substitution_len = self.rng.integers(0, len(substitutions_N), endpoint=True)  # sampling with seed

                # get the substitution from the list of Ns
                # substitution = np.random.choice(substitutions_N, substitution_len)  # sampling without seed
                substitution = self.rng.choice(substitutions_N, substitution_len)  # sampling without seed

                # substitute in seq and db and update list of possible inserts
                db_list[insertion_position_db:insertion_position_db] = substitution
                seq[insertion_position_seq:insertion_position_seq] = substitution

                substitutions_N = substitutions_N[substitution_len:]

        if not 'O' in self.dot_bracket_X:
            dot_bracket = (self.id, ''.join(db_list), self.sample_iupac(''.join(seq).upper().replace('X', '').replace('O', '')), self.gc, self.mfe)
            return _Target(dot_bracket, self._env_config)


        # get current dot-bracket to calculate the remaining maximum insertion length
        tmp_db = re.sub('\d+', '', ''.join(db_list).replace('X', '').replace('O', ''))

        # Calculate min and max extensions from the given length and already placed insertions...
        min_O_extension = min_length - len(tmp_db)
        max_O_extension = max_length - len(tmp_db)

        if min_O_extension < 0:
            min_O_extension = 0

        if max_extension <= 0:
            # print("Reached extension limit before extension of Os.")
            dot_bracket = (self.id, ''.join(db_list), self.sample_iupac(''.join(seq).upper().replace('X', '').replace('O', '')), self.gc, self.mfe)
            return _Target(dot_bracket, self._env_config)

        # select random insert length for Os
        # random_O_insert_length = np.random.randint(min_O_extension, max_O_extension)  # sampling without seed
        random_O_insert_length = self.rng.integers(min_O_extension, max_O_extension, endpoint=True)  # sampling with seed

        # and create list of dots of length random_O_insert_length to insert into dot_bracket
        substitutions_O = ["."] * random_O_insert_length
        substitutions_O_seq = ["N"] * random_O_insert_length

        # iterate over the possible inserts
        while substitutions_O:
            # update indices
            all_O = [i for i in self.findall_X('O', ''.join(db_list))]
            all_O_seq = [i for i in self.findall_X('O', ''.join(seq))]

            # get an insertion position
            # insertion_index = np.random.randint(0, len(all_O))  # sampling without seed
            insertion_index = self.rng.integers(0, len(all_O))  # sampling with seed
            insertion_position_db = all_O[insertion_index]
            insertion_position_seq = all_O_seq[insertion_index]

            # get a length of the substitution to insert
            # substitution_len = np.random.randint(0, len(substitutions_O) + 1)  # sampling without seed
            substitution_len = self.rng.integers(0, len(substitutions_O), endpoint=True)  # sampling with seed

            # get the substitution from the list of Ns
            # substitution = np.random.choice(substitutions_O, substitution_len)  # sampling without seed
            substitution = self.rng.choice(substitutions_O, substitution_len)  # sampling with seed
            # substitution_seq = np.random.choice(substitutions_O_seq, substitution_len)  # sampling without seed
            substitution_seq = self.rng.choice(substitutions_O_seq, substitution_len)  # sampling with seed

            # substitute in seq and db and update list of possible inserts
            db_list[insertion_position_db:insertion_position_db] = substitution
            seq[insertion_position_seq:insertion_position_seq] = substitution_seq
            substitutions_O = substitutions_O[substitution_len:]

        # create new input for new target
        dot_bracket = (self.id, ''.join(db_list), self.sample_iupac(''.join(seq).upper().replace('X', '').replace('O', '')), self.gc, self.mfe)

        # and return the target
        return _Target(dot_bracket, self._env_config)

    def encode_partial_pairing(self):
        """TODO
        """
        self._pairing_encoding = [None] * len(self.dot_bracket)

        for start, end in self.partition[1]:
            encoding = _encode_pairing(self.dot_bracket[start:end])
            for index, site in enumerate(encoding):
                if site is not None:
                    self._pairing_encoding[index + start] = site + start

        for enforced_pair in self.enforced_pairs.values():
            self._pairing_encoding[enforced_pair[0]] = enforced_pair[1]
            self._pairing_encoding[enforced_pair[1]] = enforced_pair[0]

    def tmp_generate_sequence_constraints(self):
        sequence_constraints = []
        for site in self.local_target:
            if site in ['A', 'C', 'G', 'U']:
                sequence_constraints.append(site)
            else:
                sequence_constraints.append('N')
        return ''.join(sequence_constraints)

    def assign_sequence_constraints(self):
        """
        Generate/Update the local target representation.
        """
        pair_assignments = {'A': 'U', 'U': 'A', 'G': 'C', 'C': 'G'}

        # get all structure constraints to be partially overwritten by sequence constraints
        new_local_target = [site for site in self.dot_bracket.rstrip()]
        # loop over sequence constraints
        for index, site in enumerate(self.sequence_constraints):
            # if a site is constrained in the sequence space
            if site != 'N':
                # assign it to the local target
                paired_site = self._pairing_encoding[index]
                # and get the paired site if exists
                new_local_target[index] = site
                # if there is a paired site, assign it, too
                if paired_site:
                    # but be careful not to overwrite existing sequence constraints that might differ from watson-crick scheme
                    if not self.sequence_constraints[paired_site] in ['A', 'C', 'G', 'U']:
                        new_local_target[paired_site] = pair_assignments[site]
        return ''.join(new_local_target)

    def get_partition(self):
        """
        Partition the inputs into sequence and structure parts (removing Ns).
        """
        sequence_pattern = re.compile(r"[A, C, G, U]+")
        structure_pattern = re.compile(r"[0, 1, 2]+")

        sequence_parts = []
        structure_parts = []

        current_index = 0

        for pattern in sequence_pattern.findall(self.sequence_constraints):
            start, end = re.search(pattern, self.sequence_constraints[current_index:]).span()
            sequence_parts.append((start + current_index, end + current_index))
            current_index += end

        current_index = 0

        tmp_db = [x for x in self.dot_bracket.rstrip()]

        for index, site in enumerate(self.dot_bracket.rstrip()):
            if site not in ['N', 'A', 'C', 'G', 'U']:  # ACGU is quite useless, isn't it? dot_bracket should not contain sequence parts?
                if site == '.':
                    tmp_db[index] = '0'
                elif site == '(':
                    tmp_db[index] = '1'
                else:
                    tmp_db[index] = '2'

        tmp_db = ''.join(tmp_db)

        for pattern in structure_pattern.findall(tmp_db):
            start, end = re.search(pattern, tmp_db[current_index:]).span()
            structure_parts.append((start + current_index, end + current_index))
            current_index += end

        return sequence_parts, structure_parts

    def __len__(self):
        return len(self.dot_bracket)

    def get_paired_site(self, site):
        """
        Get the paired site for <site> (base pair).

        Args:
            site: The site to check the pairing site for.

        Returns:
            The site that pairs with <site> if exists.TODO
        """
        return self._pairing_encoding[site]

    def reset(self):
        """
        Reset the constraints after a complete episode.
        Required for sequence_progress state since we assign nucleotides during the episode.
        """
        # reset the constraints
        self.sequence_progress = self.local_target if self._env_config.local_design else self.dot_bracket
        # and the encoding
        if self._env_config.encoding == "multihot" and self._env_config.use_conv:
            self.padded_encoding = multihot(self.dot_bracket, self.sequence_constraints, self._env_config)
        elif self._env_config.encoding == "tuple":
            self.padded_encoding = _encode_tuple(self.dot_bracket, self.sequence_constraints, self._env_config)
        else:
            self.padded_encoding = _encode_dot_bracket(self.dot_bracket, self._env_config) if not self._env_config.local_design else _encode_dot_bracket(self.local_target, self._env_config)

    def assign_sites(self, index, value, paired_site):
        """
        Assign sites during the episode if state is sequence_progress.

        """
        new_local_target = list(self.sequence_progress)
        if paired_site:
            new_local_target[index[0]] = value[0]
            new_local_target[index[1]] = value[1]
        else:
            new_local_target[index] = value
        self.sequence_progress = ''.join(new_local_target)
        if self._env_config.encoding == "multihot" and self._env_config.use_conv:
            self.padded_encoding = multihot(self.dot_bracket, self.sequence_progress, self._env_config)
            # print(self.padded_encoding)
        elif self._env_config.encoding == "tuple":
            self.padded_encoding = _encode_tuple(self.dot_bracket, self.sequence_constraints, self._env_config)
        else:
            self.padded_encoding = _encode_dot_bracket(self.sequence_progress, self._env_config)


    def reset_counter(self):
        self._current_site = 0
        self.structure_parts_encoding = _encode_structure_parts(self.local_target)

    @property
    def partition(self):
        return self._partition

    @property
    def next_structure_site(self):
        try:
            while self.structure_parts_encoding[self._current_site] is None:
                self._current_site += 1
            self.structure_parts_encoding[self._current_site] = None
            if self._env_config.predict_pairs and self.get_paired_site(self._current_site):
                self.structure_parts_encoding[self.get_paired_site(self._current_site)] = None
            return self._current_site
        except IndexError:
            return None

    @property
    def current_site(self):
        return self._current_site


class _Design(object):
    """
    Class of the designed candidate solution.
    """

    action_to_base = {0: "G", 1: "A", 2: "U", 3: "C", 4:'_'}
    action_to_pair = {0: "GC", 1: "AU", 2: "UA", 3: "CG"}

    def __init__(self, length=None, primary=None):
        """
        Initialize a candidate solution.

        Args:
            length: The length of the candidate solution.
            primary: The sequence of the candidate solution.
        """
        if primary:
            self._primary_list = primary
        else:
            self._primary_list = [None] * length
        self._dot_bracket = None
        self._current_site = 0
        self._last_assignment = None

    def get_mutated(self, mutations, sites):
        """
        Locally change the candidate solution.

        Args:
            mutations: Possible mutations for the specified sites
            sites: The sites to be mutated

        Returns:
            A Design object with the mutated candidate solution.
        """
        mutatedprimary = self._primary_list.copy()
        for site, mutation in zip(sites, mutations):
            mutatedprimary[site] = mutation
        return _Design(primary=mutatedprimary)

    def assign_sites(self, action, site, paired_site=None, predict_pairs=False):
        """
        Assign nucleotides to sites for designing a candidate solution.

        Args:
            action: The agents action to assign a nucleotide.
            site: The site to which the nucleotide is assigned to.
            paired_site: defines if the site has a corresponding pairing partner or not.
        """
        
        self._current_site += 1
        
        if predict_pairs and paired_site:
            base_current, base_paired = self.action_to_pair[action]
            self._primary_list[site] = base_current
            self._primary_list[paired_site] = base_paired
            self._last_assignment = ((site, paired_site), (base_current, base_paired), True)
        else:
            self._primary_list[site] = self.action_to_base[action]
            self._last_assignment = (site, self.action_to_base[action], False)



    @property
    def last_assignment(self):
        return self._last_assignment

    @property
    def first_unassigned_site(self):
        try:
            while self._primary_list[self._current_site] is not None:
                self._current_site += 1
            return self._current_site
        except IndexError:
            return None

    @property
    def primary(self):
        return "".join(self._primary_list)


def _random_epoch_gen(data):
    """
    Generator to get epoch data.

    Args:
        data: The targets of the epoch
    """
    while True:
        for i in np.random.permutation(len(data)):
            yield data[i]

def _sorted_data_gen(data):
    data = sorted(data, key=lambda x: len(x))
    while True:
        for target in data:
            yield target


def hamming_with_n(s1, s2):
    distance = 0
    for c1, c2 in zip(s1, s2):
        if c1 != c2:
            if c1 == 'N' or c2 == 'N':
                continue
            elif c1 == '|' and c2 in ['(', ')', '[', ']', '{', '}', '<', '>']:
                continue
            elif c2 == '|' and c1 in ['(', ')', '[', ']', '{', '}', '<', '>']:
                continue
            else:
                distance += 1
    return distance

def change_encoding(structure):
    structure = ["(" if x == "(0" else x for x in structure]
    structure = [")" if x == ")0" else x for x in structure]
    structure = ["[" if x == "(1" else x for x in structure]
    structure = ["]" if x == ")1" else x for x in structure]
    structure = ["{" if x == "(2" else x for x in structure]
    structure = ["}" if x == ")2" else x for x in structure]
    structure = ["<" if x == "(3" else x for x in structure]
    structure = [">" if x == ")3" else x for x in structure]
    return structure


@dataclass
class EpisodeInfo:
    """
    Information class.
    """

    __slots__ = ["target_id", "time", "normalized_hamming_distance", "gc_content", "agent_gc", "desired_gc", "delta_gc", "gc_satisfied", "folding", "folding_counter", "candidate", "interactions", "target"]
    target_id: int
    time: float
    normalized_hamming_distance: float
    folding: str
    folding_counter: int
    candidate: str
    gc_content: float
    agent_gc: float
    desired_gc: float
    delta_gc: float
    gc_satisfied: bool
    interactions: int
    target: int

def stockholm2records(sto_file):
    align = AlignIO.read(sto_file, "stockholm")
    return align


def stockholm2df(sto_file):
    align = stockholm2records(sto_file)
    d_list = [{'Id': r.name, 'sequence': r.seq} for r in align]
    return pd.DataFrame(d_list)

def stockholm2idlist(sto_file):
    align = stockholm2records(sto_file)
    return [r.name for r in align]

def read_e_values(tab_file):
    with open(tab_file) as f:
        lines = f.readlines()
    header = [x.strip().replace('#', '') for x in (lines[0].split('\t'))]
    data = []
    for line in lines[1:]:
        line = line.strip()
        if line.startswith('# Program:'):
            continue
        elif line.startswith('# Version:'):
            continue
        elif line.startswith('# Pipeline mode:'):
            continue
        elif line.startswith('# Query file:'):
            continue
        elif line.startswith('# Target file:'):
            continue
        elif line.startswith('# Option settings:'):
            continue
        elif line.startswith('# Current dir:'):
            continue
        elif line.startswith('# Date:'):
            continue
        elif line.startswith('# [ok]'):
            continue
        elif line.startswith('#----'):
            continue
        else:
            line_list = [x.replace('#', '') for x in line.split()]
            if line_list and line_list[0] != '':
                data.append(line_list)
    # print(data)
    # print(header)
    if data:
        df = pd.DataFrame(data, columns=['target name', 'accession', 'query name', 'accession', 'mdl', 'mdl from', 'mdl to', 'seq from', 'seq to', 'strand', 'trunc', 'pass', 'gc', 'bias', 'score', 'E-value', 'inc', 'description of target'])
        return df
    else:
        return pd.DataFrame(columns=['target name', 'accession', 'query name', 'accession', 'mdl', 'mdl from', 'mdl to', 'seq from', 'seq to', 'strand', 'trunc', 'pass', 'gc', 'bias', 'score', 'E-value', 'inc', 'description of target'])


class Infernal():
    def __init__(self,
                 working_dir: str = 'working_dir',
                 E: float = 0.001,
                 incE: float = 0.001,
                 aln_outfile = 'infernal.tab',
                 ):
        self.working_dir = str(Path(working_dir).resolve())
        self.aln = str(Path(working_dir, aln_outfile).resolve())
        self.E = E
        self.incE = incE
        # print(self.E, self.incE)

    def search_database(self,
                        cm_database: str,
                        identifier: str,
                        fasta_db: str,
                        ) -> list:

        call1 = ["echo", str(identifier)]
        call2 = ["cmfetch", "-f", f"{cm_database}", "-"]
        call3 = ["cmsearch", "--tblout", f"{self.aln}", "--nohmmonly", "-E", f"{self.E}", "-", f"{fasta_db}"]  # "-A", f"{self.aln}", "--incE", f"{self.incE}",  "-T", "30.00", "-Z", "742849.287494",

        ps1 = subprocess.Popen(call1, stdout=subprocess.PIPE)
        ps2 = subprocess.Popen(call2, stdin=ps1.stdout, stdout=subprocess.PIPE)
        subprocess.run(call3, stdin=ps2.stdout, stdout=subprocess.DEVNULL)

        # hit_ids = stockholm2idlist(self.aln)
        hits = read_e_values(self.aln)

        return hits

    def get_family_information(self,
                               queries_fasta_path : str,
                               cm_path : str,
                               clanin_path : str,
                               outpath : str,
                               ):
        # subprocess.call(["cmpress", cm_path])
        subprocess.call(["cmscan", "--rfam", "--cut_ga", "--nohmmonly", "--oskip", "--tblout", outpath, "--fmt", "2", "--clanin", clanin_path, cm_path, queries_fasta_path])



    def search(self, cm_path: str, fasta_path: str, outfile: str = None):
        if outfile:
            subprocess.call(["cmsearch", "-A", f"{outfile}", "-E", f"{self.E}", "--incE", f"{self.incE}", cm_path, fasta_path])
            try:
                hit_ids = stockholm2idlist(outfile)
            except ValueError as e:
                print('Note:', 'No hits satisfy inclusion thresholds; no alignment saved')
                hit_ids = []
            except FileNotFoundError as e:
                print('Note:', 'no alignment file found')
                hit_ids = []
        else:
            subprocess.call(["cmsearch", "-A", f"{self.aln}", "-E", f"{self.E}", "--incE", f"{self.incE}", cm_path, fasta_path])
            try:
                hit_ids = stockholm2idlist(self.aln)
            except ValueError as e:
                print('Note:', 'No hits satisfy inclusion thresholds; no alignment saved')
                hit_ids = []
            except FileNotFoundError as e:
                print('Note:', 'no alignment file found')
                hit_ids = []

        return hit_ids

    def build_noss(self, stk_path: str, out_stem: str, name: str = '', cm_dir: str = ''):
        if not cm_dir:
            if not name:
                subprocess.call(["cmbuild", "--noss", f"{self.working_dir}/{out_stem}.cm", stk_path])
            else:
                subprocess.call(["cmbuild", "--noss", "-F", "-n", name, f"{self.working_dir}/{out_stem}.cm", stk_path])
        else:
            if not name:
                subprocess.call(["cmbuild", "--noss", f"{cm_dir}/{out_stem}.cm", stk_path])
            else:
                subprocess.call(["cmbuild", "--noss", "-F", "-n", name, f"{cm_dir}/{out_stem}.cm", stk_path])

    def build(self, stk_path: str, out_stem: str, name: str = '', cm_dir: str = ''):
        if not cm_dir:
            if not name:
                subprocess.call(["cmbuild", "-F", f"{self.working_dir}/{out_stem}.cm", stk_path])
            else:
                subprocess.call(["cmbuild", "-F", "-n", name, f"{self.working_dir}/{out_stem}.cm", stk_path])
        else:
            if not name:
                subprocess.call(["cmbuild", "-F", f"{cm_dir}/{out_stem}.cm", stk_path])
            else:
                subprocess.call(["cmbuild", "-F", "-n", name, f"{cm_dir}/{out_stem}.cm", stk_path])


    def calibrate(self, cm_path: str):
        subprocess.call(["cmcalibrate", cm_path])

    def sample_msa(self, cm_path : str, outpath : str, n : int = 1000, seed : int = 42):
        subprocess.call(["cmemit", "-o", outpath, "-a", "-N", str(n), "--seed", str(seed), cm_path])


class RnaDesignEnvironment(Environment):
    """
    The environment for RNA design using deep reinforcement learning.
    """

    def __init__(self, dot_brackets, env_config):
        """TODO
        Initialize an environemnt.

        Args:
            env_config: The configuration of the environment.
        """
        self._env_config = env_config

        # targets = [_Target(dot_bracket, self._env_config) for dot_bracket in dot_brackets]
        targets = [_Target((i, db, seq, gc, en), self._env_config) for i, db, seq, gc, en in dot_brackets]
        # print(targets)
        self._target_gen = _random_epoch_gen(targets)
        if self._env_config.data_type == 'random-sort':
            self._target_gen = _sorted_data_gen(targets)

        self.target = None
        self.design = None
        self._folding = None
        self.episodes_info = []

        # try CM hitting as task
        self.cm_design = env_config.cm_design
        if self.cm_design:
            self.working_dir = env_config.working_dir
            self.max_e = 100
            Path(self.working_dir).mkdir(parents=True, exist_ok=True)
            self.cm_name = env_config.cm_name  # 'RF00651'  # 'RF00162'  # SAM riboswitch; LEN 108, MaxL 270;  RF00651 -> mir-221 CLEN 85
            self.cm_db = env_config.cm_path
            self.infernal = Infernal(working_dir=self.working_dir,
                        E=self.max_e,
                        incE=self.max_e)
        
        self.rri_design = env_config.rri_design
        if self.rri_design:
            self.rri_target = env_config.rri_target

        self.alpha = 1.0
        self.beta = 1.0
        # self.alpha_beta_threshold = 0.02
        # self.gc_look_back = -50

    def __str__(self):
        return "RnaDesignEnvironment"

    def seed(self, seed):
        return None

    def reset(self):
        """
        Reset the environment. First function called by runner. Returns first state.

        Returns:
            The first state.
        """
        self._env_config.interactions += 1
        self.target = next(self._target_gen)

        if not self._env_config.control_gc:
            self._env_config.control_gc = self.target.gc is not None

        # print(self.target.sequence_constraints)

        # get a random target with random extension of Os and Xs in the structure constraints
        # if self._env_config.variable_length:
        self.target = self.target.sample_random_target(self._env_config.min_length, self._env_config.max_length, self._env_config.min_dot_extension)
        # print(self.target.sequence_constraints)

        # inject desired gc-content if provided with env-config, else use gc content provided with file
        if self._env_config.control_gc and self._env_config.desired_gc:
            self.target.gc = self._env_config.desired_gc

        self.target.reset()

        if self._env_config.reward_function == 'structure_only':
            self.target.reset_counter()

        self.design = _Design(len(self.target))

        return self._get_state()

    def _apply_action(self, action):
        """
        Assign a nucleotide to a site.

        Args:
            action: The action chosen by the agent.
        """
        current_site = self.design.first_unassigned_site if not self._env_config.reward_function == 'structure_only' else self.target.current_site
        paired_site = self.target.get_paired_site(current_site) if self._env_config.predict_pairs else None  # None for unpaired sites
        self.design.assign_sites(action, current_site, paired_site, self._env_config.predict_pairs)

    def _get_state(self):
        """
        Get a state dependend on the padded encoding of the target structure.

        Returns:
            The next state.
        """
        start = self.target.next_structure_site if self._env_config.reward_function == 'structure_only' else self.design.first_unassigned_site

        if not self._env_config.state_representation == 'n-gram':
            if self.design.last_assignment:
                index, value, paired_site = self.design.last_assignment
                self.target.assign_sites(index, value, paired_site)

        if self._env_config.encoding == "multihot" and self._env_config.use_conv:
            state = self.target.padded_encoding[start : start + 2 * self._env_config.state_radius + 1, :, :]
            return state

        return self.target.padded_encoding[
            start : start + 2 * self._env_config.state_radius + 1
        ]

    def _get_local_design_loss(self, design):
        # initialize distance
        distance = 0
        # get folding if structure and sequence are evaluated
        folding = _fold_primary(design.primary, self._env_config) if self._env_config.reward_function == 'sequence_and_structure' else None
        #
        sequence_parts, folding_parts = self.target.partition

        if self._env_config.reward_function == 'sequence_and_structure':
            distance += hamming_with_n(design.primary, self.target.sequence_constraints)
        else:
            design = [c for c in design._primary_list]

            for index, site in enumerate(self.target.local_target):
                if site in ['A', 'C', 'G', 'U']:
                    design[index] = site

            self.design = _Design(primary=[c for c in ''.join(design).rstrip()])

            folding = _fold_primary(self.design.primary, self._env_config)
            # print(folding)

        distance += hamming_with_n(folding, self.target.dot_bracket)

        return distance, folding


    def _local_improvement(self, folded_design):
        """
        Compute Hamming distance of locally improved candidate solutions.
        Returns:
            The minimum Hamming distance of all imporved candidate solutions.
        """
        def flatten_list(list_):
            return [item for sublist in list_ for item in sublist]
        # print("enter LIS")

        if self._env_config.local_design:
            sequence_parts, structure_parts = self.target.partition
            differing_sites_per_part = []

            if self._env_config.reward_function == 'sequence_and_structure':
                for start, end in sequence_parts:
                    string_difference = _string_difference_indices(
                        self.target.sequence_constraints[start:end], self.design.primary[start:end]
                    )
                    string_difference = [index + start for index in string_difference]
                    differing_sites_per_part.append(string_difference)

            for start, end in structure_parts:
                string_difference = _string_difference_indices(
                    self.target.dot_bracket[start:end], self._folding[start:end]
                )
                string_difference = [index + start for index in string_difference]
                differing_sites_per_part.append(string_difference)

            differing_sites = flatten_list(differing_sites_per_part)
        else:

            differing_sites = _string_difference_indices(
                self.target.dot_bracket, folded_design
            )
        differing_copy = differing_sites.copy()
        for index in differing_sites:
            # print(index)
            # print(self.target.sequence_constraints[index])
            if self.target.sequence_constraints[index] != 'N':
                # print(differing_sites)
                # print(index)
                # print(self.target.sequence_constraints[index])
                differing_copy.remove(index)
                # print(differing_sites)
        differing_sites = differing_copy
        # print(differing_sites)
        hamming_distances = []
        for mutation in product("AGCU", repeat=len(differing_sites)):
            mutated = self.design.get_mutated(mutation, differing_sites)
            if self._env_config.local_design:
                hamming_distance, _ = self._get_local_design_loss(mutated)
            else:
                folded_mutated = _fold_primary(mutated.primary, self._env_config)
                hamming_distance = hamming_with_n(folded_mutated, self.target.dot_bracket)
            hamming_distances.append((hamming_distance, mutated))
            if hamming_distance == 0:  # For better timing results
                # print(f"in LIS {hamming_with_n(mutated.primary, self.target.sequence_constraints)}")
                # print(_fold_primary(mutated.primary, self._env_config))
                # print(mutated.primary)
                # print(f"in LIS: {self._get_local_design_loss(mutated)}")
                return (0, mutated)
        return min(hamming_distances, key=lambda x: x[0])

    def reward_local_design(self):

        self.gc_controler = None

        distance, self._folding = self._get_local_design_loss(self.design)
        gc_initial = (self.design.primary.upper().count('G') + self.design.primary.upper().count('C')) / len(self.design.primary)
        # print(self._folding)
        # print(f"initial: {(distance, self._folding)}")
        # print(self.design.primary)
        self.gc_controler = GcControler(self.design, self.target, self._env_config.gc_tolerance, self._env_config)
        # print(f"Before LIS {hamming_with_n(self.design.primary, self.target.sequence_constraints)}")

        # distance might get larger than sites in target because of overlapping constraints when algorithm also places sequence
        if distance > len(self.target):
            distance = len(self.target)

        if 0 < distance < self._env_config.mutation_threshold:
            distance, self.design = self._local_improvement(self._folding)
            # print(self._env_config.folding_counter)
            # print(f"after LIS {self._get_local_design_loss(self.design)}")
            # self._folding = _fold_primary(self.design.primary, self._env_config)
            # print(f"After LIS {hamming_with_n(self.design.primary, self.target.sequence_constraints)}")
            if self._env_config.control_gc:
                # print('enter GC-control')

                self.gc_controler = GcControler(self.design, self.target, self._env_config.gc_tolerance, self._env_config)

                self.design = self.gc_controler.gc_improvement_step(hamming_distance=distance)

                self._folding = _fold_primary(self.design.primary, self._env_config)

                distance = hamming_with_n(self._folding, self.target.dot_bracket)
                gc_diff_abs = self.gc_controler.gc_diff_abs(self.design)

                if self.gc_controler.gc_satisfied(self.design):
                    gc_penalty = 0
                else:
                    gc_penalty = gc_diff_abs
                # if not len(self.episodes_info) < self.gc_look_back:
                #     if np.mean([abs(ei.gc_content - self.gc_controler._desired_gc)  for ei in self.episodes_info[self.gc_look_back:]]) < self.alpha_beta_threshold:
                #         self.alpha = min(1.0, self.alpha + 0.02)
                        # self.beta = max(0.0, self.beta - 0.02)
                    # else:
                    #     self.alpha = max(0.0, self.alpha - 0.02)
                    #     self.beta = min(1.0, self.beta + 0.02)

                normalized_distance = (distance / len(self.target))

                general_distance = self.alpha * normalized_distance + self.beta * gc_penalty

                if (general_distance) > 1.0:
                    general_distance = 1.0

                episode_info = EpisodeInfo(
                target_id=self.target.id,
                time=time.time(),
                normalized_hamming_distance=general_distance,
                folding=self._folding,
                folding_counter=self._env_config.folding_counter,
                candidate=self.design.primary,
                agent_gc=gc_initial,
                desired_gc=self.target.gc,
                gc_content=(self.design.primary.upper().count('G') + self.design.primary.upper().count('C')) / len(self.design.primary),
                delta_gc=gc_diff_abs,
                gc_satisfied=self.gc_controler.gc_satisfied(self.design),
                interactions=self._env_config.interactions,
                target=self.target.dot_bracket,
                )
                self.episodes_info.append(episode_info)

                return (1 - (general_distance)) ** self._env_config.reward_exponent

        if not self._env_config.control_gc:
            # print(self._env_config.folding_counter)
            # print(f"final rewarding function: {self._get_local_design_loss(self.design)}")
            self._folding = _fold_primary(self.design.primary, self._env_config)
            normalized_distance = (distance / len(self.target))  # is this buggy?
            # print(normalized_distance)

            episode_info = EpisodeInfo(
            target_id=self.target.id,
            time=time.time(),
            normalized_hamming_distance=normalized_distance,
            folding=self._folding,
            folding_counter=self._env_config.folding_counter,
            candidate=self.design.primary,
            agent_gc=gc_initial,
            desired_gc=self.target.gc,
            gc_content=(self.design.primary.upper().count('G') + self.design.primary.upper().count('C')) / len(self.design.primary),
            delta_gc=None,
            gc_satisfied=self.gc_controler.gc_satisfied(self.design),
            interactions=self._env_config.interactions,
            target=self.target.dot_bracket,
            )
            self.episodes_info.append(episode_info)

            return (1 - normalized_distance) ** self._env_config.reward_exponent
        else:
            gc_diff_abs = self.gc_controler.gc_diff_abs(self.design)

            if self.gc_controler.gc_satisfied(self.design):
                gc_penalty = 0
            else:
                gc_penalty = gc_diff_abs

            normalized_distance = (distance / len(self.target))
            # if not len(self.episodes_info) < self.gc_look_back:
            #     if np.mean([abs(ei.gc_content - self.gc_controler._desired_gc)  for ei in self.episodes_info[self.gc_look_back:]]) < self.alpha_beta_threshold:
            #         self.alpha = min(1.0, self.alpha + 0.02)
                    # self.beta = max(0, self.beta - 0.02)
                # else:
                #     self.alpha = max(0, self.alpha - 0.02)
                #     self.beta = min(1.0, self.beta + 0.02)

            general_distance = self.alpha * normalized_distance + self.beta * gc_penalty

            if (general_distance) > 1.0:
                general_distance = 1.0

            episode_info = EpisodeInfo(
            target_id=self.target.id,
            time=time.time(),
            normalized_hamming_distance=general_distance,
            folding=self._folding,
            folding_counter=self._env_config.folding_counter,
            candidate=self.design.primary,
            agent_gc=gc_initial,
            desired_gc=self.target.gc,
            gc_content=(self.design.primary.upper().count('G') + self.design.primary.upper().count('C')) / len(self.design.primary),
            delta_gc=gc_diff_abs,
            gc_satisfied=self.gc_controler.gc_satisfied(self.design),
            interactions=self._env_config.interactions,
            target=self.target.dot_bracket,
            )
            self.episodes_info.append(episode_info)

            return (1 - (general_distance)) ** self._env_config.reward_exponent

    def _reward_cm_design(self):
        # self.design._primary_list = [x if x is not None else y for x, y in zip(self.design._primary_list, self.target.sequence_constraints)]
        self.design._primary_list = [x if x is not None else y for x, y in zip(self.design._primary_list, self.target.local_target)]
        fasta_path = Path(self.working_dir, 'tmp.fasta')

        with open(f"{fasta_path.resolve()}", 'w') as f:
            f.write(f">{self.target.id}\n{''.join(self.design.primary)}")
        
        

        try:
            hit_df = self.infernal.search_database(cm_database=self.cm_db, identifier=self.cm_name, fasta_db=str(fasta_path.resolve()))
        except ValueError as e:
            print(e)
        except Exception as e:
            print(e)
        if hit_df.empty:
            reward = -200
        else:
            hit_df.loc[:, 'score'] = hit_df['score'].astype(float)
            reward = hit_df['score'].max()
        fasta_path.unlink()

        episode_info = EpisodeInfo(
            target_id=self.target.id,
            time=time.time(),
            normalized_hamming_distance=reward,
            folding=None,
            folding_counter=1,
            candidate=self.design.primary,
            agent_gc=None,
            desired_gc=None,
            gc_content=None,
            delta_gc=None,
            gc_satisfied=None,
            interactions=None,
            target=self.target.dot_bracket,
            )
        self.episodes_info.append(episode_info)

        return reward

    def _reward_rri_design(self):
        # self.design._primary_list = [x if x is not None else y for x, y in zip(self.design._primary_list, self.target.sequence_constraints)]
        self.design._primary_list = [x if x is not None else y for x, y in zip(self.design._primary_list, self.target.local_target)]

        design = ''.join(self.design.primary)

        output = subprocess.check_output(['IntaRNA', self.rri_target, design]) 

        if output.decode('utf-8').split('\n')[-3].startswith('energy: '):
            energy = float(output.decode('utf-8').split('\n')[-3].replace('energy: ', '').replace(' kcal/mol', ''))
            reward = -1 * energy
        else:
            reward = 0

        reward = reward ** 3

        reward = max(0, reward)
        
        episode_info = EpisodeInfo(
        target_id=self.target.id,
        time=time.time(),
        normalized_hamming_distance=reward,
        folding=None,
        folding_counter=1,
        candidate=design,
        agent_gc=None,
        desired_gc=None,
        gc_content=None,
        delta_gc=None,
        gc_satisfied=None,
        interactions=None,
        target=self.target.dot_bracket,
        )
        
        self.episodes_info.append(episode_info)

        return reward

    def _get_reward(self, terminal):
        """
        Compute the reward after assignment of all nucleotides.

        Args:
            terminal: Bool defining if final timestep is reached yet.

        Returns:
            The reward at the terminal timestep or 0 if not at the terminal timestep.
        """
        if not terminal:
            return 0

        if self.cm_design:
            return self._reward_cm_design()
        elif self.rri_design:
            return self._reward_rri_design()


        # reward formulation for RNA local Design, excluding local improvement steps and gc content!!!!
        if self._env_config.local_design:
            return self.reward_local_design()
        else:
            folded_design, _ = fold(self.design.primary)
            hamming_distance = hamming(folded_design, self.target.dot_bracket)
            if 0 < hamming_distance < self._env_config.mutation_threshold:
                hamming_distance = self._local_improvement(folded_design)

            normalized_hamming_distance = hamming_distance / len(self.target)

            # For hparam optimization
            episode_info = EpisodeInfo(
                target_id=self.target.id,
                time=time.time(),
                normalized_hamming_distance=normalized_hamming_distance,
            )
            self.episodes_info.append(episode_info)

            return (1 - normalized_hamming_distance) ** self._env_config.reward_exponent

    def execute(self, actions):
        """
        Execute one interaction of the environment with the agent.

        Args:
            action: Current action of the agent.

        Returns:
            state: The next state for the agent.
            terminal: The signal for end of an episode.
            reward: The reward if at terminal timestep, else 0.
        """
        self._apply_action(actions)

        terminal = self.design.first_unassigned_site is None if not self._env_config.reward_function == 'structure_only' else all([x is None for x in self.target.structure_parts_encoding])
        state = None if terminal else self._get_state()
        reward = self._get_reward(terminal)

        return state, terminal, reward

    def close(self):
        pass

    @property
    def states(self):
        type = "int" if self._env_config.use_embedding else "float"
        if self._env_config.encoding == "multihot" and self._env_config.use_conv:
            return dict(type=type, shape=(1 + 2 * self._env_config.state_radius, 9, 1))
        if self._env_config.use_conv and not self._env_config.use_embedding:
            return dict(type=type, shape=(1 + 2 * self._env_config.state_radius, 1))
        return dict(type=type, shape=(1 + 2 * self._env_config.state_radius,))

    @property
    def actions(self):
        return dict(type="int", num_actions=4)
