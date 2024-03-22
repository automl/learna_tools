args: 
 Namespace(agent=None, algorithm='rnafold', batch_size=247, control_gc=False, conv_channels=[17, 24], conv_sizes=[0, 0], data_dir='data', data_type='random-sort', dataset=None, desired_gc=None, embedding_indices=21, embedding_size=17, encoding='tuple', entropy_regularization=4.464128875341124e-07, fc_units=12, gc_tolerance=0.01, learning_rate=0.000589774576537135, local_design=True, lstm_units=20, max_length=0, min_dot_extension=0, min_length=0, mutation_threshold=5, num_fc_layers=2, num_lstm_layers=0, predict_pairs=True, restart_timeout=None, restore_path=PosixPath('models/350_0_0'), reward_exponent=10.760044595015092, reward_function='structure_only', sequence_constraints='-', state_radius=10, state_representation='n-gram', stop_learning=False, target_structure_ids=None, target_structure_path=PosixPath('data/eterna100_v2/76.rna'), timeout=30, variable_length=False)
3.2210500240325928 0.03942928482323493 0.2595419847328244 CCGCGAAACCGUGGCGGCAACGGCCAAGCGAAGGCGAAGGGAAAACCCGGGGGCAAAGGCGGCCAAGCCCCAAAAGGACCCCCCAAAAGGGAACGCCAGGGGAAGGCAAAAGCCGGGACGAAAGCGCCGAACCGGCCAAAAGGACCCGGGAAAACCCAGCCCCACGGGAUGCGAAAUCGCGGGCCCAAAAGGGGAAAAUCCCGAAAACGACCCCCCAAACGGGAACCCGAGGGCGAGCGAAAACGCGGGGGGAAAACCGCGAAACGCGGUAAACCACCCGGGAAAACCCCAGCCCCGGGCAAGCGAAAACGCGGCGCGAAAACGCCCAAAGGGGAAAAAUCAGCCCGCAAAAGCGAAGCCCAUCGCAAGGCAAUCGCCGCCCCGAAAACGCGG
6.092986822128296 0.045668368281631955 0.24936386768447838 CGGGGAAAACCAGGGCGGAAAACCGACGGGGCGGGGUAGGGAGAACCCCGGACCAAACGGGCCUAAGGCCCAAAAGGCCCGCGGAAAACCGCACCCCAGGGGACCGCAAAAGCGGGCCCCGAAAGGGCGAAACGCGGAAAACCAGCCCCGAAAACGGAACCCCAGCGGAAGGGAAAACCCGGCAGGAAAACCGGCAACGCCGAAAUAUCAGCCGCGAAAACGCAACCGCAGGGCAAGGCAAAAGCCGGCCCGAAAACGCCCAAGGGGGGCUAACCAGCCCGCAAAAGCGAAGCCCCGGCCAGGGGAACACCCCCGCCCAAAAGGACGAAACGUGCAAAAGCACGGGGCAAAAGCCAAGGCCCCCCCUCGCCAAAAGGCCCCCGCAAAGGCCCG
8.826639413833618 0.016922454249958897 0.3155216284987277 CGCGCAAAAGCCCGCCGGAAAACCGAAGGCGCGCGCCAGCCAUAAGGCGGCGACAAAAGUUGGAAACCACCAAAAGGCGCCGGCUCAAGCCAAGCGCACGCGAAGGCAAAAGCCCGCGCGCAAACGGCCAAAGGCGCAAAAGCAGCGCGGACAACCGAACGCGUCUCGACCGGACAACCGGGGCGGAAAACCCCGAAACGGGCAAAAGCGCCCGCGAAAACGCAACGAGUGGCGAUGCCAAAAGGCGGGCCGCAAACGGGGAAACCCGCAAAAGCACCCCCGAAAACGGAACGCCAUGGGAAGGCAAAAGCCCCGCCCUAAAGGGGGAAACCCGCAAAAGCACGGGGCAAAAGCCAACCCAACGCCUAGCGAAAACGCGCGACCAAAAGGGCG
11.613985538482666 0.03397342085730353 0.2697201017811705 CGGGGAAAACCGCGCCGGAAAUCCGAAGCGGCGGGGACCCGAAAACGGCGCAGCAAAAGCGCGAAACGCCCGAAAGGGGCGGCGAAAACGCAACCCCUCGGCAACGCAAAAGCGGGGCGCUAAAGCGGAAAAUCCCCAAAAGGGCCCGCGGAAUCGCAAGCCGUCGCCUAAGGAAAACCUGGCACCAACAGGCCCAACGGGCAAAAAUGAGCCCCGAAAACGGAAGGCGAACGCAAGGCAAAAGCCCCGGGUGAAAACCCGAAACGGCGUAAACGACGGCGGAAACCCGAAGCGUGCCCGAAGCGAAAACGCGGCAGCACAAGCGCCAAAGGCCCAAAUGGAGCCGGCAAAAGCCAACGGGCCCGCAUUCCAAAAGGAGCGCCGAAAACGCCG
14.565218210220337 0.05671611641708641 0.2340966921119593 GGGGGAAAACCAUGGGGGAAAACCCAAGCCCACGGGAUAGGAAAACCUCCCGGCAAAAGCGCUAAAAGCGGAGAACCAGGGGGCAAAAGCCAACCCGUCGCGAGCCGAAAACGGGCCAGGAAAACCGGCGAAGCCGGAAAUCCAGGCGCCAAAAGGCAACGCGACGGGAACGGAAAACCGGCCUCCUAAAGGGGCAAAGCCGCAAAAGCAGGCGCCAAAAGGCCACCCGCCCCCAACCCAAAAGGGCGCAGGAUAACCGCGAAACGCGCCAAAGCAGCGGGGAAAACCCAAGGGGUCGGCUCGCGAAAACGCCCGUGGAAAACCGGCAAAGCCCGAAAACGACGGGCCAAUAGGCAAGCCGAGGGCAGGCCAAAAGGCCCACGGACAACCCCC
17.320369482040405 0.08345168680651055 0.20610687022900764 GGGGGAAAUCCACCGGGCAAAAGCCAAGGGGCGGGCAACGGAAAACCGGGCUCCCAAAGGGGCAAAGCCCGAAAUCGAGCCGCCGAAAGGCAAGCCCAUCGCAAGGUAAACACCCUCUCCAAAUGGGCUAAAAGCCCAACAGGCGAGGGGAAAACCCAAGCGAAUGGCAGGGGAAAACCCCGCCGCAAACGCGCGAAACGCGCAAAAGCAGCGGCGAAAACGCAAGCCACCCGGAGGGCAAAAGCCGCCAGCAAAAGCGGCCAAGCCGCAAAAGCAGGCGGGAAAACCCAACCGGAGCGCAACCGAAAACGGCGCGGGACAACCCGGAAACCGCGAAAACGCGCGGGGAAAACCCAAGCGCACCCCAACCGAAAACGGCGGAGCAUAAGCCCC
20.043084144592285 0.03799554943876545 0.26208651399491095 CCGCCAAAAGGGGGCCGCAAAAGCGAACGCGUGCGCACCCGAAAACGGCGGGGGAAACCCCGCAAAGCGCCAAAAGGCCCGGCCAAAAGGCAAGCGCAGCGGAACGCAAAAGCGGGGGCCAAAAGGCUGAAACAGGAAAAAUCACCCGCGACAACGCAACCGCCGGCGCACGGAAAACCGGCCCGGACAACCGGCAAAGCCCGAAAACGAGGCCGCAAAAGCGACCGCCAGGCGAUGCCAAAAGGCGGUGCCAAAAGGCGCAAAGCGCCAAAAGGCACCCGGACAACCGAACGCCAGGGGAACGCAUAAGCGCCGCCGAAAACGCCCAAAGGGCCAAAAGGCCGGGGGAAAACCCCACCCCACGCGAAGGGAAAACCCGCCAGGAAACCCCGG
22.723077058792114 0.01983531874788362 0.3053435114503817 GCCUCAACAGAACCGGGCAAAAGCCAACGCGCGGGCACCGGAAAACCGGGGAGGAAAACCGGCAAAGCCCGAAAACGACCCGCCCAAAGGCAAGCCCAGGGGAACGCAAAAGCGGGGCCGAAACCGGGCAAAGCCGGUAAGCCACCCCGCAAGAGCGAACCCCCGGGGAAGCCAAAAGGCCCGCCGAAAACGGGGGAACCCGCAAAAGCACGGCGGAAAACCGAACCCCCCCGCCUCGCAAAAGCGGCCCGCAAAAGCGCCAAAGGCCGAAAUCGCGGCGCCAAAAGGCAAGCGGACCCGGACGAACAAUCGCGCAGGAAAACCGCGAAACGCGUAAAAACCGCGCGCAAAAGCGAACGGGGCGCGGCGGCAAAAGCCCGGGGCAAACGCGGC
25.600931644439697 0.011257281951809448 0.34096692111959287 CGGGCAACAGCCGGGCGGAAAACCGUUCCGGUGCCGCCCCCAAAAGGGGCGGGCAAAAGCGGCGACGCCCCAAAAGGCCGCGCGAAAACGCAACGGCCGCGGACCCGAAAACGGCCGUGCCAAAGCCCGAAACGGGCAAAAGCCCGGGGGAAAACCCAACCGCACGCCCCGGCAAAAGCCCUCCGGAAAUCCCGGAAACCGGGAAAACCUGAGGGCAAAAGCCAAGGCGAGCGGAACGGAAAACCGGCCCGGAAAACCCCCGAAGGGCCAAGAGGAGGCCCCAACAGGGAACCGCCCCGGAAGCGAAAACGCGCGAGCAAAAGCGCGAAACGCGGACAUCCACGCGGCAAAAGCCAACCGGACCGGGCGCGAAAACGCCCCACCAAACGGCCG
28.71256971359253 0.06537286517760633 0.22391857506361323 GGGGGUAAACCCGCCGGGAAAGCCCUUGGGGUGCGGAACGGAUAACCGCGGCGCAAAAGCCCCAAGGGGGUAAACACACCGCGGAAAACCGACCCGCCCGGGACGGCAAAAGCCGGGAGGACAACCGCCCAAGGCGCAAAAGCACCCGCGAAAACGCAACCCGCGCUGGGGCCAAAAGGCGCCCCCUAACGGGCCAAAGGCGUAUAAACAGGCGGGAAAACCCCCCAGCUGGGGAUGGCAAAAGCCCCCCCGAAAACGGCGAAACGCGGAAAACCCGGGGGGAUAACCCAACCCCUGGCCAAGCGAAAACGCCGCAGGAAAACCGGCAAAGCCCCAAAAGGCGCGGCGAACACGCAAGGCCCCCCCUAUGCAAAAGCAGGCCCCCAAAGGCCC
31.637420177459717 0.03397342085730353 0.2697201017811705 CCGCCACAAGGACCCGGCUAAAGCCAAGGGGCCGGCCACGGAACACCGCGGGCGAAAACGGGGAAACCCCGAAACCGACCGCGGAAAACCGAAGCCGACGGGUACCGAAAACGGGGGCCGAAAACGGCGAAACGCGCAAUAGCACCCGGGAAAACCCAACCCGUGCCCACGGGAGAACCCGGCGGCAAAAGCGCAAAAUGCGCAAAAGCCGCCGCGAAAACGCAAGGGCGGGGGCACGCAAAAGCGCCGCCGAAAACGGGUAAAACCGCAAAAGCGCGGGGCAAAGGCCAGCCCCACCGCAAGGGAAAACCCGGGCCGAAAUCGGCGAAACGCGGAAGUCCACCCCGCAAAAGCGAGGCGGACCCCAACGCAAAAGCGGGGAGGAAAACCCGG
