import os

os.environ["OMP_NUM_THREADS"] = "1"
# os.environ['KMP_AFFINITY']='compact,1,0'

import multiprocessing

import numpy as np
import ConfigSpace as CS
from pathlib import Path
from hpbandster.core.worker import Worker


from src.learna.agent import NetworkConfig, get_network, AgentConfig
from src.learna.environment import RnaDesignEnvironment, RnaDesignEnvironmentConfig
from src.learna.design_rna import design_rna
from src.data.parse_dot_brackets import parse_dot_brackets, parse_local_design_data


class LearnaWorker(Worker):
    def __init__(self, data_dir, num_cores, train_sequences, **kwargs):
        super().__init__(**kwargs)
        self.num_cores = num_cores
        self._data_dir = data_dir
        self.sequence_ids = train_sequences
        # self.train_sequences = parse_dot_brackets(
        #     dataset="rfam_local_validation",
        #     data_dir=data_dir,
        #     target_structure_ids=train_sequences,
        # )

    def compute(self, config, budget, **kwargs):
        """
		Parameters
		----------
			budget: float
				cutoff for the agent on a single sequence
		"""

        config = self._fill_config(config)

        if config["local_design"]:
            self.train_sequences = parse_local_design_data(
            dataset=Path(self._data_dir, "rfam_local_validation").stem,
            data_dir=self._data_dir,
            target_structure_ids=self.sequence_ids,
            )
        else:
            self.train_sequences = parse_dot_brackets(
                dataset=Path(self._data_dir, "rfam_local_validation").stem,
                data_dir=self._data_dir,
                target_structure_ids=self.sequence_ids,
            )

        network_config = NetworkConfig(
            conv_sizes=[config["conv_size1"], config["conv_size2"]],
            conv_channels=[config["conv_channels1"], config["conv_channels2"]],
            num_fc_layers=config["num_fc_layers"],
            fc_units=config["fc_units"],
            num_lstm_layers=config["num_lstm_layers"],
            lstm_units=config["lstm_units"],
            embedding_size=config["embedding_size"],
        )

        agent_config = AgentConfig(
            learning_rate=config["learning_rate"],
            batch_size=config["batch_size"],
            entropy_regularization=config["entropy_regularization"],
        )

        env_config = RnaDesignEnvironmentConfig(
            reward_exponent=config["reward_exponent"],
            state_radius=config["state_radius"],
            local_design=config["local_design"],
            # sequence_reward=bool(config["sequence_reward"]),
            reward_function=config["reward_function"],
            predict_pairs=config["predict_pairs"],
            state_representation=config["state_representation"],
            data_type=config["data_type"],
            # gc_weight=config["gc_weight"],
            # structural_weight=config["structural_weight"],
            # desired_gc=config["desired_gc"],
            # gc_improvement_step=True,
            # gc_postprocessing=False,
        )

        validation_info = self._evaluate(
            budget, config["restart_timeout"], network_config, agent_config, env_config
        )

        return {
            "loss": validation_info["num_unsolved"],
            # "loss": validation_info["sum_of_min_distances"],
            # "loss": validation_info["sum_of_min_deltas_and_distances"],
            "info": {"validation_info": validation_info},
        }

    def _evaluate(
        self,
        evaluation_timeout,
        restart_timeout,
        network_config,
        agent_config,
        env_config,
    ):

        evaluation_arguments = [
            [
                [train_sequence],
                evaluation_timeout,  # timeout
                None,  # restore_path
                False,  # stop_learning
                restart_timeout,  # restart_timeout
                network_config,
                agent_config,
                env_config,
            ]
            for train_sequence in self.train_sequences
        ]

        with multiprocessing.Pool(self.num_cores) as pool:
            evaluation_results = pool.starmap(design_rna, evaluation_arguments)

        evaluation_sequence_infos = {}
        evaluation_sum_of_min_distances = 0
        evaluation_sum_of_first_distances = 0
        evaluation_num_solved = 0
        evaluation_num_unsolved = 0
        # evaluation_sum_of_min_gc_deltas = 0
        # evaluation_sum_of_min_gc_deltas_and_distances = 0

        for r in evaluation_results:
            sequence_id = r[0].target_id
            r.sort(key=lambda e: e.time)

            times = np.array(list(map(lambda e: e.time, r)))
            dists = np.array(list(map(lambda e: e.normalized_hamming_distance, r)))
            # deltas_gc = np.array(list(map(lambda e: e.delta_gc, r)))
            # deltas_and_distances = np.array(list(map(lambda e: e.delta_gc + e.normalized_hamming_distance, r)))

            evaluation_sum_of_min_distances += dists.min()
            evaluation_sum_of_first_distances += dists[0]

            # evaluation_sum_of_min_gc_deltas += deltas_gc.min()
            # evaluation_sum_of_min_gc_deltas_and_distances += deltas_and_distances.min()

            evaluation_num_solved += dists.min() == 0.0
            evaluation_num_unsolved += dists.min() != 0

            evaluation_sequence_infos[sequence_id] = {
                "num_episodes": len(r),
                "mean_time_per_episode": float((times[1:] - times[:-1]).mean()),
                "min_distance": float(dists.min()),
                "last_distance": float(dists[-1]),
                # "min_delta_gc": float(deltas_gc.min()),
                # "last_delta_gc": float(deltas_gc[-1]),
                # "min_deltas_and_distances": float(deltas_and_distances.min()),
                # "last_deltas_and_distances": float(deltas_and_distances[-1]),
            }

        evaluation_info = {
            "num_solved": int(evaluation_num_solved),
            "num_unsolved": int(evaluation_num_unsolved),
            "sum_of_min_distances": float(evaluation_sum_of_min_distances),
            "sum_of_first_distances": float(evaluation_sum_of_first_distances),
            # "sum_of_min_gc_deltas": float(evaluation_sum_of_min_gc_deltas),
            # "sum_of_min_deltas_and_distances": float(evaluation_sum_of_min_gc_deltas_and_distances),
            "squence_infos": evaluation_sequence_infos,
        }

        return evaluation_info

    @staticmethod
    def get_configspace():
        config_space = CS.ConfigurationSpace()

        # parameters for PPO here
        config_space.add_hyperparameter(
            CS.UniformFloatHyperparameter(
                "learning_rate", lower=1e-6, upper=1e-3, log=True, default_value=5e-4  # FR: changed learning rate lower from 1e-5 to 1e-6, ICLR: Learna (5,99e-4), Meta-LEARNA (6.44e-5)
            )
        )
        config_space.add_hyperparameter(
            CS.UniformIntegerHyperparameter(
                "batch_size", lower=32, upper=256, log=True, default_value=32  # FR: changed batch size upper from 128 to 256, configs from ICLR used 126 (LEARNA) and 123 (Meta-LEARNA)
            )
        )
        config_space.add_hyperparameter(
            CS.UniformFloatHyperparameter(
                "entropy_regularization",
                lower=1e-7,  # FR: changed entropy regularization lower from 1e-5 to 1e-7, ICLR: LEARNA (6,76e-5), Meta-LEARNA (151e-4)
                upper=1e-2,
                log=True,
                default_value=1.5e-3,
            )
        )

        config_space.add_hyperparameter(
            CS.UniformFloatHyperparameter(
                "reward_exponent", lower=1, upper=12, default_value=1  # FR: changed reward_exponent upper from 10 to 12, ICLR: Learna (9.34), Meta-LEARNA (8.93)
            )
        )

        config_space.add_hyperparameter(
            CS.UniformFloatHyperparameter(
                "state_radius_relative", lower=0, upper=1, default_value=0
            )
        )

        # parameters for the architecture
        config_space.add_hyperparameter(
            CS.UniformIntegerHyperparameter(
                "conv_radius1", lower=0, upper=8, default_value=1
            )
        )
        config_space.add_hyperparameter(
            CS.UniformIntegerHyperparameter(
                "conv_channels1", lower=1, upper=32, log=True, default_value=32
            )
        )

        config_space.add_hyperparameter(
            CS.UniformIntegerHyperparameter(
                "conv_radius2", lower=0, upper=4, default_value=0
            )
        )
        config_space.add_hyperparameter(
            CS.UniformIntegerHyperparameter(
                "conv_channels2", lower=1, upper=32, log=True, default_value=1
            )
        )

        config_space.add_hyperparameter(
            CS.UniformIntegerHyperparameter(
                "num_fc_layers", lower=1, upper=2, default_value=2
            )
        )
        config_space.add_hyperparameter(
            CS.UniformIntegerHyperparameter(
                "fc_units", lower=8, upper=64, log=True, default_value=50
            )
        )

        config_space.add_hyperparameter(
            CS.UniformIntegerHyperparameter(
                "num_lstm_layers", lower=0, upper=3, default_value=0  # FR: changed lstm layers upper from 2 to 3
            )
        )
        config_space.add_hyperparameter(
            CS.UniformIntegerHyperparameter(
                "lstm_units", lower=1, upper=64, log=True, default_value=1
            )
        )

        config_space.add_hyperparameter(
            CS.UniformIntegerHyperparameter(
                "embedding_size", lower=0, upper=8, default_value=1  # FR: changed embedding size upper from 4 to 8
            )
        )

        # config_space.add_hyperparameter(
        #     CS.UniformIntegerHyperparameter(
        #         "sequence_reward", lower=0, upper=1, default_value=0
        #     )
        # )

        config_space.add_hyperparameter(
            CS.CategoricalHyperparameter(
                "reward_function", choices=['sequence_and_structure', 'structure_replace_sequence', 'structure_only']
            )
        )

        config_space.add_hyperparameter(
            CS.CategoricalHyperparameter(
                "state_representation", choices=['n-gram', 'sequence_progress']
            )
        )

        config_space.add_hyperparameter(
            CS.CategoricalHyperparameter(
                "data_type", choices=['random', 'random-sort']
            )
        )


        config_space.add_hyperparameter(
            CS.UniformIntegerHyperparameter(
                "predict_pairs", lower=0, upper=1, default_value=1
            )
        )


        # config_space.add_hyperparameter(
        #     CS.UniformFloatHyperparameter(
        #         "structural_weight", lower=0, upper=1, default_value=1
        #     )
        # )

        # config_space.add_hyperparameter(
        #     CS.UniformFloatHyperparameter(
        #         "gc_weight", lower=0, upper=1, default_value=1
        #     )
        # )


        return config_space

    @staticmethod
    def _fill_config(config):
        config["conv_size1"] = 1 + 2 * config["conv_radius1"]
        if config["conv_radius1"] == 0:
            config["conv_size1"] = 0
        del config["conv_radius1"]

        config["conv_size2"] = 1 + 2 * config["conv_radius2"]
        if config["conv_radius2"] == 0:
            config["conv_size2"] = 0
        del config["conv_radius2"]

        if config["conv_size1"] != 0:
            min_state_radius = config["conv_size1"] + config["conv_size1"] - 1
            max_state_radius = 32  # FR changed max state radius from 32 to 64, ICLR: LEARNA (32), Meta-LEARNA (29)
            config["state_radius"] = int(
                min_state_radius
                + (max_state_radius - min_state_radius) * config["state_radius_relative"]
            )
        else:
            min_state_radius = config["conv_size2"] + config["conv_size2"] - 1
            max_state_radius = 32  # FR changed max state radius from 32 to 64, ICLR: LEARNA (32), Meta-LEARNA (29)
            config["state_radius"] = int(
                min_state_radius
                + (max_state_radius - min_state_radius) * config["state_radius_relative"]
            )
        del config["state_radius_relative"]

        # config["desired_gc"] = np.random.choice([.1, .2, .3, .4, .5, .6, .7, .8, .9])
        config["restart_timeout"] = None
        config["local_design"] = True
        # config["reward_function"] = 'structure_only'
        # config["predict_pairs"] = 1

        return config
