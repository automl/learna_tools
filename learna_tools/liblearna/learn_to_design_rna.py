import multiprocessing
from pathlib import Path

from learna_tools.liblearna.agent import NetworkConfig, get_network, AgentConfig, ppo_agent_kwargs, get_agent
from learna_tools.liblearna.environment import RnaDesignEnvironment, RnaDesignEnvironmentConfig

from learna_tools.tensorforce.threaded_runner import clone_worker_agent, ThreadedRunner


def episode_finished(stats):
    """
    Function called after each episode of the agent (after designing an entire candidate solution).

    Args:
       stats: Statistics to be printed.

    Returns:
       True, meaning to continue running.
    """
    # print(stats)
    return True


def learn_to_design_rna(
    dot_brackets,
    timeout,
    worker_count,
    save_path,
    restore_path,
    network_config,
    agent_config,
    env_config,
):
    """
    Main function for training the agent for RNA design. Instanciate agents and environments
    to run in a threaded runner using asynchronous parallel PPO.

    Args:
        TODO
        timeout: Maximum time to run.
        worker_count: The number of workers to run training on.
        save_path: Path for saving the trained model.
        restore_path: Path to restore saved configurations/models from.
        network_config: The configuration of the network.
        agent_config: The configuration of the agent.
        env_config: The configuration of the environment.

    Returns:
        Information on the episodes.
    """
    env_config.use_conv = any(map(lambda x: x > 1, network_config.conv_sizes))
    env_config.use_embedding = bool(network_config.embedding_size)
    environments = [
        RnaDesignEnvironment(dot_brackets, env_config) for _ in range(worker_count)
    ]

    network = get_network(network_config)
    agent = get_agent(
        environment=environments[0],
        network=network,
        agent_config=agent_config,
        session_config=None,
        restore_path=restore_path,
    )
    agents = clone_worker_agent(
        agent,
        worker_count,
        environments[0],
        network,
        ppo_agent_kwargs(agent_config, session_config=None),
    )
    threaded_runner = ThreadedRunner(agents, environments)
    # Bug in threaded runner requires a summary report
    threaded_runner.run(
        timeout=timeout, episode_finished=episode_finished, summary_report=lambda x: x
    )

    if save_path:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        agent.save_model(directory=save_path.joinpath("last_model"))

    episodes_infos = [environment.episodes_info for environment in environments]
    return episodes_infos


if __name__ == "__main__":
    import argparse
    from ..data.parse_dot_brackets import parse_dot_brackets, parse_local_design_data

    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument("--data_dir", default="data", help="Data directory")
    parser.add_argument("--dataset", type=Path, help="Available: eterna, rfam_taneda")
    parser.add_argument(
        "--target_structure_ids",
        default=None,
        type=int,
        nargs="+",
        help="List of target structure ids to run on",
    )

    # Model
    parser.add_argument("--restore_path", type=Path, help="From where to load model")
    parser.add_argument("--save_path", type=Path, help="Where to save models")

    # Exectuion behaviour
    parser.add_argument("--timeout", type=int, help="Maximum time to run")
    parser.add_argument("--worker_count", type=int, help="Number of threads to use")

    # Hyperparameters
    parser.add_argument("--learning_rate", type=float, help="Learning rate to use")
    parser.add_argument(
        "--mutation_threshold", type=int, help="Enable MUTATION with set threshold"
    )
    parser.add_argument(
        "--reward_exponent", default=1, type=float, help="Exponent for reward shaping"
    )
    parser.add_argument(
        "--state_radius", default=0, type=int, help="Radius around current site"
    )
    parser.add_argument(
        "--conv_sizes", type=int, default=[1, 2], nargs="+", help="Size of conv kernels"
    )
    parser.add_argument(
        "--conv_channels",
        type=int,
        default=[50, 2],
        nargs="+",
        help="Channel size of conv",
    )
    parser.add_argument(
        "--num_fc_layers", type=int, default=2, help="Number of FC layers to use"
    )
    parser.add_argument(
        "--fc_units", type=int, default=50, help="Number of units to use per FC layer"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for ppo agent"
    )
    parser.add_argument(
        "--entropy_regularization", type=float, default=1.5e-3, help="The output entropy"
    )
    parser.add_argument("--embedding_size", type=int, default=0, help="Size of embedding")
    parser.add_argument(
        "--lstm_units", type=int, default=0, help="Number of lstm units in each layer"
    )
    parser.add_argument(
        "--num_lstm_layers", type=int, default=0, help="Number of lstm layers"
    )
    # parser.add_argument("--gc_tolerance", default=0.04, type=float, help="The tolerance of the gc-content")
    # parser.add_argument("--desired_gc", default=0.5, type=float, help="The desired gc-content of the solution")
    # parser.add_argument("--gc_improvement_step", action="store_true", help="Control the gc-content of the solution")
    # parser.add_argument("--gc_postprocessing", action="store_true", help="Control gc-content only via postprocessing")
    # parser.add_argument("--gc_reward", action="store_true", help="Include gc-content into reward function")
    # parser.add_argument("--gc_weight", default=1.0, type=float, help="The weighting factor for the gc-content constraint")
    # parser.add_argument("--structural_weight", default=1.0, type=float, help="The weighting factor for the structural constraint")
    parser.add_argument("--reward_function", type=str, default='structure_only', help="Decide if hamming distance is computed based on the folding only or also on the sequence parts")
    # parser.add_argument("--training_data", default="random", type=str, help="Choose the training data for local design: random sequences, motif based sequences")
    parser.add_argument("--local_design", action="store_true", help="Choose if agent should do RNA local Design")
    parser.add_argument("--predict_pairs", action="store_true", help="Choose if Actions are used to directly predict watson-crick base pairs")
    parser.add_argument("--state_representation", type=str, default='n-gram', help="Choose between n-gram and sequence_progress to show the nucleotides already placed in the state")
    parser.add_argument("--data_type", type=str, default='random', help="Choose type of training data, random motifs or motifs with balanced brackets")
    parser.add_argument("--sequence_constraints", type=str, default='-', help="Perform local design with knowledge about sequence")
    parser.add_argument("--encoding", type=str, default='tuple', help="Choose encoding of the structure and seuqence constraints: 'tuple', 'sequence_bias', 'multihot'. Note: multihot only without embedding, requires conv (will be 2D)")
    parser.add_argument(
        "--embedding_indices", type=int, default=9, help="Number of lstm layers"
    )


    args = parser.parse_args()

    network_config = NetworkConfig(
        conv_sizes=args.conv_sizes,
        conv_channels=args.conv_channels,
        num_fc_layers=args.num_fc_layers,
        fc_units=args.fc_units,
        embedding_size=args.embedding_size,
        lstm_units=args.lstm_units,
        num_lstm_layers=args.num_lstm_layers,
        conv2d=args.encoding == "multihot",
        embedding_indices=args.embedding_indices,
    )
    agent_config = AgentConfig(learning_rate=args.learning_rate)
    env_config = RnaDesignEnvironmentConfig(
        mutation_threshold=args.mutation_threshold,
        reward_exponent=args.reward_exponent,
        state_radius=args.state_radius,
        # gc_tolerance=args.gc_tolerance,
        # desired_gc=args.desired_gc,
        # gc_improvement_step=args.gc_improvement_step,
        # gc_postprocessing=args.gc_postprocessing,
        # gc_weight=args.gc_weight,
        # structural_weight=args.structural_weight,
        # gc_reward=args.gc_reward,
        local_design=args.local_design,
        # num_actions=args.num_actions,
        # keep_sequence=args.keep_sequence,
        # sequence_reward=args.sequence_reward,
        reward_function=args.reward_function,
        predict_pairs=args.predict_pairs,
        state_representation=args.state_representation,
        data_type=args.data_type,
        sequence_constraints=args.sequence_constraints,
        encoding=args.encoding,
        # structure_only=args.structure_only,
        # training_data=args.training_data,
    )
    dot_brackets = parse_dot_brackets(
        dataset=args.dataset,
        data_dir=args.data_dir,
        target_structure_ids=args.target_structure_ids,
    )
    if args.local_design:
        dot_brackets = parse_local_design_data(
            dataset=args.dataset,
            data_dir=args.data_dir,
            target_structure_ids=args.target_structure_ids,
        )

    learn_to_design_rna(
        dot_brackets,
        timeout=args.timeout,
        worker_count=args.worker_count or multiprocessing.cpu_count(),
        save_path=args.save_path,
        restore_path=args.restore_path,
        network_config=network_config,
        agent_config=agent_config,
        env_config=env_config,
    )
