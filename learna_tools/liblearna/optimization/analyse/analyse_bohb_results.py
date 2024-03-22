import pandas as pd
import matplotlib.pyplot as plt
import hpbandster.core.result as hpres
import hpbandster.visualization as hpvis

from pathlib import Path

def analyse_bohb_run(run):
    # load the example run from the log files
    result = hpres.logged_results_to_HBS_result(f"results/bohb/{run}")

    # get all executed runs
    all_runs = result.get_all_runs()

    # get the 'dict' that translates config ids to the actual configurations
    id2conf = result.get_id2config_mapping()

    # Here is how you get he incumbent (best configuration)
    inc_id = result.get_incumbent_id()

    # let's grab the run on the highest budget
    inc_runs = result.get_runs_by_id(inc_id)
    inc_run = inc_runs[-1]

    # We have access to all information: the config, the loss observed during
    #optimization, and all the additional information
    inc_config = id2conf[inc_id]['config']
    print('### Incumbent config:', inc_id, '\n', inc_config)
    inc_test_loss = inc_run.info['validation_info']

    print(f"Validation info of Incumbent: {inc_test_loss}")

    inc_loss = inc_run.loss
    print('### Incumbent loss:', inc_loss)

    # Let's plot the observed losses grouped by budget,
    hpvis.losses_over_time(all_runs)

    # the number of concurent runs,
    hpvis.concurrent_runs_over_time(all_runs)

    # and the number of finished runs.
    hpvis.finished_runs_over_time(all_runs)

    # This one visualizes the spearman rank correlation coefficients of the losses
    # between different budgets.
    hpvis.correlation_across_budgets(result)

    # For model based optimizers, one might wonder how much the model actually helped.
    # The next plot compares the performance of configs picked by the model vs. random ones
    hpvis.performance_histogram_model_vs_random(all_runs, id2conf)

    plt.show()

    # get the best configuration based on the number of solved tasks during validation
    min_solved = 0

    all_solved = [(x.info['validation_info']['num_solved'], x.info['validation_info']["sum_of_min_distances"], x.config_id, x.budget, id2conf[x.config_id]['config']) for x in all_runs if x.info and int(x.info['validation_info']['num_solved']) >= min_solved]

    all_solved_dict = [
        {'num_solved': x.info['validation_info']['num_solved'], 'sum_of_min_distances': x.info['validation_info']["sum_of_min_distances"], 'config_id': x.config_id, 'budget': x.budget, 'start_time': x['time_stamps']['started'], 'finishing_time': x['time_stamps']['finished'], 'config': id2conf[x.config_id]['config']} for x in all_runs if x.info and int(x.info['validation_info']['num_solved']) >= min_solved
    ]
        
    # we can also get the training information
    training_info = [x.info["train_info"] for x in all_runs if x.info]
    # print(training_info[:2])
    
    # or find configurations that failed
    all_failed = [(x.config_id, id2conf[x.config_id]['config']) for x in all_runs if not x.info]
    all_failed_config_ids = [x[0][0] for x in all_failed]
    print('Number of failed configurations:', len(all_failed))
    print('Ids of failed configurations:', all_failed_config_ids)

    all_solved_sorted = sorted(all_solved, key=lambda x: x[0], reverse=True)

    print(f"number of configurations evaluated: {len(all_solved)}")

    print(f"{len(all_solved_sorted)} configurations solved at least {min_solved} targets:")
    print(f"Most solving config: {all_solved_sorted[0][2]}; Solved tasks: {all_solved_sorted[0][0]}; Sum of min distances: {all_solved_sorted[0][1]}")

    print("Ten best configurations based on soved validation Tasks:")

    for index, i in enumerate(all_solved_sorted[:10]):
        print(f"[{index + 1}]")
        print(f"Config id: {i[2]}")
        print(f"Number of solved targets: {i[0]}")
        print(f"Sum of min distances: {i[1]}")
        # print(f"Budget: {i[3]}")
        print(f"Config: {i[4]}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--path", type=str, default="results/bohb", help="Path to results of run"
    )

    parser.add_argument(
        "--run", type=str, help="The id of the run"
    )

    args = parser.parse_args()

    analyse_bohb_run(args.run)