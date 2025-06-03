import os
from pprint import pprint
from time import perf_counter
from typing import Any, Dict, List, Tuple

import numpy as np
import sympy as sp
from matplotlib import pyplot as plt
from prettytable import PrettyTable

from evosym_run import evosym
from individual import Individual
from numerify import numerify_tree
from symbolic_expression import expr_from_postfix
from utils import Add, Transpose


def test_run(
    display_info: bool = False, display_graphs: bool = False
) -> Tuple[
    Individual,
    float,
    float,
    List[float],
    List[float],
    List[float],
    Dict[str, Any],
    Dict[str, Any],
    float,
]:
    xlim = [-2.0, 2.0]
    ylim = [-2.0, 2]
    x = np.linspace(xlim[0], xlim[1], 100)
    y = 0.5 * np.exp(x) * np.cos(2 * x) + np.sin(x)
    y_prime = 4 * np.cos(2 * x) - 0.5
    data = (x, y)
    data_prime = (x, y_prime)

    vars = [sp.Symbol("x")]
    operators = {"+": 2, "-": 2, "*": 2, "sin": 1, "cos": 1, "exp": 1}
    constants_range = [-2.0, 2]

    # PARAMETER DEFINITION
    population_size = 50
    generations = 50
    crossover_method = "normal"
    crossover_yields_two_children = False
    meancrossover_rate = 0.0
    mutation_rate = 0.1
    divine_mutation_rate = 0.0

    mutation_method = "nodal"
    delete_mutation_rate = 0.0
    selection_method = "tournament"
    tournaments_size = 2
    size_penalty = "sqrt"
    fitness_method = "mse"
    derivative_data = False

    start = perf_counter()
    (
        best_ind,
        best_fit,
        worst_fit,
        gen_best_fits,
        gen_worst_fits,
        gen_mean_fits,
        extra_info,
    ) = evosym(
        data,
        population_size,
        generations,
        vars,
        operators,
        data_prime=data_prime if derivative_data else None,
        crossover_yields_twot_children=crossover_yields_two_children,
        crossover_method=crossover_method,
        meancrossover_rate=meancrossover_rate,
        mutation_rate=mutation_rate,
        delete_mutation_rate=delete_mutation_rate,
        divine_mutation_rate=divine_mutation_rate,
        mutation_method=mutation_method,
        fitness_method=fitness_method,
        size_penalty=size_penalty,
        selection_method=selection_method,
        constants_range=constants_range,
    )
    time_took = round(perf_counter() - start, 2)

    best_pst = best_ind.tree.post_order()

    parameters = {
        "pop_size": population_size,
        "gen_number": generations,
        "crossover_yields_twot_children": crossover_yields_two_children,
        "crossover_method": crossover_method,
        "meancrossover_rate": meancrossover_rate,
        "mutation_rate": mutation_rate,
        "delete_mutation_rate": delete_mutation_rate,
        "mutation_method": mutation_method,
        "divine_mutation_rate": divine_mutation_rate,
        "sel_method": selection_method,
        "tournament_size": tournaments_size,
        "fitness_method": fitness_method,
        "size_penalty": size_penalty,
        "derivative_data": derivative_data,
    }

    if display_info:
        print("\n---------------------------------------", end="\n\n")

        print("Run parameters:")
        pprint(parameters, sort_dicts=False)
        print("\nRun symbolic variables:")
        pprint(vars)
        print("\nRun operators:")
        pprint(operators, sort_dicts=False)

        print("\nTarget expression:")
        pprint("0.5exp(x) * cos(2x) + sin(x) + N(0, 0.2)")

        best_expr = expr_from_postfix(best_pst, operators)
        print("\nBest expression:")
        print(best_expr)
        print("\nDepth of best tree:", best_ind.tree.depth())
        print("Number of nodes of best tree:", best_ind.tree.size())
        print("Best fitness:", best_fit, end="\n")

        print("Took " + str(time_took) + "s")

        print("\n---------------------------------------")

    if display_graphs:
        figure, axes = plt.subplots(2, 2, figsize=(16, 8))
        gens_best_size = extra_info["gens_best_size"]
        gens_mean_size = extra_info["gens_mean_size"]
        best_eval = numerify_tree(best_ind.tree, vars, operators)(x)
        evaluation = (
            best_eval.reshape(-1)
            if not isinstance(best_eval, float | int)
            else [best_eval] * len(x)
        )
        axes[0][0].title.set_text("Target data vs best fitting curve")
        axes[0][0].grid(True)
        axes[0][0].set_xlim(xlim)
        axes[0][0].set_ylim(ylim)
        axes[0][0].plot(x, y, label="Target data")
        axes[0][0].plot(x, evaluation, label="Best fitting curve")
        axes[0][0].legend()

        axes[0][1].title.set_text("Best and mean geenration sizes")
        axes[0][1].plot(
            [gen for gen in range(generations)],
            gens_best_size,
            label="Best sizes",
            color="red",
        )
        axes[0][1].plot(
            [gen for gen in range(generations)],
            gens_mean_size,
            label="Mean sizes",
        )
        axes[0][1].legend()

        axes[1][0].title.set_text("Mean generation fitness")
        axes[1][0].plot(
            [gen for gen in range(generations)],
            gen_mean_fits,
            label="Mean fitness",
            color="purple",
        )

        axes[1][1].title.set_text("Best generation fitness")
        axes[1][1].plot(
            [gen for gen in range(generations)],
            gen_best_fits,
            label="Best fitness",
            color="green",
        )
        plt.tight_layout()
        plt.show()

    return (
        best_ind,
        best_fit,
        worst_fit,
        gen_best_fits,
        gen_worst_fits,
        gen_mean_fits,
        parameters,
        extra_info,
        time_took,
    )


def main() -> None:
    N_RUNS = 1
    OUTPUT_PATH = "./runs/"
    GENERATIONS = 50
    save = False

    best_inds, best_fits, worst_fits, mean_fits = [], [], [], []
    best_ind_sizes, mean_sizes = [], []
    gens_best_sizes = [0.0] * GENERATIONS
    gens_mean_sizes = [0.0] * GENERATIONS
    pars = {}
    times = []
    for i in range(N_RUNS):
        ind, bf, wf, _, _, mf, pars, extra, time = test_run(
            display_graphs=True, display_info=True
        )
        best_inds.append(ind)
        best_fits.append(bf)
        worst_fits.append(wf)
        mean_fits.append(mf[-1])
        gens_best_size = extra["gens_best_size"]
        gens_mean_size = extra["gens_mean_size"]
        gens_best_sizes = Add(gens_best_sizes, gens_best_size)
        gens_mean_sizes = Add(gens_mean_sizes, gens_mean_size)
        best_ind_sizes.append(gens_best_size[-1])
        mean_sizes.append(gens_mean_size[-1])
        times.append(time)
        print(f"(Done {i + 1} / {N_RUNS})")

    matrix = [best_fits, worst_fits, mean_fits, best_ind_sizes, mean_sizes, times]
    matrix = Transpose(matrix)

    table = PrettyTable(
        [
            "Best fitness",
            "Worst fitness",
            "Mean fitness",
            "Best ind size",
            "Ind mean size",
            "Execution Time",
        ]
    )
    for i in range(N_RUNS):
        table.add_row(matrix[i])

    table2 = PrettyTable(
        [
            "Mean Max Fitness",
            "Mean Min Fitness",
            "Mean Mean Fitness",
            "Mean Best Ind Size",
            "Mean Ind Mean Size",
            "Mean Execution Time",
        ]
    )
    table2.add_row(
        list(
            map(
                float,
                [
                    np.mean(best_fits),
                    np.mean(worst_fits),
                    np.mean(mean_fits),
                    np.mean(best_ind_sizes),
                    np.mean(mean_sizes),
                    np.mean(times),
                ],
            )
        )
    )

    table3 = PrettyTable()
    table3.add_column(
        "Best Ind Size per Gen", list(map(lambda x: x / N_RUNS, gens_best_sizes))
    )
    table3.add_column(
        "Mean Ind Size per Gen", list(map(lambda x: x / N_RUNS, gens_mean_sizes))
    )

    if save:
        csv_string = table.get_csv_string()
        csv_string2 = table2.get_csv_string()
        csv_string3 = table3.get_csv_string()
        runsets_files = os.listdir(OUTPUT_PATH)
        current_index = len(runsets_files)
        path = OUTPUT_PATH + f"runset_{current_index}/"
        os.makedirs(path)
        with open(path + "run_info.txt", "w") as f:
            pars_str = str(pars)
            items = list(map(lambda line: line + "\n", pars_str[1:-1].split(", ")))
            f.writelines(items)
            f.write("---\n")
            f.write(csv_string)
            f.write("---\n")
            f.write(csv_string2)
            f.write("---\n")
            f.write(csv_string3)

    return None


if __name__ == "__main__":
    main()
