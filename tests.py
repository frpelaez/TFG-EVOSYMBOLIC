import os
from pprint import pprint
from time import perf_counter
from typing import Any, Dict, List, Tuple

import numpy as np
import sympy as sp
from matplotlib import pyplot as plt
from prettytable import PrettyTable

from datastructures import BT
from evosym_run import evosym
from individual import Individual
from numerify import fitness, numerify_tree
from random_search import random_search
from symbolic_expression import expr_from_postfix
from utils import Add, Transpose, calculate_deriv

# RUN INFO
xlim = [-2.0, 2.0]
ylim = [-4.0, 4.0]
x = np.linspace(xlim[0], xlim[1], 100)
y = np.exp(-0.7 * x**2) + np.random.normal(0, 0.2, 100)
data = (x, y)
y_prime = 0
y_prime_aprox = calculate_deriv(data)
use_aprox = True
data_prime = (x, y_prime_aprox) if use_aprox else (x, y_prime)

vars = [sp.Symbol("x")]
operators = {"+": 2, "-": 2, "*": 2, "sin": 1, "cos": 1, "exp": 1}
constants_range = [-2.0, 2.0]

# PARAMETER DEFINITION
population_size = 50
generations = 50
iterations = population_size * generations
comp_depth = 2
crossover_method = "normal"
crossover_yields_two_children = True
meancrossover_rate = 0.0
mutation_rate = 0.05
divine_mutation_rate = 0.0

mutation_method = "nodal"
delete_mutation_rate = 0.0
selection_method = "tournament"
tournaments_size = 2
size_penalty = "none"
fitness_method = "mse"
derivative_data = True
alpha = 0.85
terminal_rate = -1

# EXTRA INFO
N_RUNS = 5
OUTPUT_PATH = "./runs/"
GENERATIONS = generations
save = True
show_info = True
show_graphs = True
use_random_search = False


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
        alpha=alpha,
        crossover_yields_two_children=crossover_yields_two_children,
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
        "comp_depth": comp_depth,
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
        "uses_deriv_aprox": use_aprox,
        "alpha": alpha,
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
        pprint("1 - cos(x) * cos(x)")

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
        axes[0][0].plot(x, evaluation, label="Best fitting curve", linestyle="dashed")
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
    best_inds, best_fits, worst_fits, mean_fits = [], [], [], []
    best_ind_sizes, mean_sizes = [], []
    gens_best_fits = [0.0] * GENERATIONS
    gens_mean_fits = [0.0] * GENERATIONS
    gens_best_sizes = [0.0] * GENERATIONS
    gens_mean_sizes = [0.0] * GENERATIONS
    pars = {}
    times = []
    hits = 0
    best_obj_fits = []
    best_gens = []

    if not use_random_search:
        for i in range(N_RUNS):
            ind, bf, wf, gen_best_fit, _, gen_mean_fit, pars, extra, time = test_run(
                display_graphs=show_graphs, display_info=show_info
            )
            best_inds.append(ind)
            obj_fit = fitness(
                ind.tree,
                data,
                vars,
                operators,
                method="mse",
                size_penalty="none",
            )
            best_obj_fits.append(obj_fit)
            best_fits.append(bf)
            if obj_fit < 0.05:
                hits += 1
            worst_fits.append(wf)
            mean_fits.append(gen_mean_fit[-1])
            gens_best_size = extra["gens_best_size"]
            gens_mean_size = extra["gens_mean_size"]
            best_gen = extra["best_gen"]
            best_gens.append(best_gen)
            gens_best_fits = Add(gens_best_fits, gen_best_fit)
            gens_mean_fits = Add(gens_mean_fits, gen_mean_fit)
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
                "Number Of Hits",
                "Mean Obj Fitness",
                "Best Ind Gen",
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
            + [hits]
            + [float(np.mean(best_obj_fits))]
            + [float(np.mean(best_gens))]
        )

        table3 = PrettyTable()
        table3.add_column(
            "Best Ind Size per Gen", list(map(lambda x: x / N_RUNS, gens_best_sizes))
        )
        table3.add_column(
            "Mean Ind Size per Gen", list(map(lambda x: x / N_RUNS, gens_mean_sizes))
        )

        table4 = PrettyTable()
        table4.add_column(
            "Best Fit per Gen", list(map(lambda x: x / N_RUNS, gens_best_fits))
        )
        table4.add_column(
            "Mean Fit per Gen", list(map(lambda x: x / N_RUNS, gens_mean_fits))
        )

        if save:
            csv_string = table.get_csv_string()
            csv_string2 = table2.get_csv_string()
            csv_string3 = table3.get_csv_string()
            csv_string4 = table4.get_csv_string()
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
                f.write("---\n")
                f.write("Best expression")
                f.write(
                    str(
                        expr_from_postfix(
                            min(
                                best_inds, key=lambda ind: ind.fitness
                            ).tree.post_order(),
                            operators,
                        )
                    )
                )
            with open(path + "fitness_info.txt", "w") as f:
                head, body = csv_string4.split("\n", 1)
                head = "gen " + head.replace(" ", "-").replace(",", " ")
                body = body.replace(",", " ")
                gen = 0
                text = ""
                for line in body.split("\n"):
                    if line != "":
                        new_line = str(gen) + " " + line
                        text = text + new_line
                        gen += 1
                f.write(head + "\n" + text)
    else:
        hits = 0
        best_iteration = 0
        fitnesses_values = [0.0] * iterations
        best_ind = Individual(BT(), float("inf"))
        top_fit = float("inf")
        for i in range(N_RUNS):
            print(f"\nrunning random search {i + 1}/{N_RUNS}")
            ind, best_fit, fitnesses, best_iter = random_search(
                data,
                vars,
                operators,
                iterations,
                terminal_rate,
                fitness_method=fitness_method,
                derivative_data=data_prime,
                alpha=alpha,
                size_penalty=size_penalty,
            )
            if best_fit < top_fit:
                top_fit = best_fit
                best_ind = ind
            obj_fit = fitness(
                ind.tree,
                data,
                vars,
                operators,
                method="mse",
                size_penalty="none",
            )
            if obj_fit < 0.05:
                hits += 1
            best_iteration += best_iter
            fitnesses_values = Add(fitnesses_values, fitnesses)

        if save:
            print("\nsaving results")
            runsets_files = os.listdir(OUTPUT_PATH)
            current_index = len(runsets_files)
            path = OUTPUT_PATH + f"runset_{current_index}-random-search/"
            os.makedirs(path)
            with open(path + "fitness_info.txt", "w") as f:
                f.write("iter fitness\n")
                for i, fit in enumerate(fitnesses_values):
                    line = f"{i} {fit}\n"
                    f.write(line)
            with open(path + "best_expr.txt", "w") as f:
                best_expr = expr_from_postfix(best_ind.tree.post_order(), operators)
                f.write(str(best_expr))
            with open(path + "hit_info.txt", "w") as f:
                f.write("hits best-iter best-fitness\n")
                f.write(f"{hits} {best_iteration / N_RUNS} {top_fit}")

        figure, axes = plt.subplots(1, 2)
        best_eval = numerify_tree(best_ind.tree, vars, operators)(x)
        evaluation = (
            best_eval.reshape(-1)
            if not isinstance(best_eval, float | int)
            else [best_eval] * len(x)
        )
        axes[0].title.set_text("Target data vs best fitting curve")
        axes[0].grid(True)
        axes[0].set_xlim(xlim)
        axes[0].set_ylim(ylim)
        axes[0].plot(x, y, label="Target data")
        axes[0].plot(x, evaluation, label="Best fitting curve", linestyle="dashed")
        axes[0].legend()

        axes[1].title.set_text("Fitness over iterations")
        axes[1].grid(True)
        axes[1].plot(
            [i for i in range(iterations)], fitnesses_values, label="fit evolution"
        )
        axes[1].legend()

        plt.tight_layout()
        plt.show()

    return None


if __name__ == "__main__":
    main()
