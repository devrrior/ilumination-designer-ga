import numpy as np

from persistance.db import Session
from persistance.models.bulb import Bulb
from sqlalchemy import func
from utils.random_utils import random_int
import matplotlib.pyplot as plt

session = Session()


class LightsDesignerAlgorithm:
    def __init__(
        self,
        width: float,
        height: float,
        grid_width: int,
        grid_height: int,
        light_type: str,
        min_flux_needed: int,
        init_population: int,
        max_population: int,
        prob_mutation: float,
        prob_mutation_per_grid: float,
        prob_crossover: float,
        generations: int,
    ):
        self.width = width
        self.height = height
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.light_type = light_type
        self.min_flux_needed = min_flux_needed
        self.init_population = init_population
        self.max_population = max_population
        self.prob_mutation = prob_mutation
        self.prob_mutation_per_grid = prob_mutation_per_grid
        self.prob_crossover = prob_crossover
        self.generations = generations

    def run(self):
        # Run the algorithm
        init_population: np.ndarray = self.generate_initial_population()
        population = init_population
        population_history = []
        statistics_history = []

        for _ in range(self.generations):
            # Select parents
            parents = self.select_parents(population)

            # Crossover
            pairs = self.make_pairs_to_crossover(parents)

            unmutated_parents = []
            for pair in pairs:
                new_system1, new_system2 = self.crossover(pair[0], pair[1])
                unmutated_parents.append(new_system1)
                unmutated_parents.append(new_system2)

            unmutated_parents = np.array(unmutated_parents)

            # Mutation
            mutated_population = self.mutate(unmutated_parents)

            # Merge old population with mutated population
            population = np.concatenate((population, mutated_population))

            # Get best, worst and mean fitness
            best, worst, mean = self.get_mean_best_worst_fitness(population)
            statistics_history.append(
                {"best": best["fitness"], "worst": worst["fitness"], "mean": mean}
            )
            print(f"Best: {best['fitness']}, Worst: {worst['fitness']}, Mean: {mean}")
            population_history.append(population)

            # Prune population
            population = self.prune(population)

        return population_history, statistics_history

    def generate_initial_population(self) -> np.ndarray:
        population = []
        width_per_grid = self.width / self.grid_width
        height_per_grid = self.height / self.grid_height

        for _ in range(self.init_population):
            grids = np.array([])
            for i in range(self.grid_width):
                for j in range(self.grid_height):
                    random_bulb = session.query(Bulb).order_by(func.rand()).first()
                    grids = np.append(
                        grids,
                        self.calculate_data_per_grid(
                            width_per_grid,
                            height_per_grid,
                            random_bulb,
                            i,
                            j,
                        ),
                    )

            mean = self.calculate_mean(grids)
            system = {
                "grids": grids,
                "mean": mean,
                "standard_deviation": self.calculate_standard_deviation(grids, mean),
                "fitness": abs(self.min_flux_needed - mean),
            }
            population.append(system)

        return np.array(population)

    def calculate_data_per_grid(
        self,
        width_per_grid: float,
        height_per_grid: float,
        random_bulb: Bulb,
        num_of_grid_x: int,
        num_of_grid_y: int,
    ) -> dict:
        return {
            "x": num_of_grid_x * width_per_grid + width_per_grid / 2,
            "y": num_of_grid_y * height_per_grid + height_per_grid / 2,
            "bulb": random_bulb,
            "luxes": random_bulb.lumens / (width_per_grid * height_per_grid),
        }

    def calculate_mean(self, grids: np.ndarray):
        mean = 0
        for grid in grids:
            mean += grid["luxes"]

        return mean / len(grids)

    def calculate_standard_deviation(self, grids: np.ndarray, mean: float):
        standard_deviation = 0
        for grid in grids:
            standard_deviation += (grid["luxes"] - mean) ** 2

        return np.sqrt(standard_deviation / len(grids))

    def select_parents(self, population: np.ndarray):
        parents = []
        for indv in population:
            if (random_int(0, 100) / 100) <= self.prob_crossover:
                parents.append(indv)

        return np.array(parents)

    def make_pairs_to_crossover(self, parents: np.ndarray):
        pairs = []
        for indv1 in parents:
            random_i = random_int(0, len(parents) - 1)
            indv2 = parents[random_i]
            pairs.append([indv1, indv2])

        return np.array(pairs)

    def crossover(self, indv1: dict, indv2: dict):
        grids1 = indv1["grids"]
        grids2 = indv2["grids"]

        crossover_point = random_int(0, len(grids1) - 1)
        new_grids1 = np.concatenate(
            (grids1[:crossover_point], grids2[crossover_point:])
        )
        new_grids2 = np.concatenate(
            (grids2[:crossover_point], grids1[crossover_point:])
        )

        new_system1_mean = self.calculate_mean(new_grids1)
        new_system1 = {
            "grids": new_grids1,
            "mean": new_system1_mean,
            "standard_deviation": self.calculate_standard_deviation(
                new_grids1, new_system1_mean
            ),
            "fitness": abs(self.min_flux_needed - new_system1_mean),
        }

        new_system2_mean = self.calculate_mean(new_grids2)
        new_system2 = {
            "grids": new_grids2,
            "mean": new_system2_mean,
            "standard_deviation": self.calculate_standard_deviation(
                new_grids2, new_system2_mean
            ),
            "fitness": abs(self.min_flux_needed - new_system2_mean),
        }

        return new_system1, new_system2

    def mutate(self, population: np.ndarray):
        width_per_grid = self.width / self.grid_width
        height_per_grid = self.height / self.grid_height
        for indv in population:
            if (random_int(0, 100) / 100) <= self.prob_mutation:
                grids = indv["grids"]
                for i in range(len(grids)):
                    if (random_int(0, 100) / 100) <= self.prob_mutation_per_grid:
                        random_bulb = session.query(Bulb).order_by(func.rand()).first()
                        old_x = grids[i]["x"]
                        old_y = grids[i]["y"]
                        grids[i] = self.calculate_data_per_grid(
                            width_per_grid,
                            height_per_grid,
                            random_bulb,
                            0,
                            0,
                        )

                        grids[i]["x"] = old_x
                        grids[i]["y"] = old_y

                indv["grids"] = grids
                indv["mean"] = self.calculate_mean(grids)
                indv["standard_deviation"] = self.calculate_standard_deviation(
                    grids, indv["mean"]
                )
                indv["fitness"] = abs(self.min_flux_needed - indv["mean"])

        return population

    def get_mean_best_worst_fitness(self, population: np.ndarray):
        best = population[0]
        worst = population[0]
        mean = 0
        for indv in population:
            mean += indv["fitness"]
            if indv["fitness"] < best["fitness"]:
                best = indv
            if indv["fitness"] > worst["fitness"]:
                worst = indv

        return best, worst, mean / len(population)

    def prune(self, population: np.ndarray):
        # keep the best
        population = sorted(population, key=lambda x: x["fitness"])
        population = population[: self.max_population]
        return population

    def get_fitness(self, mean: float, standard_deviation: float):
        # sacar el fitness en base al siguiente puntaje
        # abs(self.min_flux_needed - new_system2_mean) = 0 -> puntaje de 10
        # standard_deviation = 0 -> puntaje de 10

        return 10 - (abs(self.min_flux_needed - mean) + standard_deviation)


if __name__ == "__main__":
    width = 5
    height = 5
    grid_width = 8
    grid_height = 5
    light_type = "LED"
    min_flux_needed = 300
    init_population = 10
    max_population = 10
    prob_crossover = 0.4
    prob_mutation = 0.6
    prob_mutation_per_grid = 0.5
    generations = 50
    algo = LightsDesignerAlgorithm(
        width,
        height,
        grid_width,
        grid_height,
        light_type,
        min_flux_needed,
        init_population,
        max_population,
        prob_mutation,
        prob_mutation_per_grid,
        prob_crossover,
        generations,
    )
    population_history, statistics_history = algo.run()

    # graficar historial de estadisticas
    stats_figure, stats_ax = plt.subplots()
    stats_figure.canvas.manager.set_window_title("Historial de estadisticas")
    stats_ax.set_title("Historial de estadisticas")
    stats_ax.set_xlabel("Generaciones")
    stats_ax.set_ylabel("Fitness")
    stats_ax.plot([x["best"] for x in statistics_history], label="Mejor")
    stats_ax.plot([x["worst"] for x in statistics_history], label="Peor")
    stats_ax.plot([x["mean"] for x in statistics_history], label="Promedio")
    stats_ax.legend()

    # graficar el mejor individuo de la ultima generacion
    # este representa el mejor diseño de iluminacion, por lo que se debe primero graficar el grid
    best_system = population_history[-1][0]
    best_system_grids = best_system["grids"]
    best_system_mean = best_system["mean"]
    best_system_standard_deviation = best_system["standard_deviation"]
    best_system_fitness = best_system["fitness"]

    best_system_figure, best_system_ax = plt.subplots()
    best_system_figure.canvas.manager.set_window_title("Mejor sistema de iluminacion")
    best_system_ax.set_title("Mejor sistema de iluminacion")
    best_system_ax.set_xlabel("Ancho")
    best_system_ax.set_ylabel("Alto")
    best_system_ax.set_xlim(0, width)
    best_system_ax.set_ylim(0, height)
    # draw grid
    for i in range(grid_width):
        best_system_ax.axvline(i * (width / grid_width), color="black")
    for i in range(grid_height):
        best_system_ax.axhline(i * (height / grid_height), color="black")

    # draw bulbs
    for grid in best_system_grids:
        best_system_ax.scatter(grid["x"], grid["y"], s=grid["bulb"].lumens, color="yellow")

    # agregar luxes por grid
    for grid in best_system_grids:
        best_system_ax.text(grid["x"], grid["y"], f"{grid['luxes']:.2f}", fontsize=8)





    plt.show()
