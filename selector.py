import numpy as np
import numpy.typing as npt
from revolve2.experimentation.optimization.ea import population_management, selection
from database.population import Population
from database.individual import Individual


class Selector:
    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def select_parents(self, population: Population, offspring_size: int) -> npt.NDArray[np.float_]:
        return np.array(
            [
                selection.multiple_unique(
                    2,
                    [individual.genotype for individual in population.individuals],
                    [individual.fitness for individual in population.individuals],
                    lambda _, fitness: selection.tournament(self.rng, fitness, k=1),
                )
                for _ in range(offspring_size)
            ],
        )

    def select_survivors(
            rng: np.random.Generator,
            original_population: list[Individual],
            offspring_population: list[Individual],
    ) -> list[Individual]:
        """
        Select survivors using a tournament.

        :param rng: Random number generator.
        :param original_population: The population the parents come from.
        :param offspring_population: The offspring.
        :returns: A newly created population.
        """
        original_survivors, offspring_survivors = population_management.steady_state(
            [i.genotype for i in original_population],
            [i.fitness for i in original_population],
            [i.genotype for i in offspring_population],
            [i.fitness for i in offspring_population],
            lambda n, genotypes, fitnesses: selection.multiple_unique(
                n,
                genotypes,
                fitnesses,
                lambda _, fitnesses: selection.tournament(rng, fitnesses, k=2),
            ),
        )

        return [
            Individual(
                original_population[i].genotype,
                original_population[i].fitness,
            )
            for i in original_survivors
        ] + [
            Individual(
                offspring_population[i].genotype,
                offspring_population[i].fitness,
            )
            for i in offspring_survivors
        ]

    def find_best_robot(self, current_best: Individual | None, population: list[Individual]) -> Individual:
        return max(
            population + ([] if current_best is None else [current_best]),
            key=lambda x: x.fitness,
        )
