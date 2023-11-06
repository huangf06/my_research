import config
import logging
import multineat
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session
from database.base import Base
from database.experiment import Experiment
from database.generation import Generation
from database.individual import Individual
from database.population import Population
from database.genotype import Genotype
from revolve2.experimentation.logging import setup_logging
from revolve2.experimentation.rng import make_rng, seed_from_time
from evaluator import Evaluator
from revolve2.experimentation.database import OpenMethod, open_database_sqlite
from selector import Selector
import numpy as np
import numpy.typing as npt
from revolve2.experimentation.optimization.ea import population_management, selection


class ExperimentRunner:
    def __init__(self, engine: Engine):
        self.engine = engine

    @classmethod
    def setup_database(cls):
        engine = open_database_sqlite(
            config.DATABASE_FILE, open_method=OpenMethod.NOT_EXISTS_AND_CREATE
        )
        Base.metadata.create_all(engine)
        return engine

    def save_to_database(self, dbengine: Engine, obj: Base) -> None:
        """
        Save an object to the database and commit the session.

        :param session: The database session to use for saving the object.
        :param obj: The object to save to the database.
        """
        with Session(dbengine, expire_on_commit=False) as session:
            session.add(obj)
            session.commit()

    def select_parents(
            rng: np.random.Generator,
            population: list[Individual],
            offspring_size: int,
    ) -> npt.NDArray[np.float_]:
        """
        Select pairs of parents using a tournament.

        :param rng: Random number generator.
        :param population: The population to select from.
        :param offspring_size: The number of parent pairs to select.
        :returns: Pairs of indices of selected parents. offspring_size x 2 ints.
        """
        return np.array(
            [
                selection.multiple_unique(
                    2,
                    [individual.genotype for individual in population],
                    [individual.fitness for individual in population],
                    lambda _, fitnesses: selection.tournament(rng, fitnesses, k=1),
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

    def find_best_robot(
            current_best: Individual | None, population: list[Individual]
    ) -> Individual:
        """
        Return the best robot between the population and the current best individual.

        :param current_best: The current best individual.
        :param population: The population.
        :returns: The best individual.
        """
        return max(
            population + [] if current_best is None else [current_best],
            key=lambda x: x.fitness,
        )

    def run_experiment(self, offspring_fitnesses=None) -> None:
        logging.info("----------------")
        logging.info("Start experiment")

        # Set up the random number generator.
        rng_seed = seed_from_time()
        rng = make_rng(rng_seed)

        # Create and save the experiment instance.
        experiment = Experiment(rng_seed=rng_seed)
        logging.info("Saving experiment configuration.")
        self.save_to_database(self.engine, experiment)

        # Intialize the evaluator that will be used to evaluate robots.
        evaluator = Evaluator(headless=True, num_simulators=config.NUM_SIMULATORS)

        # CPPN innovation databases.
        innov_db_body = multineat.InnovationDatabase()
        innov_db_brain = multineat.InnovationDatabase()

        # Create an initial population.
        logging.info("Generating initial population.")
        initial_genotypes = [
            Genotype.random(
                innov_db_body=innov_db_body,
                innov_db_brain=innov_db_brain,
                rng=rng,
            )
            for _ in range(config.POPULATION_SIZE)
        ]

        # Evaluate the initial population.
        logging.info("Evaluating initial population.")
        initial_fitness = evaluator.evaluate(
            [genotype.develop() for genotype in initial_genotypes]
        )

        # Create a population of individuals, combining genotype with fitness.
        population = Population(
            individuals=[
                Individual(genotype=genotype, fitness=fitness)
                for genotype, fitness in zip(initial_genotypes, initial_fitness)
            ]
        )

        # Finish the zeroth generation and save it to the database.
        generation = Generation(
            experiment=experiment, generation_index=0, population=population
        )
        logging.info("Saving generation.")
        self.save_to_database(self.engine, generation)

        # Start the actual optimization process.
        logging.info("Start optimization process.")
        while generation.generation_index < config.NUM_GENERATIONS:
            logging.info(
                f"Generation {generation.generation_index + 1} / {config.NUM_GENERATIONS}."
            )

            selector = Selector(rng)

            # Create offspring.
            parents = selector.select_parents(population, config.OFFSPRING_SIZE)
            offspring_genotypes = [
                Genotype.crossover(
                    population.individuals[parent1_i].genotype,
                    population.individuals[parent2_i].genotype,
                    rng,
                ).mutate(innov_db_body, innov_db_brain, rng)
                for parent1_i, parent2_i in parents
            ]

            # Evaluate the offspring.
            offspring_fitness = evaluator.evaluate(
                [genotype.develop() for genotype in offspring_genotypes]
            )

            # Make an intermediate offspring population.
            offspring_population = [
                Individual(genotype, fitness)
                for genotype, fitness in zip(offspring_genotypes, offspring_fitness)
            ]

            population = selector.select_survivors(rng, population, offspring_population,)

            # Make it all into a generation and save it to the database.
            generation = Generation(
                experiment=experiment,
                generation_index=generation.generation_index + 1,
                population=population,
            )
            logging.info("Saving generation.")
            self.save_to_database(self.engine, generation)

    def run(self) -> None:
        """
        Run the experiment.
        """
        setup_logging(file_name="log.txt")
        for _ in range(config.NUM_REPETITIONS):
            self.run_experiment()