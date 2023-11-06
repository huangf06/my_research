from revolve2.ci_group import fitness_functions, terrains
from revolve2.ci_group.simulation import make_standard_batch_parameters
from revolve2.modular_robot import ModularRobot
from revolve2.modular_robot_simulation import (
    ModularRobotScene,
    Terrain,
    simulate_scenes,
)
from revolve2.mujoco_simulator import LocalSimulator


class Evaluator:
    """Provides evaluation of robots."""

    def __init__(
            self,
            headless: bool,
            num_simulators: int,
            terrain: Terrain = terrains.flat(),
            fitness_function=fitness_functions.xy_displacement,
    ) -> None:
        """
        Initialize this object.

        :param headless: `headless` parameter for the physics simulator.
        :param num_simulators: `num_simulators` parameter for the physics simulator.
        """
        self._simulator = LocalSimulator(
            headless=headless, num_simulators=num_simulators
        )
        self._terrain = terrain
        self._fitness_function = fitness_function

    def set_terrain(self, terrain: Terrain) -> None:
        """
        Set the terrain for the simulations.

        :param terrain: The terrain to use for the simulations.
        """
        self._terrain = terrain

    def set_fitness_function(self, fitness_function) -> None:
        """
        Set the fitness function for evaluating the robots.

        :param fitness_function: The fitness function to use for evaluating the robots.
        """
        self._fitness_function = fitness_function

    def evaluate(
            self,
            robots: list[ModularRobot],
    ) -> list[float]:
        """
        Evaluate multiple robots.

        :param robots: The robots to simulate.
        :returns: Fitness of the robots.
        """
        # Create the scenes.
        scenes = []
        for robot in robots:
            scene = ModularRobotScene(terrain=self._terrain)
            scene.add_robot(robot)
            scenes.append(scene)

        # Simulate all scenes.
        scene_states = simulate_scenes(
            simulator=self._simulator,
            batch_parameters=make_standard_batch_parameters(),
            scenes=scenes,
        )

        # Calculate the fitness using the provided fitness function.
        fitness = [
            self._fitness_function(
                states[0].get_modular_robot_simulation_state(robot),
                states[-1].get_modular_robot_simulation_state(robot),
            )
            for robot, states in zip(robots, scene_states)
        ]

        return fitness
