"""Population class."""

from database.base import Base
from database.individual import Individual
from revolve2.experimentation.optimization.ea import Population as GenericPopulation


class Population(Base, GenericPopulation[Individual], kw_only=True):
    """A population of individuals."""

    __tablename__ = "population"
