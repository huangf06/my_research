"""Experiment class."""

import sqlalchemy.orm as orm
from database.base import Base, HasId

class Experiment(Base, HasId):
    """Experiment description."""

    __tablename__ = "experiment"

    # The seed for the rng.
    rng_seed: orm.Mapped[int] = orm.mapped_column(nullable=False)
