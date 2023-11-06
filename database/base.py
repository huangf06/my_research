"""Base class."""

import sqlalchemy.orm as orm


class Base(orm.MappedAsDataclass, orm.DeclarativeBase):
    """Base class for all SQLAlchemy models in this example."""

    pass

class HasId(orm.MappedAsDataclass):
    """An SQLAlchemy mixin that provides an id column."""

    id: orm.Mapped[int] = orm.mapped_column(
        primary_key=True,
        unique=True,
        autoincrement=True,
        nullable=False,
        init=False,
    )