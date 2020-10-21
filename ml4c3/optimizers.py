# Imports: third party
from tensorflow.keras import optimizers
from tensorflow_addons.optimizers import (
    RectifiedAdam,
    TriangularCyclicalLearningRate,
    Triangular2CyclicalLearningRate,
)


def get_optimizer(
    name: str,
    learning_rate: float,
    steps_per_epoch: int = None,
    learning_rate_schedule: str = None,
    optimizer_kwargs=None,
):
    if not optimizer_kwargs:
        optimizer_kwargs = {}
    name = str.lower(name)
    rate_or_schedule = _get_learning_rate_schedule(
        learning_rate,
        learning_rate_schedule,
        steps_per_epoch,
    )
    try:
        opt = optimizers.get(name)
        opt.__init__(rate_or_schedule, **optimizer_kwargs)
        return opt
    except ValueError:
        pass
    if name in NON_KERAS_OPTIMIZERS:
        return NON_KERAS_OPTIMIZERS[name](rate_or_schedule, **optimizer_kwargs)
    raise ValueError(f"Unknown optimizer {name}.")


def _get_learning_rate_schedule(
    learning_rate: float,
    learning_rate_schedule: str = None,
    steps_per_epoch: int = None,
):
    if learning_rate_schedule is None:
        return learning_rate
    if learning_rate_schedule == "triangular":
        return TriangularCyclicalLearningRate(
            initial_learning_rate=learning_rate / 5,
            maximal_learning_rate=learning_rate,
            step_size=steps_per_epoch * 5,
        )
    if learning_rate_schedule == "triangular2":
        return Triangular2CyclicalLearningRate(
            initial_learning_rate=learning_rate / 5,
            maximal_learning_rate=learning_rate,
            step_size=steps_per_epoch * 5,
        )
    else:
        raise ValueError(f'Learning rate schedule "{learning_rate_schedule}" unknown.')


NON_KERAS_OPTIMIZERS = {
    "radam": RectifiedAdam,
}
