"""Hyperparameter optimisers compared under a matched budget.

Papers that report "we tuned with PSO" rarely establish that the metaheuristic
earned its keep. The usual comparison is against an untuned model, which only
shows that tuning helps at all. Here four optimisers search an identical space,
are stopped by the same wall-clock budget, and are all measured by the same
validation metric:

* random search, the control that is too often left out;
* Optuna's TPE, the Bayesian approach;
* particle swarm optimisation;
* a genetic algorithm.

Every optimiser works on the unit hypercube and decodes through the same
``SearchSpace``, so none of them can benefit from a different parameterisation.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

LOGGER = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Search space
# --------------------------------------------------------------------------- #
@dataclass
class Hyperparameter:
    name: str
    kind: str  # "float", "log", "int", "choice"
    low: float = 0.0
    high: float = 1.0
    choices: tuple = ()


class SearchSpace:
    """Maps a point in [0, 1]^d to a hyperparameter dictionary."""

    def __init__(self, parameters: list[Hyperparameter]) -> None:
        self.parameters = parameters

    @property
    def dim(self) -> int:
        return len(self.parameters)

    def decode(self, vector: np.ndarray) -> dict:
        values = {}
        for position, parameter in enumerate(self.parameters):
            u = float(np.clip(vector[position], 0.0, 1.0))
            if parameter.kind == "float":
                values[parameter.name] = parameter.low + u * (parameter.high - parameter.low)
            elif parameter.kind == "log":
                log_low, log_high = np.log(parameter.low), np.log(parameter.high)
                values[parameter.name] = float(np.exp(log_low + u * (log_high - log_low)))
            elif parameter.kind == "int":
                span = parameter.high - parameter.low + 1
                values[parameter.name] = int(
                    min(parameter.high, parameter.low + int(u * span))
                )
            elif parameter.kind == "choice":
                index = min(len(parameter.choices) - 1, int(u * len(parameter.choices)))
                values[parameter.name] = parameter.choices[index]
            else:
                raise ValueError(f"unknown kind {parameter.kind!r}")
        return values

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        return rng.random(self.dim)


def tabnet_search_space(include_heads: bool = True, tokens: bool = False) -> SearchSpace:
    """The space searched for both the vanilla and the gated architecture.

    ``n_heads`` is only meaningful for the multi-head variant; it is included for
    the gated model and omitted for vanilla so that neither wastes budget on a
    dimension it cannot use.
    """
    parameters = [
        Hyperparameter("n_d", "choice", choices=(8, 16, 24, 32, 48, 64)),
        Hyperparameter("n_steps", "int", low=3, high=7),
        Hyperparameter("gamma", "float", low=1.0, high=2.0),
        Hyperparameter("lambda_sparse", "log", low=1e-5, high=1e-2),
        Hyperparameter("learning_rate", "log", low=5e-3, high=5e-2),
        Hyperparameter("batch_size", "choice", choices=(1024, 2048, 4096)),
        Hyperparameter("n_independent", "int", low=1, high=3),
        Hyperparameter("n_shared", "int", low=1, high=3),
        Hyperparameter("focal_gamma", "float", low=0.5, high=3.0),
    ]
    if include_heads:
        parameters.append(Hyperparameter("n_heads", "int", low=2, high=8))
    if tokens:
        parameters.append(Hyperparameter("wang_heads", "choice", choices=(2, 4, 6, 8)))
    return SearchSpace(parameters)


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
@dataclass
class HPOResult:
    optimiser: str
    best_value: float
    best_params: dict
    n_evaluations: int
    elapsed_seconds: float
    history: list[dict] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "optimiser": self.optimiser,
            "best_value": self.best_value,
            "best_params": self.best_params,
            "n_evaluations": self.n_evaluations,
            "elapsed_seconds": self.elapsed_seconds,
            "history": self.history,
        }


class BudgetExhausted(Exception):
    """Raised internally once the wall-clock budget for a search is spent."""


class BudgetedObjective:
    """Wraps the objective with a wall-clock budget and an evaluation log.

    All four optimisers go through this, so the comparison is by time spent
    rather than by iteration count - one TPE proposal and one swarm update do
    not cost the same, and the optimisers' own overhead counts against the
    budget too.

    The budget is a hard deadline. ``fn(params, deadline)`` receives the absolute
    ``time.perf_counter()`` value at which the budget ends and must raise
    ``BudgetExhausted`` if it reaches it mid-evaluation; the model trainer does
    so after every epoch and every 20 batches. An interrupted evaluation is
    logged with ``status="interrupted"`` and contributes no value, so the
    incumbent at the end is the best evaluation that *completed* inside the
    budget and no search runs past it.

    ``fn`` may return a float or ``(float, info)``; ``info`` (epochs run, say) is
    stored with the evaluation.
    """

    def __init__(self, fn: Callable, space: SearchSpace, budget_seconds: float):
        self.fn = fn
        self.space = space
        self.budget_seconds = budget_seconds
        self.started = time.perf_counter()
        self.deadline = self.started + budget_seconds
        self.history: list[dict] = []
        self.best_value = -np.inf
        self.best_params: dict = {}

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.started

    @property
    def exhausted(self) -> bool:
        return self.elapsed >= self.budget_seconds

    @property
    def n_completed(self) -> int:
        return sum(1 for h in self.history if h["status"] == "ok")

    def __call__(self, vector: np.ndarray) -> float:
        if self.exhausted:
            raise BudgetExhausted
        params = self.space.decode(np.asarray(vector, dtype=float))
        started = self.elapsed
        try:
            result = self.fn(params, self.deadline)
        except BudgetExhausted:
            self.history.append({
                "status": "interrupted", "started": round(started, 2),
                "ended": round(self.elapsed, 2), "params": params,
            })
            raise
        value, info = (result if isinstance(result, tuple) else (result, {}))
        value = float(value)
        if value > self.best_value:
            self.best_value, self.best_params = value, params
        self.history.append({
            "status": "ok", "started": round(started, 2), "ended": round(self.elapsed, 2),
            "elapsed": round(self.elapsed, 2), "value": value, "best": self.best_value,
            "params": params, **info,
        })
        return value


# --------------------------------------------------------------------------- #
# Optimisers
# --------------------------------------------------------------------------- #
def random_search(objective: BudgetedObjective, seed: int = 0) -> HPOResult:
    rng = np.random.default_rng(seed)
    try:
        while not objective.exhausted:
            objective(objective.space.sample(rng))
    except BudgetExhausted:
        pass
    return _finish("random", objective)


def optuna_tpe(objective: BudgetedObjective, seed: int = 0) -> HPOResult:
    import optuna

    # Optuna logs a stack trace whenever a trial raises, including the exception
    # we use to stop it, which would bury the real output.
    optuna.logging.set_verbosity(optuna.logging.CRITICAL)
    space = objective.space

    def trial_objective(trial: "optuna.Trial") -> float:
        # Sampling in the unit cube keeps the space identical to the others.
        vector = np.array(
            [trial.suggest_float(p.name, 0.0, 1.0) for p in space.parameters], dtype=float
        )
        return objective(vector)

    # Default TPE: the first ten trials are random, the rest model-based.
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    try:
        remaining = max(1.0, objective.budget_seconds - objective.elapsed)
        study.optimize(trial_objective, n_trials=10_000, timeout=remaining, catch=())
    except BudgetExhausted:
        pass
    return _finish("tpe", objective)


def particle_swarm(
    objective: BudgetedObjective,
    seed: int = 0,
    n_particles: int = 6,
    inertia: float = 0.729,
    cognitive: float = 1.49445,
    social: float = 1.49445,
) -> HPOResult:
    """Standard constriction-coefficient PSO (Clerc & Kennedy, 2002).

    A swarm of six is small by the usual recommendation of 10-30, but the budget
    here buys only a few dozen objective evaluations. A larger swarm would spend
    the whole budget on one or two generations and degenerate into random search,
    which is precisely the comparison this study is trying to make honestly.
    """
    rng = np.random.default_rng(seed)
    space = objective.space

    position = rng.random((n_particles, space.dim))
    velocity = rng.uniform(-0.1, 0.1, size=(n_particles, space.dim))
    personal_best = position.copy()
    personal_value = np.full(n_particles, -np.inf)
    global_best, global_value = position[0].copy(), -np.inf

    try:
        for i in range(n_particles):
            personal_value[i] = objective(position[i])
            if personal_value[i] > global_value:
                global_value, global_best = personal_value[i], position[i].copy()

        while not objective.exhausted:
            r1 = rng.random((n_particles, space.dim))
            r2 = rng.random((n_particles, space.dim))
            velocity = (
                inertia * velocity
                + cognitive * r1 * (personal_best - position)
                + social * r2 * (global_best - position)
            )
            # Velocities are clipped to half the cube and positions to the cube.
            velocity = np.clip(velocity, -0.5, 0.5)
            position = np.clip(position + velocity, 0.0, 1.0)

            for i in range(n_particles):
                value = objective(position[i])
                if value > personal_value[i]:
                    personal_value[i], personal_best[i] = value, position[i].copy()
                if value > global_value:
                    global_value, global_best = value, position[i].copy()
    except BudgetExhausted:
        pass
    return _finish("pso", objective)


def genetic_algorithm(
    objective: BudgetedObjective,
    seed: int = 0,
    population_size: int = 6,
    crossover_rate: float = 0.9,
    mutation_rate: float = 0.2,
    mutation_scale: float = 0.15,
    elite: int = 2,
) -> HPOResult:
    """Real-coded GA: tournament selection, blend crossover, Gaussian mutation.

    Population matched to the swarm size for the same reason: enough generations
    to let selection act, given a budget of a few dozen evaluations.
    """
    rng = np.random.default_rng(seed)
    space = objective.space

    population = rng.random((population_size, space.dim))
    fitness = np.full(population_size, -np.inf)

    try:
        for i in range(population_size):
            fitness[i] = objective(population[i])

        while not objective.exhausted:
            order = np.argsort(-fitness)
            children = [population[i].copy() for i in order[:elite]]

            while len(children) < population_size:
                parent_a = _tournament(population, fitness, rng)
                parent_b = _tournament(population, fitness, rng)
                if rng.random() < crossover_rate:
                    # BLX-alpha: sample inside and slightly beyond the parents' box.
                    alpha = 0.5
                    low = np.minimum(parent_a, parent_b)
                    high = np.maximum(parent_a, parent_b)
                    span = high - low
                    child = rng.uniform(low - alpha * span, high + alpha * span)
                else:
                    child = parent_a.copy()

                mutate = rng.random(space.dim) < mutation_rate
                child[mutate] += rng.normal(0.0, mutation_scale, size=int(mutate.sum()))
                children.append(np.clip(child, 0.0, 1.0))

            population = np.array(children[:population_size])
            for i in range(population_size):
                fitness[i] = objective(population[i])
    except BudgetExhausted:
        pass
    return _finish("ga", objective)


def _tournament(population: np.ndarray, fitness: np.ndarray, rng, size: int = 3) -> np.ndarray:
    contenders = rng.choice(len(population), size=min(size, len(population)), replace=False)
    return population[contenders[np.argmax(fitness[contenders])]].copy()


def _finish(name: str, objective: BudgetedObjective) -> HPOResult:
    return HPOResult(
        optimiser=name,
        best_value=float(objective.best_value),
        best_params=objective.best_params,
        n_evaluations=objective.n_completed,
        elapsed_seconds=round(objective.elapsed, 2),
        history=objective.history,
    )


OPTIMISERS = {
    "random": random_search,
    "tpe": optuna_tpe,
    "pso": particle_swarm,
    "ga": genetic_algorithm,
}


OPTIMISER_IDS = {"random": 1, "tpe": 2, "pso": 3, "ga": 4}


def optimiser_seed(repeat: int, fold: int, optimiser: str, architecture: str) -> int:
    """Independent seed for one search, so no two searches share a start.

    Every optimiser previously drew from ``default_rng(0)``, which made the first
    population of PSO and GA identical to the first draws of random search.
    """
    arch_id = sum(ord(c) for c in architecture)
    sequence = np.random.SeedSequence([repeat, fold, OPTIMISER_IDS[optimiser], arch_id])
    return int(sequence.generate_state(1)[0])


def run_optimiser(
    name: str,
    fn: Callable,
    space: SearchSpace,
    budget_seconds: float,
    seed: int = 0,
) -> HPOResult:
    if name not in OPTIMISERS:
        raise KeyError(f"unknown optimiser {name!r}; available: {sorted(OPTIMISERS)}")
    if name == "tpe":
        import optuna  # noqa: F401 - imported before the clock starts, not on it
    objective = BudgetedObjective(fn, space, budget_seconds)
    LOGGER.info("%s: searching for %.0fs", name, budget_seconds)
    result = OPTIMISERS[name](objective, seed=seed)
    LOGGER.info(
        "%s: best=%.5f after %d completed evaluations in %.0fs",
        name, result.best_value, result.n_evaluations, result.elapsed_seconds,
    )
    return result
