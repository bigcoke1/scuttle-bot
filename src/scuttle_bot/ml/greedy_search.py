"""Greedy (coordinate-descent) hyperparameter search, shared by the rf,
logistic, and nn trainers.

A full grid search over k hyperparameters costs the *product* of their
candidate counts. This instead optimizes one hyperparameter at a time -- hold
the rest at their current best, sweep a single hyperparameter's candidates,
keep the winner, move to the next -- so the cost is the *sum* of the candidate
counts (per pass). It's near-optimal rather than globally optimal: a purely
greedy sweep can miss interactions between hyperparameters. Repeating the
sweep until a full pass yields no improvement recovers most of what a single
pass would miss, while still evaluating far fewer configs than a grid.
"""

import statistics
from typing import Callable


def cross_val_metric(model_factory: Callable, X, y, random_states, metric_key: str = "log_loss", plot: bool = False) -> float:
    """Mean of ``model.train(...).metrics[metric_key]`` for
    ``model_factory(random_state)`` across several train/test splits (one per
    random_state). metric_key selects which metric the search optimizes
    (e.g. 'log_loss', 'brier_score', 'accuracy'). Plotting/saving stay off by
    default because this is called once per candidate config in the search."""
    values = []
    for random_state in random_states:
        model = model_factory(random_state)
        metrics = model.train(X, y, plot=plot)
        values.append(metrics[metric_key])
    return statistics.mean(values)


def greedy_hyperparameter_search(
    score_fn: Callable[[dict], float],
    param_grid: dict[str, list],
    baseline: dict | None = None,
    max_passes: int = 2,
    greater_is_better: bool = True,
) -> tuple[dict, float, list]:
    """Coordinate-descent hyperparameter search.

    Args:
        score_fn: maps a full hyperparameter dict to a scalar score.
        param_grid: {hyperparameter: [candidate values...]}, swept in
            insertion order.
        baseline: starting value for each hyperparameter; any key missing
            here (or absent from param_grid) falls back to that grid's first
            listed value.
        max_passes: maximum number of full sweeps over every hyperparameter.
            Stops early as soon as a whole pass produces no improvement, so a
            converged search costs less than this cap.
        greater_is_better: whether a higher score is better (e.g. accuracy) or
            a lower one (e.g. log loss, which is the default selection metric).

    Returns:
        (best_params, best_score, history) where history is a list of every
        config evaluated, in order, for logging/inspection.
    """
    def is_better(candidate: float, incumbent: float) -> bool:
        return candidate > incumbent if greater_is_better else candidate < incumbent

    best_params = {name: values[0] for name, values in param_grid.items()}
    if baseline:
        best_params.update({k: v for k, v in baseline.items() if k in param_grid})

    history: list[dict] = []
    best_score = score_fn(best_params)
    history.append({"event": "baseline", "params": dict(best_params), "score": best_score})
    print(f"[greedy] baseline {best_params} -> {best_score:.4f}")

    for pass_idx in range(max_passes):
        improved = False
        for name, values in param_grid.items():
            for value in values:
                if value == best_params[name]:
                    continue  # already the current best for this hyperparameter
                trial = dict(best_params)
                trial[name] = value
                score = score_fn(trial)
                history.append({"event": f"pass{pass_idx}", "param": name, "value": value, "score": score})

                marker = ""
                if is_better(score, best_score):
                    best_score = score
                    best_params = trial
                    improved = True
                    marker = "  <- new best"
                print(f"[greedy] pass {pass_idx}: {name}={value} -> {score:.4f}{marker}")

        if not improved:
            print(f"[greedy] pass {pass_idx}: no improvement; stopping.")
            break

    print(f"[greedy] best {best_params} -> {best_score:.4f}")
    return best_params, best_score, history
