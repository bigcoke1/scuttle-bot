import json
import statistics

from scuttle_bot.data.dataset import Dataset
from scuttle_bot.ml.nn.nn_model import NeuralNetworkModel
from scuttle_bot.ml.feature_encoder import FeatureEncoder
from scuttle_bot.ml.greedy_search import cross_val_metric, greedy_hyperparameter_search

MODELS_DIR = "src/scuttle_bot/ml/nn/models"
PLOTS_DIR = "src/scuttle_bot/ml/nn/plots"

# Models are selected by log loss (a proper scoring rule): unlike accuracy it
# scores the predicted win probability, penalizing confident-wrong predictions
# far more than borderline ones. Lower is better, so the greedy search minimizes.
SELECTION_METRIC = "log_loss"

# Only variant C is trained now -- it was the best-performing feature set
# across all three model types (draft + individual player ranked/mastery
# stats; no bans, no average-tier summary). Variants A/B/D were dropped.
VARIANT = "C"
FEATURE_CONFIG = dict(use_bans=False, use_avg_tier=False, use_player_stats=True)

# Candidate values per hyperparameter, swept in this order by the greedy
# coordinate-descent search (see ml/greedy_search.py). hidden_sizes is the
# network architecture (per-layer widths); the search picks depth/width too.
PARAM_GRID = {
    "hidden_sizes": [(64,), (128, 64), (256, 128, 64)],
    "dropout": [0.0, 0.2, 0.3],
    "lr": [1e-2, 1e-3, 1e-4],
    "weight_decay": [0.0, 1e-4, 1e-3],
}
# Starting point for the greedy sweep: the previously hardcoded variant-C net.
BASELINE = dict(hidden_sizes=(128, 64), dropout=0.2, lr=1e-3, weight_decay=0.0)

# Folds used to score each candidate during the search (kept small for speed).
SEARCH_RANDOM_STATES = [0, 1, 2]
# Folds for the final fit.
FINAL_RANDOM_STATES = [0, 1, 2, 3, 4]

# Fixed (not searched) training-loop settings.
EPOCHS = 50
TEST_SIZE = 0.2


def main():
    dataset = Dataset(db_path="src/scuttle_bot/cache/ml_dataset.db")
    df = dataset.retrieve_dataset()
    participants_df = dataset.retrieve_match_participants()
    print(f"{len(df)} matches")

    variant_models_dir = f"{MODELS_DIR}/{VARIANT}"
    variant_plots_dir = f"{PLOTS_DIR}/{VARIANT}"

    # The encoder is hyperparameter-independent, so fit it once and reuse it
    # for every candidate config and the final fit.
    encoder = FeatureEncoder(f"{variant_models_dir}/", **FEATURE_CONFIG)
    X, y = encoder.fit_transform(df, participants_df)
    input_size = X.shape[1]

    def make_model(params, random_state):
        return NeuralNetworkModel(
            input_size=input_size,
            random_state=random_state,
            test_size=TEST_SIZE,
            epochs=EPOCHS,
            **params,
        )

    def score_fn(params):
        return cross_val_metric(lambda rs: make_model(params, rs), X, y, SEARCH_RANDOM_STATES, metric_key=SELECTION_METRIC)

    print("\n=== Greedy hyperparameter search (NeuralNetwork, variant C) ===")
    best_params, best_search_log_loss, history = greedy_hyperparameter_search(
        score_fn, PARAM_GRID, baseline=BASELINE, greater_is_better=False
    )

    print(f"\n=== Final fit with best hyperparameters: {best_params} ===")
    per_fold = {"accuracy": [], "log_loss": [], "brier_score": []}
    for random_state in FINAL_RANDOM_STATES:
        print(f"\n--- Final model C, random_state={random_state} ---")
        subfix = f"_{VARIANT}_{random_state}"
        model = make_model(best_params, random_state)
        metrics = model.train(X, y, path_subfix=subfix, plots_dir=variant_plots_dir)
        model.save(path_subfix=subfix, output_dir=variant_models_dir)
        for key in per_fold:
            per_fold[key].append(metrics[key])

    means = {key: statistics.mean(vals) for key, vals in per_fold.items()}
    stds = {key: statistics.pstdev(vals) for key, vals in per_fold.items()}
    print(f"\nFinal C log_loss: {[f'{v:.4f}' for v in per_fold['log_loss']]}")
    print(f"Final C mean log_loss: {means['log_loss']:.4f} (+/- {stds['log_loss']:.4f}) "
          f"| mean accuracy: {means['accuracy']:.4f} | mean brier: {means['brier_score']:.4f}")

    summary = {
        "variant": VARIANT,
        "feature_config": FEATURE_CONFIG,
        "selection_metric": SELECTION_METRIC,
        "best_hyperparameters": best_params,
        "search": {
            "param_grid": PARAM_GRID,
            "search_random_states": SEARCH_RANDOM_STATES,
            "best_search_log_loss": best_search_log_loss,
            "configs_evaluated": len(history),
        },
        "final_random_states": FINAL_RANDOM_STATES,
        "final_per_fold": per_fold,
        "final_mean": means,
        "final_std": stds,
    }

    with open(f"{variant_models_dir}/cv_summary.json", "w") as f:
        json.dump(summary, f, indent=4, default=str)
    with open(f"{MODELS_DIR}/cv_summary.json", "w") as f:
        json.dump(summary, f, indent=4, default=str)


if __name__ == "__main__":
    main()
