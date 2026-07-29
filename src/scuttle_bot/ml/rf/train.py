import json
import statistics

from scuttle_bot.data.dataset import Dataset
from scuttle_bot.ml.rf.rf_model import RandomForestModel
from scuttle_bot.ml.feature_encoder import FeatureEncoder
from scuttle_bot.ml.greedy_search import cross_val_metric, greedy_hyperparameter_search

MODELS_DIR = "src/scuttle_bot/ml/rf/models"
PLOTS_DIR = "src/scuttle_bot/ml/rf/plots"

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
# coordinate-descent search (see ml/greedy_search.py). The search evaluates
# the sum of these counts per pass, not their product.
PARAM_GRID = {
    "n_estimators": [300, 500, 800],
    "max_depth": [None, 16, 24, 32],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", 0.3],
}
# Starting point for the greedy sweep: the previously hardcoded config.
BASELINE = dict(n_estimators=500, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features="sqrt")

# Folds used to score each candidate during the search (kept small for speed).
SEARCH_RANDOM_STATES = [0, 1, 2]
# Folds for the final fit. Includes random_state=3, whose model
# (rf_model_C_3.pkl) is the one WinPredictor loads and serves.
FINAL_RANDOM_STATES = [0, 1, 2, 3, 4]


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

    def make_model(params, random_state):
        return RandomForestModel(random_state=random_state, test_size=0.2, **params)

    def score_fn(params):
        return cross_val_metric(lambda rs: make_model(params, rs), X, y, SEARCH_RANDOM_STATES, metric_key=SELECTION_METRIC)

    print("\n=== Greedy hyperparameter search (RandomForest, variant C) ===")
    best_params, best_search_log_loss, history = greedy_hyperparameter_search(
        score_fn, PARAM_GRID, baseline=BASELINE, greater_is_better=False
    )

    # Final fit with the chosen hyperparameters across all folds, saving each
    # model + config + confusion-matrix plot. Keep the _C_{rs} naming so the
    # served artifact paths (rf_model_C_3.pkl, etc.) are overwritten in place.
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
