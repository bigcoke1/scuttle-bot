import json
import statistics

from scuttle_bot.data.dataset import Dataset
from scuttle_bot.ml.rf.rf_model import RandomForestModel
from scuttle_bot.ml.feature_encoder import FeatureEncoder

MODELS_DIR = "src/scuttle_bot/ml/rf/models"
PLOTS_DIR = "src/scuttle_bot/ml/rf/plots"

# A: draft only, B: draft + average tier, C: draft + individual player stats,
# D: draft + player stats + bans
MODEL_CONFIGS = {
    "A": dict(use_bans=False, use_avg_tier=False, use_player_stats=False),
    "B": dict(use_bans=False, use_avg_tier=True, use_player_stats=False),
    "C": dict(use_bans=False, use_avg_tier=False, use_player_stats=True),
    "D": dict(use_bans=True, use_avg_tier=False, use_player_stats=True),
}

# Each variant is trained 5 times on different train/test splits (one per
# random state) so we can report an average accuracy instead of a single split's.
# This is on top of, not a substitute for, each forest's own internal bagging
# (n_estimators trees each fit on a bootstrap resample) -- bagging reduces the
# variance of a single fitted forest, it doesn't give an outer accuracy estimate.
RANDOM_STATES = [0, 1, 2, 3, 4]


def train_variant(name, df, participants_df):
    config = MODEL_CONFIGS[name]
    print(f"\n=== Training model {name} ({config}) ===")

    variant_models_dir = f"{MODELS_DIR}/{name}"
    variant_plots_dir = f"{PLOTS_DIR}/{name}"

    # The encoder doesn't depend on random_state, so it's fit once per variant
    # and reused across all 5 runs.
    encoder = FeatureEncoder(f"{variant_models_dir}/", **config)
    X, y = encoder.fit_transform(df, participants_df)

    accuracies = []
    for random_state in RANDOM_STATES:
        print(f"\n--- Model {name}, random_state={random_state} ---")
        subfix = f"_{name}_{random_state}"

        model = RandomForestModel(
            random_state=random_state,
            test_size=0.2,
            n_estimators=500
        )

        metrics = model.train(X, y, path_subfix=subfix, plots_dir=variant_plots_dir)
        model.save(path_subfix=subfix, output_dir=variant_models_dir)
        accuracies.append(metrics["accuracy"])

    mean_accuracy = statistics.mean(accuracies)
    std_accuracy = statistics.pstdev(accuracies)

    print(f"\nModel {name} accuracies: {[f'{a:.4f}' for a in accuracies]}")
    print(f"Model {name} mean accuracy: {mean_accuracy:.4f} (+/- {std_accuracy:.4f})")

    summary = {
        "variant": name,
        "config": config,
        "random_states": RANDOM_STATES,
        "accuracies": accuracies,
        "mean_accuracy": mean_accuracy,
        "std_accuracy": std_accuracy,
    }

    with open(f"{variant_models_dir}/cv_summary.json", "w") as f:
        json.dump(summary, f, indent=4)

    return summary


def model_A(df, participants_df):
    return train_variant("A", df, participants_df)


def model_B(df, participants_df):
    return train_variant("B", df, participants_df)


def model_C(df, participants_df):
    return train_variant("C", df, participants_df)


def model_D(df, participants_df):
    return train_variant("D", df, participants_df)


def main():
    dataset = Dataset(db_path="src/scuttle_bot/cache/ml_dataset.db")
    df = dataset.retrieve_dataset()
    participants_df = dataset.retrieve_match_participants()

    print(len(df))

    summaries = [
        model_A(df, participants_df),
        model_B(df, participants_df),
        model_C(df, participants_df),
        model_D(df, participants_df),
    ]

    print("\n=== Summary (mean accuracy over 5 random states) ===")
    for summary in summaries:
        print(f"Model {summary['variant']}: {summary['mean_accuracy']:.4f} (+/- {summary['std_accuracy']:.4f})")

    with open(f"{MODELS_DIR}/cv_summary.json", "w") as f:
        json.dump(summaries, f, indent=4)


if __name__ == "__main__":
    main()
