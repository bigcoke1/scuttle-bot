from scuttle_bot.data.dataset import Dataset
from scuttle_bot.ml.logistic.logistic_model import LogisticModel
from scuttle_bot.ml.feature_encoder import FeatureEncoder

ENCODER_PATH = "src/scuttle_bot/ml/logistic/"

def main():
    dataset = Dataset(db_path="src/scuttle_bot/cache/ml_dataset.db")
    df = dataset.retrieve_dataset()
    participants_df = dataset.retrieve_match_participants()

    print(df["average_tier"].value_counts())
    print(df["match_id"].nunique())
    print(len(df))

    """
    encoder = FeatureEncoder(ENCODER_PATH)

    X, y = encoder.fit_transform(df, participants_df)

    model = LogisticModel(
        test_size=0.2,
        C=1.0,
        max_iter=5000  # slightly safer for convergence
    )

    model.train(X, y)
    model.save()
    """


if __name__ == "__main__":
    main()