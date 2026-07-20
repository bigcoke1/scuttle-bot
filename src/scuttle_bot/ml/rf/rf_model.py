import json
import os
import joblib
from datetime import datetime
from typing import Literal, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

MODELS_DIR = "src/scuttle_bot/ml/rf/models"
PLOTS_DIR = "src/scuttle_bot/ml/rf/plots"

class RandomForestModel:
    def __init__(
        self,
        random_state=42,
        test_size=0.2,
        n_estimators=500,
        max_depth: Optional[int] = None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features: Literal['sqrt', 'log2'] | int | float | None = 'sqrt',
        n_jobs=-1,
        encoder_path="src/scuttle_bot/ml/rf/artifacts/encoder.pkl"
    ):
        self.random_state = random_state
        self.test_size = test_size
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.n_jobs = n_jobs
        self.encoder_path = encoder_path

        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )

        self.metrics = {}

    def train(self, X, y, path_subfix="", plots_dir=None):
        """
        Train random forest model
        """

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state
        )

        self.model.fit(X_train, y_train)

        predictions = self.model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)

        self.metrics = {
            "accuracy": accuracy,
            "classification_report": classification_report(
                y_test,
                predictions,
                output_dict=True
            )
        }

        print(f"Accuracy: {accuracy:.4f}")

        self.plot_confusion_matrix(y_test, predictions, path_subfix, output_dir=plots_dir)

        return self.metrics

    def plot_confusion_matrix(self, y_test, predictions, path_subfix="", output_dir=None):
        """
        Plot and save the confusion matrix for a trained model's predictions.
        """

        output_dir = output_dir or PLOTS_DIR
        os.makedirs(output_dir, exist_ok=True)

        labels = ["Red Win" if not c else "Blue Win" for c in self.model.classes_]
        cm = confusion_matrix(y_test, predictions, labels=self.model.classes_)

        fig, ax = plt.subplots(figsize=(6, 6))
        ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels).plot(
            ax=ax, cmap="Blues", colorbar=False
        )
        model_name = path_subfix.lstrip("_") or "Random Forest"
        ax.set_title(f"Confusion Matrix - Model {model_name}")

        plot_path = f"{output_dir}/confusion_matrix{path_subfix}.png"
        fig.savefig(plot_path, bbox_inches="tight")
        plt.close(fig)

        print(f"Confusion matrix saved to {plot_path}")

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def save(self, path_subfix="", output_dir=None):
        """
        Save model and training config
        """

        output_dir = output_dir or MODELS_DIR
        os.makedirs(output_dir, exist_ok=True)

        model_path = f"{output_dir}/rf_model{path_subfix}.pkl"
        config_path = f"{output_dir}/rf_config{path_subfix}.json"

        # Save sklearn model
        joblib.dump(self.model, model_path)

        # Save metadata/config
        config = {
            "model_type": "RandomForestClassifier",
            "random_state": self.random_state,
            "test_size": self.test_size,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "max_features": self.max_features,
            "metrics": self.metrics,
            "encoder_path": self.encoder_path,
            "saved_at": datetime.now().isoformat()
        }

        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)

        print(f"Model saved to {model_path}")
        print(f"Config saved to {config_path}")

    def load(self, model_path="src/scuttle_bot/ml/rf/rf_model.pkl"):
        self.model = joblib.load(model_path)
        print("Model loaded successfully")
