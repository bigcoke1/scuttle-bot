import json
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

MODELS_DIR = "src/scuttle_bot/ml/nn/models"
PLOTS_DIR = "src/scuttle_bot/ml/nn/plots"


class NNModel(nn.Module):
    """
    Configurable MLP. hidden_sizes sets network complexity: a short/narrow
    tuple gives a shallow net (variant A), a long/wide one a deep net (D).
    """

    def __init__(self, input_size, hidden_sizes=(64,), output_size=1, dropout=0.0):
        super().__init__()
        layers = []
        in_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_size = hidden_size
        layers.append(nn.Linear(in_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class NeuralNetworkModel:
    def __init__(
        self,
        input_size,
        hidden_sizes=(64,),
        dropout=0.0,
        random_state=42,
        test_size=0.2,
        lr=1e-3,
        weight_decay=0.0,
        epochs=50,
        batch_size=64,
        encoder_path="src/scuttle_bot/ml/nn/artifacts/encoder.pkl"
    ):
        self.input_size = input_size
        self.hidden_sizes = tuple(hidden_sizes)
        self.dropout = dropout
        self.random_state = random_state
        self.test_size = test_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.encoder_path = encoder_path

        torch.manual_seed(self.random_state)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

        self.model = NNModel(
            input_size=self.input_size,
            hidden_sizes=self.hidden_sizes,
            output_size=1,
            dropout=self.dropout
        ).to(self.device)

        self.metrics = {}

    def _to_dense(self, X):
        return X.toarray() if hasattr(X, "toarray") else np.asarray(X)

    def train(self, X, y, path_subfix="", plots_dir=None):
        """
        Train the neural network
        """

        X_dense = self._to_dense(X).astype(np.float32)
        y_arr = np.asarray(y, dtype=np.float32).reshape(-1, 1)

        X_train, X_test, y_train, y_test = train_test_split(
            X_dense,
            y_arr,
            test_size=self.test_size,
            random_state=self.random_state
        )

        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
            batch_size=self.batch_size,
            shuffle=True
        )

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                optimizer.zero_grad()
                logits = self.model(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * X_batch.size(0)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch + 1}/{self.epochs} - loss: {epoch_loss / len(train_loader.dataset):.4f}")

        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.from_numpy(X_test).to(self.device))
            probs = torch.sigmoid(logits).cpu().numpy()
        predictions = (probs >= 0.5).astype(int).reshape(-1)
        y_test_labels = y_test.reshape(-1).astype(int)

        accuracy = accuracy_score(y_test_labels, predictions)

        self.metrics = {
            "accuracy": accuracy,
            "classification_report": classification_report(
                y_test_labels,
                predictions,
                output_dict=True
            )
        }

        print(f"Accuracy: {accuracy:.4f}")

        self.plot_confusion_matrix(y_test_labels, predictions, path_subfix, output_dir=plots_dir)

        return self.metrics

    def plot_confusion_matrix(self, y_test, predictions, path_subfix="", output_dir=None):
        """
        Plot and save the confusion matrix for a trained model's predictions.
        """

        output_dir = output_dir or PLOTS_DIR
        os.makedirs(output_dir, exist_ok=True)

        labels = ["Red Win", "Blue Win"]
        cm = confusion_matrix(y_test, predictions, labels=[0, 1])

        fig, ax = plt.subplots(figsize=(6, 6))
        ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels).plot(
            ax=ax, cmap="Blues", colorbar=False
        )
        model_name = path_subfix.lstrip("_") or "Neural Network"
        ax.set_title(f"Confusion Matrix - Model {model_name}")

        plot_path = f"{output_dir}/confusion_matrix{path_subfix}.png"
        fig.savefig(plot_path, bbox_inches="tight")
        plt.close(fig)

        print(f"Confusion matrix saved to {plot_path}")

    def predict(self, X):
        self.model.eval()
        X_dense = self._to_dense(X).astype(np.float32)
        with torch.no_grad():
            logits = self.model(torch.from_numpy(X_dense).to(self.device))
            probs = torch.sigmoid(logits).cpu().numpy()
        return (probs >= 0.5).astype(int).reshape(-1)

    def predict_proba(self, X):
        self.model.eval()
        X_dense = self._to_dense(X).astype(np.float32)
        with torch.no_grad():
            logits = self.model(torch.from_numpy(X_dense).to(self.device))
            probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)
        return np.column_stack([1 - probs, probs])

    def save(self, path_subfix="", output_dir=None):
        """
        Save model weights and training config
        """

        output_dir = output_dir or MODELS_DIR
        os.makedirs(output_dir, exist_ok=True)

        model_path = f"{output_dir}/nn_model{path_subfix}.pt"
        config_path = f"{output_dir}/nn_config{path_subfix}.json"

        torch.save(self.model.state_dict(), model_path)

        config = {
            "model_type": "NNModel",
            "input_size": self.input_size,
            "hidden_sizes": list(self.hidden_sizes),
            "dropout": self.dropout,
            "random_state": self.random_state,
            "test_size": self.test_size,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "metrics": self.metrics,
            "encoder_path": self.encoder_path,
            "saved_at": datetime.now().isoformat()
        }

        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)

        print(f"Model saved to {model_path}")
        print(f"Config saved to {config_path}")

    def load(self, model_path="src/scuttle_bot/ml/nn/nn_model.pt"):
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        print("Model loaded successfully")
