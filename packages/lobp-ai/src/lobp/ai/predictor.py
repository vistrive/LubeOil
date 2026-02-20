"""Quality prediction models for blend properties."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import structlog

from lobp.ai.base import BaseModel, ModelMetadata, PredictionResult
from lobp.core.config import settings

logger = structlog.get_logger()


class QualityPredictor(BaseModel):
    """
    Neural network-based quality predictor for blend properties.

    Predicts key quality parameters:
    - Viscosity at 40°C and 100°C
    - Flash point
    - Pour point
    - Density
    - TBN (Total Base Number)

    Uses historical blend data and real-time sensor readings
    to predict final blend quality before completion.
    """

    def __init__(self, version: str = "1.0.0"):
        super().__init__("quality_predictor", version)
        self._feature_names: list[str] = []
        self._target_names: list[str] = [
            "viscosity_40c",
            "viscosity_100c",
            "flash_point",
            "pour_point",
            "density",
            "tbn",
        ]
        self._scaler_params: dict[str, Any] = {}

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the quality prediction model.

        Args:
            X: Feature matrix (ingredient ratios, process parameters)
            y: Target matrix (quality parameters)
        """
        if not self.validate_input(X) or not self.validate_input(y):
            raise ValueError("Invalid training data")

        logger.info(
            "Training quality predictor",
            samples=len(X),
            features=X.shape[1] if len(X.shape) > 1 else 1,
            targets=y.shape[1] if len(y.shape) > 1 else 1,
        )

        # Normalize features
        self._scaler_params = {
            "X_mean": np.mean(X, axis=0),
            "X_std": np.std(X, axis=0) + 1e-8,
            "y_mean": np.mean(y, axis=0),
            "y_std": np.std(y, axis=0) + 1e-8,
        }

        X_scaled = (X - self._scaler_params["X_mean"]) / self._scaler_params["X_std"]
        y_scaled = (y - self._scaler_params["y_mean"]) / self._scaler_params["y_std"]

        # Simple neural network weights (placeholder for full TensorFlow model)
        # In production, this would use tf.keras
        n_features = X_scaled.shape[1]
        n_targets = y_scaled.shape[1]
        hidden_size = 64

        # Initialize weights (Xavier initialization)
        self._model = {
            "W1": np.random.randn(n_features, hidden_size) * np.sqrt(2.0 / n_features),
            "b1": np.zeros(hidden_size),
            "W2": np.random.randn(hidden_size, hidden_size) * np.sqrt(2.0 / hidden_size),
            "b2": np.zeros(hidden_size),
            "W3": np.random.randn(hidden_size, n_targets) * np.sqrt(2.0 / hidden_size),
            "b3": np.zeros(n_targets),
        }

        # Training loop (simplified gradient descent)
        learning_rate = 0.001
        epochs = 100

        for epoch in range(epochs):
            # Forward pass
            z1 = X_scaled @ self._model["W1"] + self._model["b1"]
            a1 = np.maximum(0, z1)  # ReLU
            z2 = a1 @ self._model["W2"] + self._model["b2"]
            a2 = np.maximum(0, z2)  # ReLU
            y_pred = a2 @ self._model["W3"] + self._model["b3"]

            # Compute loss
            loss = np.mean((y_pred - y_scaled) ** 2)

            # Backward pass (simplified)
            d3 = 2 * (y_pred - y_scaled) / len(X)
            dW3 = a2.T @ d3
            db3 = np.sum(d3, axis=0)

            d2 = (d3 @ self._model["W3"].T) * (z2 > 0)
            dW2 = a1.T @ d2
            db2 = np.sum(d2, axis=0)

            d1 = (d2 @ self._model["W2"].T) * (z1 > 0)
            dW1 = X_scaled.T @ d1
            db1 = np.sum(d1, axis=0)

            # Update weights
            self._model["W3"] -= learning_rate * dW3
            self._model["b3"] -= learning_rate * db3
            self._model["W2"] -= learning_rate * dW2
            self._model["b2"] -= learning_rate * db2
            self._model["W1"] -= learning_rate * dW1
            self._model["b1"] -= learning_rate * db1

        # Calculate accuracy on training data
        y_pred_final = self._forward(X_scaled)
        mse = np.mean((y_pred_final - y_scaled) ** 2)
        r2 = 1 - mse / np.var(y_scaled)

        self._metadata = ModelMetadata(
            name=self.model_name,
            version=self.version,
            created_at=datetime.now(timezone.utc),
            trained_samples=len(X),
            accuracy=float(r2),
            parameters={"hidden_size": hidden_size, "epochs": epochs},
        )

        self._is_loaded = True
        logger.info("Training complete", accuracy=r2, loss=loss)

    def _forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass through the network."""
        z1 = X @ self._model["W1"] + self._model["b1"]
        a1 = np.maximum(0, z1)
        z2 = a1 @ self._model["W2"] + self._model["b2"]
        a2 = np.maximum(0, z2)
        return a2 @ self._model["W3"] + self._model["b3"]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict quality parameters for given inputs.

        Args:
            X: Feature matrix (ingredient ratios, process parameters)

        Returns:
            Predicted quality parameters
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call train() or load() first.")

        if not self.validate_input(X):
            raise ValueError("Invalid input data")

        # Scale input
        X_scaled = (X - self._scaler_params["X_mean"]) / self._scaler_params["X_std"]

        # Predict
        y_scaled = self._forward(X_scaled)

        # Inverse scale
        y = y_scaled * self._scaler_params["y_std"] + self._scaler_params["y_mean"]
        return y

    def predict_with_confidence(
        self, X: np.ndarray
    ) -> list[dict[str, PredictionResult]]:
        """
        Predict with confidence intervals.

        Args:
            X: Feature matrix

        Returns:
            List of prediction results with confidence intervals for each target
        """
        predictions = self.predict(X)
        confidence = self.get_confidence(X)

        results = []
        for i, pred_row in enumerate(predictions):
            row_results = {}
            for j, target_name in enumerate(self._target_names):
                # Estimate confidence interval (simplified)
                std_estimate = abs(pred_row[j]) * 0.05  # 5% std estimate
                row_results[target_name] = PredictionResult(
                    value=float(pred_row[j]),
                    lower_bound=float(pred_row[j] - 2 * std_estimate),
                    upper_bound=float(pred_row[j] + 2 * std_estimate),
                    confidence=float(confidence[i]),
                    model_name=self.model_name,
                    model_version=self.version,
                )
            results.append(row_results)

        return results

    def assess_off_spec_risk(
        self,
        predictions: dict[str, float],
        targets: dict[str, float],
        tolerances: dict[str, float],
    ) -> tuple[float, list[str]]:
        """
        Assess off-spec risk based on predictions vs targets.

        Args:
            predictions: Predicted quality values
            targets: Target specification values
            tolerances: Tolerance ranges for each parameter

        Returns:
            Tuple of (risk percentage, list of risk factors)
        """
        risk_factors = []
        total_deviation = 0.0
        param_count = 0

        for param in ["viscosity_40c", "flash_point", "pour_point"]:
            if param in predictions and param in targets:
                pred = predictions[param]
                target = targets[param]
                tolerance = tolerances.get(param, 5.0)

                deviation = abs(pred - target)
                normalized_deviation = deviation / tolerance

                if normalized_deviation > 1.0:
                    risk_factors.append(
                        f"{param}: predicted {pred:.2f}, target {target:.2f} "
                        f"(deviation: {deviation:.2f})"
                    )

                total_deviation += min(normalized_deviation, 2.0)
                param_count += 1

        # Calculate risk percentage
        if param_count > 0:
            avg_deviation = total_deviation / param_count
            risk_percent = min(100.0, avg_deviation * 50)  # Scale to percentage
        else:
            risk_percent = 0.0

        return risk_percent, risk_factors

    def save(self, path: Path | None = None) -> None:
        """Save model to disk."""
        save_path = path or self.model_path
        save_path.mkdir(parents=True, exist_ok=True)

        # Save weights
        np.savez(
            save_path / "weights.npz",
            **{k: v for k, v in self._model.items()},
        )

        # Save scaler params
        np.savez(
            save_path / "scaler.npz",
            **self._scaler_params,
        )

        # Save metadata
        if self._metadata:
            with open(save_path / "metadata.json", "w") as f:
                json.dump(
                    {
                        "name": self._metadata.name,
                        "version": self._metadata.version,
                        "created_at": self._metadata.created_at.isoformat(),
                        "trained_samples": self._metadata.trained_samples,
                        "accuracy": self._metadata.accuracy,
                        "parameters": self._metadata.parameters,
                    },
                    f,
                    indent=2,
                )

        logger.info("Model saved", path=str(save_path))

    def load(self, path: Path | None = None) -> None:
        """Load model from disk."""
        load_path = path or self.model_path

        if not load_path.exists():
            raise FileNotFoundError(f"Model not found at {load_path}")

        # Load weights
        weights = np.load(load_path / "weights.npz")
        self._model = {k: weights[k] for k in weights.files}

        # Load scaler params
        scaler = np.load(load_path / "scaler.npz")
        self._scaler_params = {k: scaler[k] for k in scaler.files}

        # Load metadata
        with open(load_path / "metadata.json") as f:
            meta = json.load(f)
            self._metadata = ModelMetadata(
                name=meta["name"],
                version=meta["version"],
                created_at=datetime.fromisoformat(meta["created_at"]),
                trained_samples=meta["trained_samples"],
                accuracy=meta["accuracy"],
                parameters=meta["parameters"],
            )

        self._is_loaded = True
        logger.info("Model loaded", path=str(load_path))
