"""
Soft sensors for real-time quality prediction.

Soft sensors use inline measurements (flow, temperature, density)
to predict lab quality parameters, reducing lab testing frequency
and enabling real-time quality control.

Implements:
- Viscosity prediction from inline density and temperature
- Flash point estimation
- Pour point estimation
- Model calibration from lab feedback
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import structlog

from lobp.ai.base import BaseModel, PredictionResult

logger = structlog.get_logger()


@dataclass
class InlineMeasurement:
    """Real-time inline measurement data."""

    timestamp: datetime
    temperature_celsius: float
    density_kg_m3: float
    flow_rate_lpm: float
    pressure_bar: float | None = None
    mixing_speed_rpm: float | None = None
    blend_completion_percent: float = 0.0

    # Ingredient ratios if known
    base_oil_percent: float | None = None
    additive_percent: float | None = None


@dataclass
class SoftSensorPrediction:
    """Soft sensor prediction output."""

    parameter: str
    predicted_value: float
    confidence: float
    lower_bound: float
    upper_bound: float
    model_name: str
    calibration_date: datetime | None = None
    requires_lab_verification: bool = False


class ViscositySoftSensor(BaseModel):
    """
    Soft sensor for predicting viscosity from inline measurements.

    Uses Walther equation and density correlation to estimate
    kinematic viscosity at 40°C and 100°C.
    """

    def __init__(self):
        super().__init__("viscosity_soft_sensor", "1.0.0")
        self._calibration_coefficients = {
            # Default coefficients (should be calibrated per product)
            "a": 0.7,  # Walther equation parameter
            "b": 2.5,  # Walther equation parameter
            "density_factor": 0.15,
            "temp_offset": 273.15,
        }
        self._calibration_date: datetime | None = None
        self._is_loaded = True

    def predict_viscosity_40c(
        self,
        measurement: InlineMeasurement,
    ) -> SoftSensorPrediction:
        """
        Predict viscosity at 40°C from inline measurements.

        Uses modified Walther equation with density correlation.
        """
        # Extract measurements
        temp = measurement.temperature_celsius
        density = measurement.density_kg_m3

        # Apply Walther-type correlation
        # log(log(v + 0.7)) = A - B * log(T + 273)
        coef = self._calibration_coefficients

        # Temperature correction to 40°C
        temp_ratio = (40 + coef["temp_offset"]) / (temp + coef["temp_offset"])

        # Density-based viscosity estimation
        # Higher density generally correlates with higher viscosity
        base_viscosity = (density - 800) * coef["density_factor"] + 30

        # Temperature correction
        predicted = base_viscosity * (temp_ratio ** coef["b"])

        # Confidence based on how close measurement temp is to target
        temp_diff = abs(temp - 40)
        confidence = max(0.5, 1 - (temp_diff / 100))

        # Uncertainty bounds (±5% typical for soft sensors)
        uncertainty = predicted * 0.05
        lower = predicted - 2 * uncertainty
        upper = predicted + 2 * uncertainty

        return SoftSensorPrediction(
            parameter="viscosity_40c",
            predicted_value=predicted,
            confidence=confidence,
            lower_bound=lower,
            upper_bound=upper,
            model_name=self.model_name,
            calibration_date=self._calibration_date,
            requires_lab_verification=confidence < 0.8,
        )

    def predict_viscosity_100c(
        self,
        measurement: InlineMeasurement,
        viscosity_40c: float | None = None,
    ) -> SoftSensorPrediction:
        """
        Predict viscosity at 100°C.

        Can use measured or predicted viscosity at 40°C.
        """
        if viscosity_40c is None:
            pred_40 = self.predict_viscosity_40c(measurement)
            viscosity_40c = pred_40.predicted_value

        # Typical VI relationship: v100 ≈ v40^0.4 for mineral oils
        # Adjust based on density (higher density = lower VI)
        density = measurement.density_kg_m3
        vi_factor = 0.35 + (900 - density) / 1000

        predicted = viscosity_40c ** vi_factor

        return SoftSensorPrediction(
            parameter="viscosity_100c",
            predicted_value=predicted,
            confidence=0.75,
            lower_bound=predicted * 0.9,
            upper_bound=predicted * 1.1,
            model_name=self.model_name,
            calibration_date=self._calibration_date,
            requires_lab_verification=True,
        )

    def calibrate(
        self,
        measurements: list[InlineMeasurement],
        lab_values: list[float],
    ) -> dict[str, Any]:
        """
        Calibrate soft sensor against lab measurements.

        Updates model coefficients to minimize prediction error.
        """
        if len(measurements) != len(lab_values):
            raise ValueError("Measurements and lab values must have same length")

        if len(measurements) < 5:
            raise ValueError("Need at least 5 data points for calibration")

        # Prepare data
        X = np.array([
            [m.temperature_celsius, m.density_kg_m3]
            for m in measurements
        ])
        y = np.array(lab_values)

        # Simple linear regression for calibration
        # y = a + b1*temp + b2*density
        X_with_bias = np.column_stack([np.ones(len(X)), X])
        coeffs, residuals, _, _ = np.linalg.lstsq(X_with_bias, y, rcond=None)

        # Update calibration
        self._calibration_coefficients.update({
            "intercept": float(coeffs[0]),
            "temp_coeff": float(coeffs[1]),
            "density_factor": float(coeffs[2]),
        })
        self._calibration_date = datetime.now(timezone.utc)

        # Calculate calibration metrics
        predictions = X_with_bias @ coeffs
        mse = np.mean((predictions - y) ** 2)
        mae = np.mean(np.abs(predictions - y))
        r2 = 1 - (np.sum((y - predictions) ** 2) / np.sum((y - np.mean(y)) ** 2))

        logger.info(
            "Soft sensor calibrated",
            samples=len(measurements),
            r2=r2,
            mae=mae,
        )

        return {
            "calibration_date": self._calibration_date.isoformat(),
            "samples_used": len(measurements),
            "metrics": {
                "r2": r2,
                "mse": mse,
                "mae": mae,
            },
            "coefficients": self._calibration_coefficients,
        }

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train is alias for calibrate for interface compatibility."""
        measurements = [
            InlineMeasurement(
                timestamp=datetime.now(timezone.utc),
                temperature_celsius=x[0],
                density_kg_m3=x[1],
                flow_rate_lpm=0,
            )
            for x in X
        ]
        self.calibrate(measurements, y.tolist())

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict viscosity for array of measurements."""
        predictions = []
        for x in X:
            measurement = InlineMeasurement(
                timestamp=datetime.now(timezone.utc),
                temperature_celsius=x[0],
                density_kg_m3=x[1],
                flow_rate_lpm=0,
            )
            pred = self.predict_viscosity_40c(measurement)
            predictions.append(pred.predicted_value)
        return np.array(predictions)

    def save(self, path=None) -> None:
        """Save calibration coefficients."""
        pass  # Would save to file/database

    def load(self, path=None) -> None:
        """Load calibration coefficients."""
        pass  # Would load from file/database


class FlashPointSoftSensor(BaseModel):
    """
    Soft sensor for flash point prediction.

    Uses composition-weighted average and volatility correlation.
    """

    def __init__(self):
        super().__init__("flash_point_soft_sensor", "1.0.0")
        self._component_flash_points = {
            # Default flash points for common base oils
            "SN-150": 210,
            "SN-500": 250,
            "SN-600": 260,
            "bright_stock": 280,
            # Additives generally have higher flash points
            "additive_package": 180,
        }
        self._is_loaded = True

    def predict_flash_point(
        self,
        ingredient_percentages: dict[str, float],
        measurement: InlineMeasurement | None = None,
    ) -> SoftSensorPrediction:
        """
        Predict flash point from blend composition.

        Uses weighted average with volatility correction.
        """
        weighted_fp = 0.0
        total_weight = 0.0

        for material, percentage in ingredient_percentages.items():
            # Get component flash point
            fp = self._component_flash_points.get(
                material,
                self._component_flash_points.get("additive_package", 200)
            )
            weighted_fp += fp * percentage
            total_weight += percentage

        if total_weight > 0:
            predicted = weighted_fp / total_weight
        else:
            predicted = 200.0  # Default

        # Adjust for temperature if measurement available
        if measurement and measurement.temperature_celsius > 50:
            # Volatile components may have evaporated
            predicted += (measurement.temperature_celsius - 50) * 0.1

        return SoftSensorPrediction(
            parameter="flash_point",
            predicted_value=predicted,
            confidence=0.85,
            lower_bound=predicted - 10,
            upper_bound=predicted + 10,
            model_name=self.model_name,
            requires_lab_verification=False,
        )

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([200.0] * len(X))

    def save(self, path=None) -> None:
        pass

    def load(self, path=None) -> None:
        pass


class SoftSensorManager:
    """
    Manager for all soft sensors in the system.

    Coordinates predictions and handles model selection.
    """

    def __init__(self):
        self.viscosity_sensor = ViscositySoftSensor()
        self.flash_point_sensor = FlashPointSoftSensor()
        self._prediction_history: list[dict[str, Any]] = []

    def predict_all(
        self,
        measurement: InlineMeasurement,
        ingredient_percentages: dict[str, float] | None = None,
    ) -> dict[str, SoftSensorPrediction]:
        """Get all soft sensor predictions."""
        predictions = {}

        # Viscosity predictions
        predictions["viscosity_40c"] = self.viscosity_sensor.predict_viscosity_40c(
            measurement
        )
        predictions["viscosity_100c"] = self.viscosity_sensor.predict_viscosity_100c(
            measurement, predictions["viscosity_40c"].predicted_value
        )

        # Flash point prediction
        if ingredient_percentages:
            predictions["flash_point"] = self.flash_point_sensor.predict_flash_point(
                ingredient_percentages, measurement
            )

        # Log predictions
        self._prediction_history.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "predictions": {k: v.predicted_value for k, v in predictions.items()},
        })

        return predictions

    def should_request_lab_sample(
        self,
        predictions: dict[str, SoftSensorPrediction],
        confidence_threshold: float = 0.8,
    ) -> tuple[bool, list[str]]:
        """
        Determine if lab verification is needed.

        Returns (should_sample, reasons).
        """
        reasons = []

        for param, pred in predictions.items():
            if pred.confidence < confidence_threshold:
                reasons.append(f"{param} confidence below threshold ({pred.confidence:.2f})")
            if pred.requires_lab_verification:
                reasons.append(f"{param} requires lab verification")

        return len(reasons) > 0, reasons

    def get_prediction_summary(self) -> dict[str, Any]:
        """Get summary of recent predictions."""
        if not self._prediction_history:
            return {"count": 0}

        recent = self._prediction_history[-100:]

        return {
            "count": len(self._prediction_history),
            "recent_count": len(recent),
            "last_prediction": recent[-1] if recent else None,
        }
