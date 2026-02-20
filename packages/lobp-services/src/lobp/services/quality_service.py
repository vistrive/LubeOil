"""Quality service for measurements and predictions."""

from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from lobp.models.quality import (
    MeasurementSource,
    QualityMeasurement,
    QualityPrediction,
    QualityStatus,
)
from lobp.schemas.quality import (
    QualityMeasurementCreate,
    QualityMeasurementUpdate,
    QualityPredictionCreate,
)


class QualityService:
    """Service for quality measurement and prediction operations."""

    def __init__(self, db: AsyncSession):
        self.db = db

    # Quality Measurements

    async def get_measurements(
        self,
        blend_id: str | None = None,
        tank_tag: str | None = None,
        skip: int = 0,
        limit: int = 100,
    ) -> list[QualityMeasurement]:
        """Get quality measurements with optional filtering."""
        query = select(QualityMeasurement)

        if blend_id:
            query = query.where(QualityMeasurement.blend_id == blend_id)
        if tank_tag:
            query = query.where(QualityMeasurement.tank_tag == tank_tag)

        query = query.order_by(QualityMeasurement.measurement_time.desc())
        query = query.offset(skip).limit(limit)
        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def get_measurement_by_id(self, measurement_id: str) -> QualityMeasurement | None:
        """Get a quality measurement by ID."""
        query = select(QualityMeasurement).where(QualityMeasurement.id == measurement_id)
        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    async def create_measurement(
        self, measurement_data: QualityMeasurementCreate
    ) -> QualityMeasurement:
        """Create a new quality measurement."""
        measurement = QualityMeasurement(
            blend_id=measurement_data.blend_id,
            tank_tag=measurement_data.tank_tag,
            sample_id=measurement_data.sample_id,
            measurement_time=measurement_data.measurement_time,
            source=measurement_data.source,
            analyzer_tag=measurement_data.analyzer_tag,
            viscosity_40c=measurement_data.viscosity_40c,
            viscosity_100c=measurement_data.viscosity_100c,
            viscosity_index=measurement_data.viscosity_index,
            flash_point=measurement_data.flash_point,
            pour_point=measurement_data.pour_point,
            density_15c=measurement_data.density_15c,
            tbn=measurement_data.tbn,
            tan=measurement_data.tan,
            water_content_ppm=measurement_data.water_content_ppm,
            sulfur_content_ppm=measurement_data.sulfur_content_ppm,
            color=measurement_data.color,
            appearance=measurement_data.appearance,
        )

        self.db.add(measurement)
        await self.db.commit()
        await self.db.refresh(measurement)
        return measurement

    async def update_measurement(
        self, measurement_id: str, update_data: QualityMeasurementUpdate
    ) -> QualityMeasurement | None:
        """Update a quality measurement."""
        measurement = await self.get_measurement_by_id(measurement_id)
        if not measurement:
            return None

        data = update_data.model_dump(exclude_unset=True)
        for field, value in data.items():
            setattr(measurement, field, value)

        if update_data.certified and not measurement.certification_date:
            measurement.certification_date = datetime.now(timezone.utc).isoformat()

        measurement.updated_at = datetime.now(timezone.utc)
        await self.db.commit()
        await self.db.refresh(measurement)
        return measurement

    async def evaluate_quality_status(
        self,
        measurement: QualityMeasurement,
        target_viscosity_40c: float | None,
        target_flash_point: float | None,
        target_pour_point: float | None,
        viscosity_tolerance: float = 2.0,
        flash_point_tolerance: float = 5.0,
        pour_point_tolerance: float = 3.0,
    ) -> QualityStatus:
        """Evaluate quality status based on targets and tolerances."""
        deviations = []

        # Check viscosity
        if measurement.viscosity_40c and target_viscosity_40c:
            deviation = abs(
                (measurement.viscosity_40c - target_viscosity_40c) / target_viscosity_40c
            ) * 100
            if deviation > viscosity_tolerance * 2:
                return QualityStatus.OFF_SPEC
            elif deviation > viscosity_tolerance:
                deviations.append("viscosity")

        # Check flash point
        if measurement.flash_point and target_flash_point:
            deviation = abs(measurement.flash_point - target_flash_point)
            if deviation > flash_point_tolerance * 2:
                return QualityStatus.OFF_SPEC
            elif deviation > flash_point_tolerance:
                deviations.append("flash_point")

        # Check pour point
        if measurement.pour_point and target_pour_point:
            deviation = abs(measurement.pour_point - target_pour_point)
            if deviation > pour_point_tolerance * 2:
                return QualityStatus.OFF_SPEC
            elif deviation > pour_point_tolerance:
                deviations.append("pour_point")

        if deviations:
            return QualityStatus.MARGINAL
        return QualityStatus.ON_SPEC

    async def get_latest_blend_measurement(
        self, blend_id: str, source: MeasurementSource | None = None
    ) -> QualityMeasurement | None:
        """Get the latest measurement for a blend."""
        query = (
            select(QualityMeasurement)
            .where(QualityMeasurement.blend_id == blend_id)
            .order_by(QualityMeasurement.measurement_time.desc())
        )

        if source:
            query = query.where(QualityMeasurement.source == source)

        query = query.limit(1)
        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    # Quality Predictions

    async def get_predictions(
        self, blend_id: str, skip: int = 0, limit: int = 100
    ) -> list[QualityPrediction]:
        """Get quality predictions for a blend."""
        query = (
            select(QualityPrediction)
            .where(QualityPrediction.blend_id == blend_id)
            .order_by(QualityPrediction.prediction_time.desc())
            .offset(skip)
            .limit(limit)
        )
        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def get_prediction_by_id(self, prediction_id: str) -> QualityPrediction | None:
        """Get a quality prediction by ID."""
        query = select(QualityPrediction).where(QualityPrediction.id == prediction_id)
        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    async def create_prediction(
        self, prediction_data: QualityPredictionCreate
    ) -> QualityPrediction:
        """Create a new quality prediction."""
        prediction = QualityPrediction(
            blend_id=prediction_data.blend_id,
            prediction_time=prediction_data.prediction_time,
            model_version=prediction_data.model_version,
            model_name=prediction_data.model_name,
            predicted_viscosity_40c=prediction_data.predicted_viscosity_40c,
            predicted_viscosity_100c=prediction_data.predicted_viscosity_100c,
            predicted_flash_point=prediction_data.predicted_flash_point,
            predicted_pour_point=prediction_data.predicted_pour_point,
            predicted_density=prediction_data.predicted_density,
            predicted_tbn=prediction_data.predicted_tbn,
            off_spec_risk_percent=prediction_data.off_spec_risk_percent,
            overall_confidence=prediction_data.overall_confidence,
            risk_factors_json=prediction_data.risk_factors_json,
            recommendations_json=prediction_data.recommendations_json,
        )

        self.db.add(prediction)
        await self.db.commit()
        await self.db.refresh(prediction)
        return prediction

    async def verify_prediction(
        self,
        prediction_id: str,
        measurement_id: str,
        accuracy: float,
    ) -> QualityPrediction | None:
        """Verify a prediction against an actual measurement."""
        prediction = await self.get_prediction_by_id(prediction_id)
        if not prediction:
            return None

        prediction.verified = True
        prediction.verification_measurement_id = measurement_id
        prediction.prediction_accuracy = accuracy
        prediction.updated_at = datetime.now(timezone.utc)

        await self.db.commit()
        await self.db.refresh(prediction)
        return prediction

    async def get_latest_prediction(self, blend_id: str) -> QualityPrediction | None:
        """Get the latest prediction for a blend."""
        query = (
            select(QualityPrediction)
            .where(QualityPrediction.blend_id == blend_id)
            .order_by(QualityPrediction.prediction_time.desc())
            .limit(1)
        )
        result = await self.db.execute(query)
        return result.scalar_one_or_none()
