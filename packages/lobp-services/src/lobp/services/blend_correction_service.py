"""
Blend correction service for salvaging marginal batches.

Implements:
- Deviation analysis and root cause identification
- Correction suggestions (additive adjustments)
- Blend-back calculations (mixing with on-spec material)
- Downgrade recommendations
- Cost-benefit analysis of correction options
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from lobp.models.blend import Blend
from lobp.models.quality import QualityMeasurement
from lobp.models.recipe import Recipe
from lobp.models.inventory import Material


class CorrectionType(str, Enum):
    """Types of blend corrections."""

    ADDITIVE_ADJUSTMENT = "additive_adjustment"
    BLEND_BACK = "blend_back"
    DOWNGRADE = "downgrade"
    REPROCESS = "reprocess"
    SCRAP = "scrap"


@dataclass
class QualityDeviation:
    """A quality parameter deviation."""

    parameter: str
    measured_value: float
    target_value: float
    tolerance: float
    deviation: float
    deviation_percent: float
    severity: str  # minor, moderate, severe


@dataclass
class CorrectionOption:
    """A possible correction action."""

    correction_type: CorrectionType
    description: str
    materials_needed: list[dict[str, Any]]
    estimated_cost: float
    success_probability: float
    additional_volume: float
    time_required_hours: float
    recommended: bool
    notes: str


class BlendCorrectionService:
    """Service for analyzing and correcting off-spec blends."""

    def __init__(self, db: AsyncSession):
        self.db = db

        # Correction factors for common additives
        self.additive_effects = {
            # additive_type: {parameter: effect_per_percent}
            "vi_improver": {
                "viscosity_40c": 5.0,  # cSt per 1% additive
                "viscosity_100c": 1.0,
                "viscosity_index": 15.0,
            },
            "pour_point_depressant": {
                "pour_point": -5.0,  # °C per 0.1% additive
            },
            "detergent": {
                "tbn": 2.0,  # per 1% additive
            },
            "antioxidant": {
                "oxidation_stability": 10.0,
            },
        }

    async def analyze_deviations(
        self,
        blend_id: str,
        measurement_id: str | None = None,
    ) -> list[QualityDeviation]:
        """
        Analyze quality deviations for a blend.

        Compares measured values against recipe specifications.
        """
        # Get blend and recipe
        query = (
            select(Blend)
            .options(
                selectinload(Blend.recipe),
                selectinload(Blend.quality_measurements),
            )
            .where(Blend.id == blend_id)
        )
        result = await self.db.execute(query)
        blend = result.scalar_one_or_none()

        if not blend or not blend.recipe:
            raise ValueError(f"Blend {blend_id} or recipe not found")

        recipe = blend.recipe

        # Get measurement
        if measurement_id:
            measurement = next(
                (m for m in blend.quality_measurements if m.id == measurement_id),
                None
            )
        else:
            # Use latest measurement
            measurements = sorted(
                blend.quality_measurements,
                key=lambda m: m.measurement_time,
                reverse=True
            )
            measurement = measurements[0] if measurements else None

        if not measurement:
            raise ValueError("No quality measurement found")

        deviations = []

        # Check each parameter
        checks = [
            ("viscosity_40c", measurement.viscosity_40c,
             recipe.target_viscosity_40c, recipe.viscosity_tolerance),
            ("viscosity_100c", measurement.viscosity_100c,
             recipe.target_viscosity_100c, recipe.viscosity_tolerance),
            ("flash_point", measurement.flash_point,
             recipe.target_flash_point, recipe.flash_point_tolerance),
            ("pour_point", measurement.pour_point,
             recipe.target_pour_point, recipe.pour_point_tolerance),
            ("density", measurement.density_15c,
             recipe.target_density, 1.0),  # 1% tolerance
            ("tbn", measurement.tbn,
             recipe.target_tbn, 5.0),  # 5% tolerance
        ]

        for param, measured, target, tolerance in checks:
            if measured is None or target is None:
                continue

            deviation = measured - target
            if target != 0:
                deviation_percent = abs(deviation / target) * 100
            else:
                deviation_percent = 0

            # Determine severity
            if deviation_percent <= tolerance:
                severity = "on_spec"
            elif deviation_percent <= tolerance * 1.5:
                severity = "minor"
            elif deviation_percent <= tolerance * 2:
                severity = "moderate"
            else:
                severity = "severe"

            if severity != "on_spec":
                deviations.append(QualityDeviation(
                    parameter=param,
                    measured_value=measured,
                    target_value=target,
                    tolerance=tolerance,
                    deviation=deviation,
                    deviation_percent=deviation_percent,
                    severity=severity,
                ))

        return deviations

    async def suggest_corrections(
        self,
        blend_id: str,
        deviations: list[QualityDeviation] | None = None,
    ) -> list[CorrectionOption]:
        """
        Suggest correction options for a deviated blend.

        Evaluates multiple correction strategies and ranks by
        cost-effectiveness.
        """
        if deviations is None:
            deviations = await self.analyze_deviations(blend_id)

        if not deviations:
            return []  # No corrections needed

        # Get blend details
        query = (
            select(Blend)
            .options(selectinload(Blend.recipe))
            .where(Blend.id == blend_id)
        )
        result = await self.db.execute(query)
        blend = result.scalar_one_or_none()

        if not blend:
            raise ValueError(f"Blend {blend_id} not found")

        options = []

        # Analyze each deviation and generate options
        for deviation in deviations:
            # Option 1: Additive adjustment
            additive_option = await self._calculate_additive_correction(
                blend, deviation
            )
            if additive_option:
                options.append(additive_option)

            # Option 2: Blend-back
            blend_back_option = await self._calculate_blend_back(
                blend, deviation
            )
            if blend_back_option:
                options.append(blend_back_option)

        # Option 3: Downgrade (if severe deviations)
        severe_deviations = [d for d in deviations if d.severity == "severe"]
        if severe_deviations:
            downgrade_option = await self._calculate_downgrade(
                blend, deviations
            )
            if downgrade_option:
                options.append(downgrade_option)

        # Rank options by cost-effectiveness
        options.sort(key=lambda o: (
            -o.success_probability,
            o.estimated_cost,
        ))

        # Mark recommended option
        if options:
            options[0].recommended = True

        return options

    async def _calculate_additive_correction(
        self,
        blend: Blend,
        deviation: QualityDeviation,
    ) -> CorrectionOption | None:
        """Calculate additive needed to correct deviation."""
        param = deviation.parameter
        correction_needed = -deviation.deviation  # Opposite of deviation

        # Find suitable additive
        additive_type = None
        effect_per_percent = 0

        for add_type, effects in self.additive_effects.items():
            if param in effects:
                additive_type = add_type
                effect_per_percent = effects[param]
                break

        if not additive_type or effect_per_percent == 0:
            return None

        # Calculate additive quantity needed
        percent_needed = correction_needed / effect_per_percent
        volume_needed = blend.actual_volume_liters * (percent_needed / 100)

        if volume_needed <= 0:
            return None

        # Get additive cost (estimate)
        additive_cost_per_liter = 5.0  # Default estimate
        material_query = select(Material).where(
            Material.category.ilike(f"%{additive_type.replace('_', '%')}%")
        )
        mat_result = await self.db.execute(material_query)
        material = mat_result.scalar_one_or_none()
        if material:
            additive_cost_per_liter = material.standard_cost_per_liter or 5.0

        estimated_cost = volume_needed * additive_cost_per_liter

        return CorrectionOption(
            correction_type=CorrectionType.ADDITIVE_ADJUSTMENT,
            description=f"Add {additive_type.replace('_', ' ')} to correct {param}",
            materials_needed=[{
                "type": additive_type,
                "quantity_liters": volume_needed,
                "percentage": percent_needed,
            }],
            estimated_cost=estimated_cost,
            success_probability=0.85 if deviation.severity == "minor" else 0.70,
            additional_volume=volume_needed,
            time_required_hours=0.5,
            recommended=False,
            notes=f"Add {volume_needed:.1f}L ({percent_needed:.2f}%) to adjust {param} by {correction_needed:.2f}",
        )

    async def _calculate_blend_back(
        self,
        blend: Blend,
        deviation: QualityDeviation,
    ) -> CorrectionOption | None:
        """Calculate blend-back with on-spec material."""
        # Blend-back formula:
        # (current_value * current_vol + target_value * add_vol) / (current_vol + add_vol) = target_value
        # Solving for add_vol:
        # add_vol = current_vol * (current_value - target_value) / (target_value - desired_value)

        # This simplifies to needing equal volume of target material for 50% deviation

        current_vol = blend.actual_volume_liters
        deviation_ratio = abs(deviation.deviation / deviation.target_value) if deviation.target_value else 0

        # Volume of on-spec material needed
        blend_back_volume = current_vol * deviation_ratio * 2

        if blend_back_volume <= 0 or blend_back_volume > current_vol * 2:
            return None  # Not practical

        # Estimate cost based on recipe cost
        cost_per_liter = 2.0  # Default
        if blend.material_cost and blend.actual_volume_liters:
            cost_per_liter = blend.material_cost / blend.actual_volume_liters

        estimated_cost = blend_back_volume * cost_per_liter

        return CorrectionOption(
            correction_type=CorrectionType.BLEND_BACK,
            description=f"Blend back with on-spec {blend.recipe.code if blend.recipe else 'product'}",
            materials_needed=[{
                "type": "on_spec_product",
                "quantity_liters": blend_back_volume,
                "source": "finished_product_tank",
            }],
            estimated_cost=estimated_cost,
            success_probability=0.95,
            additional_volume=blend_back_volume,
            time_required_hours=1.0 + (blend_back_volume / 5000),
            recommended=False,
            notes=f"Blend {blend_back_volume:.0f}L of on-spec product to dilute deviation",
        )

    async def _calculate_downgrade(
        self,
        blend: Blend,
        deviations: list[QualityDeviation],
    ) -> CorrectionOption | None:
        """Calculate downgrade to lower-spec product."""
        # Find possible downgrade targets
        # E.g., SAE 10W-40 with low VI could become SAE 15W-40

        # This is a simplified example
        revenue_loss_percent = 15  # Typical downgrade loss

        original_value = blend.material_cost * 1.3  # Assume 30% margin
        downgrade_value = original_value * (1 - revenue_loss_percent / 100)
        cost = original_value - downgrade_value

        return CorrectionOption(
            correction_type=CorrectionType.DOWNGRADE,
            description="Downgrade to lower specification product",
            materials_needed=[],
            estimated_cost=cost,
            success_probability=1.0,  # Always possible
            additional_volume=0,
            time_required_hours=0.5,
            recommended=False,
            notes=f"Revenue loss ~{revenue_loss_percent}%. Deviations: {', '.join(d.parameter for d in deviations)}",
        )

    async def get_correction_history(
        self,
        recipe_id: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Get history of corrections for learning."""
        # In a full implementation, this would query a corrections table
        return []

    async def apply_correction(
        self,
        blend_id: str,
        correction_type: CorrectionType,
        materials: list[dict[str, Any]],
        performed_by: str,
    ) -> dict[str, Any]:
        """
        Record a correction being applied.

        This would integrate with blend service to update the blend.
        """
        return {
            "blend_id": blend_id,
            "correction_type": correction_type.value,
            "status": "applied",
            "applied_at": datetime.now(timezone.utc).isoformat(),
            "performed_by": performed_by,
            "materials": materials,
        }


class BlendSalvageAnalyzer:
    """
    Analyzer for determining if a blend can be salvaged.

    Considers:
    - Severity of deviations
    - Available correction options
    - Cost vs. scrap value
    - Tank availability for corrections
    """

    def __init__(self, correction_service: BlendCorrectionService):
        self.correction_service = correction_service

    async def analyze_salvageability(
        self,
        blend_id: str,
    ) -> dict[str, Any]:
        """
        Analyze if blend can be economically salvaged.

        Returns recommendation with justification.
        """
        deviations = await self.correction_service.analyze_deviations(blend_id)

        if not deviations:
            return {
                "blend_id": blend_id,
                "salvageable": True,
                "status": "on_spec",
                "recommendation": "No correction needed",
                "deviations": [],
                "options": [],
            }

        options = await self.correction_service.suggest_corrections(
            blend_id, deviations
        )

        # Determine if salvageable
        salvageable = any(o.success_probability >= 0.7 for o in options)

        # Get scrap value estimate
        query = select(Blend).where(Blend.id == blend_id)
        result = await self.correction_service.db.execute(query)
        blend = result.scalar_one_or_none()

        scrap_value = (blend.material_cost or 0) * 0.3  # 30% recovery

        # Find best option
        best_option = None
        for option in options:
            if option.estimated_cost < (blend.material_cost or 0) - scrap_value:
                if best_option is None or option.success_probability > best_option.success_probability:
                    best_option = option

        recommendation = "Scrap"
        if best_option:
            recommendation = best_option.description

        return {
            "blend_id": blend_id,
            "salvageable": salvageable,
            "status": "deviated",
            "deviation_count": len(deviations),
            "severities": [d.severity for d in deviations],
            "recommendation": recommendation,
            "scrap_value": scrap_value,
            "deviations": [
                {
                    "parameter": d.parameter,
                    "deviation": d.deviation,
                    "severity": d.severity,
                }
                for d in deviations
            ],
            "options": [
                {
                    "type": o.correction_type.value,
                    "description": o.description,
                    "cost": o.estimated_cost,
                    "success_probability": o.success_probability,
                    "recommended": o.recommended,
                }
                for o in options
            ],
        }
