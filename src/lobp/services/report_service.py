"""
Report generation service for COA, batch reports, and documentation.

Implements:
- Automatic Certificate of Analysis (COA) generation
- Batch production reports
- Quality deviation reports
- Traceability reports
"""

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from lobp.models.blend import Blend, BlendIngredient
from lobp.models.quality import QualityMeasurement, QualityPrediction
from lobp.models.recipe import Recipe


class ReportService:
    """Service for generating production reports and certificates."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def generate_coa(
        self,
        blend_id: str,
        certified_by: str | None = None,
        include_ai_predictions: bool = False,
    ) -> dict[str, Any]:
        """
        Generate Certificate of Analysis for a completed blend.

        Args:
            blend_id: ID of the blend
            certified_by: Name of certifying authority
            include_ai_predictions: Include AI prediction data

        Returns:
            Complete COA data structure
        """
        # Get blend with all related data
        query = (
            select(Blend)
            .options(
                selectinload(Blend.recipe),
                selectinload(Blend.ingredients),
                selectinload(Blend.quality_measurements),
                selectinload(Blend.quality_predictions),
            )
            .where(Blend.id == blend_id)
        )
        result = await self.db.execute(query)
        blend = result.scalar_one_or_none()

        if not blend:
            raise ValueError(f"Blend {blend_id} not found")

        # Get final quality measurement
        final_measurement = None
        for m in blend.quality_measurements:
            if m.is_final and m.certified:
                final_measurement = m
                break

        # If no certified final, get latest lab measurement
        if not final_measurement:
            lab_measurements = [
                m for m in blend.quality_measurements
                if m.source.value == "lab_analysis"
            ]
            if lab_measurements:
                final_measurement = max(
                    lab_measurements,
                    key=lambda x: x.measurement_time
                )

        # Build COA
        coa = {
            "document_type": "Certificate of Analysis",
            "coa_number": f"COA-{blend.batch_number}",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "certified_by": certified_by,

            # Product Information
            "product": {
                "batch_number": blend.batch_number,
                "product_name": blend.recipe.name if blend.recipe else "Unknown",
                "product_code": blend.recipe.code if blend.recipe else "Unknown",
                "production_date": blend.actual_start,
                "volume_liters": blend.actual_volume_liters,
            },

            # Specifications
            "specifications": {},

            # Test Results
            "test_results": {},

            # Compliance
            "compliance": {
                "meets_specifications": True,
                "deviations": [],
            },

            # Traceability
            "traceability": {
                "recipe_version": blend.recipe.version if blend.recipe else None,
                "blend_tank": blend.blend_tank_tag,
                "ingredients": [],
            },
        }

        # Add specifications from recipe
        if blend.recipe:
            recipe = blend.recipe
            coa["specifications"] = {
                "viscosity_40c": {
                    "target": recipe.target_viscosity_40c,
                    "min": (recipe.target_viscosity_40c or 0) * (1 - recipe.viscosity_tolerance / 100),
                    "max": (recipe.target_viscosity_40c or 0) * (1 + recipe.viscosity_tolerance / 100),
                    "unit": "cSt",
                },
                "viscosity_100c": {
                    "target": recipe.target_viscosity_100c,
                    "unit": "cSt",
                },
                "flash_point": {
                    "target": recipe.target_flash_point,
                    "min": (recipe.target_flash_point or 0) - recipe.flash_point_tolerance,
                    "unit": "°C",
                    "test_method": "ASTM D92",
                },
                "pour_point": {
                    "target": recipe.target_pour_point,
                    "max": (recipe.target_pour_point or 0) + recipe.pour_point_tolerance,
                    "unit": "°C",
                    "test_method": "ASTM D97",
                },
                "density": {
                    "target": recipe.target_density,
                    "unit": "kg/m³",
                    "test_method": "ASTM D4052",
                },
            }

        # Add test results
        if final_measurement:
            coa["test_results"] = {
                "viscosity_40c": {
                    "value": final_measurement.viscosity_40c,
                    "unit": "cSt",
                    "test_method": "ASTM D445",
                    "status": self._check_spec_status(
                        final_measurement.viscosity_40c,
                        coa["specifications"].get("viscosity_40c", {})
                    ),
                },
                "viscosity_100c": {
                    "value": final_measurement.viscosity_100c,
                    "unit": "cSt",
                    "test_method": "ASTM D445",
                },
                "flash_point": {
                    "value": final_measurement.flash_point,
                    "unit": "°C",
                    "test_method": "ASTM D92",
                    "status": self._check_spec_status(
                        final_measurement.flash_point,
                        coa["specifications"].get("flash_point", {})
                    ),
                },
                "pour_point": {
                    "value": final_measurement.pour_point,
                    "unit": "°C",
                    "test_method": "ASTM D97",
                    "status": self._check_spec_status(
                        final_measurement.pour_point,
                        coa["specifications"].get("pour_point", {})
                    ),
                },
                "density": {
                    "value": final_measurement.density_15c,
                    "unit": "kg/m³",
                    "test_method": "ASTM D4052",
                },
                "color": {
                    "value": final_measurement.color,
                    "test_method": "ASTM D1500",
                },
                "appearance": {
                    "value": final_measurement.appearance or "Bright & Clear",
                },
                "water_content": {
                    "value": final_measurement.water_content_ppm,
                    "unit": "ppm",
                    "test_method": "ASTM D6304",
                },
                "measurement_date": final_measurement.measurement_time,
                "sample_id": final_measurement.sample_id,
            }

            # Check overall compliance
            deviations = []
            for param, result in coa["test_results"].items():
                if isinstance(result, dict) and result.get("status") == "fail":
                    deviations.append(param)

            coa["compliance"]["meets_specifications"] = len(deviations) == 0
            coa["compliance"]["deviations"] = deviations

        # Add ingredient traceability
        for ing in blend.ingredients:
            coa["traceability"]["ingredients"].append({
                "material_code": ing.material_code,
                "material_name": ing.material_name,
                "percentage": ing.actual_percentage or ing.target_percentage,
                "volume_liters": ing.actual_volume_liters or ing.target_volume_liters,
                "source_tank": ing.source_tank_tag,
                "source_batch": ing.source_batch_number,
            })

        # Add AI predictions if requested
        if include_ai_predictions and blend.quality_predictions:
            latest_prediction = max(
                blend.quality_predictions,
                key=lambda x: x.prediction_time
            )
            coa["ai_analysis"] = {
                "prediction_confidence": latest_prediction.overall_confidence,
                "off_spec_risk": latest_prediction.off_spec_risk_percent,
                "model_version": latest_prediction.model_version,
                "ai_optimized": blend.ai_optimized,
            }

        return coa

    def _check_spec_status(
        self,
        value: float | None,
        spec: dict[str, Any],
    ) -> str:
        """Check if value meets specification."""
        if value is None:
            return "not_tested"

        min_val = spec.get("min")
        max_val = spec.get("max")

        if min_val is not None and value < min_val:
            return "fail"
        if max_val is not None and value > max_val:
            return "fail"

        return "pass"

    async def generate_batch_report(
        self,
        blend_id: str,
    ) -> dict[str, Any]:
        """
        Generate comprehensive batch production report.

        Includes production timeline, resource usage, quality data,
        and cost breakdown.
        """
        # Get blend with all related data
        query = (
            select(Blend)
            .options(
                selectinload(Blend.recipe).selectinload(Recipe.ingredients),
                selectinload(Blend.ingredients),
                selectinload(Blend.quality_measurements),
                selectinload(Blend.quality_predictions),
            )
            .where(Blend.id == blend_id)
        )
        result = await self.db.execute(query)
        blend = result.scalar_one_or_none()

        if not blend:
            raise ValueError(f"Blend {blend_id} not found")

        # Calculate production metrics
        duration_hours = None
        if blend.actual_start and blend.actual_end:
            start = datetime.fromisoformat(str(blend.actual_start))
            end = datetime.fromisoformat(str(blend.actual_end))
            duration_hours = (end - start).total_seconds() / 3600

        # Calculate yield
        total_input = sum(
            ing.actual_volume_liters or ing.target_volume_liters
            for ing in blend.ingredients
        )
        yield_percent = (
            (blend.actual_volume_liters / total_input * 100)
            if total_input > 0 else 0
        )

        # Build report
        report = {
            "report_type": "Batch Production Report",
            "report_number": f"BPR-{blend.batch_number}",
            "generated_at": datetime.now(timezone.utc).isoformat(),

            # Batch Summary
            "batch_summary": {
                "batch_number": blend.batch_number,
                "recipe_code": blend.recipe.code if blend.recipe else None,
                "recipe_name": blend.recipe.name if blend.recipe else None,
                "status": blend.status.value,
                "priority": blend.priority.value,
            },

            # Production Timeline
            "timeline": {
                "scheduled_start": blend.scheduled_start,
                "scheduled_end": blend.scheduled_end,
                "actual_start": blend.actual_start,
                "actual_end": blend.actual_end,
                "duration_hours": duration_hours,
            },

            # Volume & Yield
            "production": {
                "target_volume": blend.target_volume_liters,
                "actual_volume": blend.actual_volume_liters,
                "variance": blend.actual_volume_liters - blend.target_volume_liters,
                "variance_percent": (
                    (blend.actual_volume_liters - blend.target_volume_liters)
                    / blend.target_volume_liters * 100
                    if blend.target_volume_liters > 0 else 0
                ),
                "yield_percent": yield_percent,
            },

            # Equipment
            "equipment": {
                "blend_tank": blend.blend_tank_tag,
                "destination_tank": blend.destination_tank_tag,
                "mixing_speed_rpm": blend.mixing_speed_rpm,
                "mixing_temperature": blend.mixing_temperature_celsius,
                "mixing_duration_minutes": blend.mixing_duration_minutes,
            },

            # Ingredients
            "ingredients": [],

            # Quality Summary
            "quality": {
                "off_spec_risk": blend.off_spec_risk_percent,
                "quality_approved": blend.quality_approved,
                "lab_sample_id": blend.lab_sample_id,
                "measurements_count": len(blend.quality_measurements),
                "ai_optimized": blend.ai_optimized,
                "ai_confidence": blend.ai_confidence_score,
            },

            # Cost Summary
            "costs": {
                "material_cost": blend.material_cost,
                "energy_consumed_kwh": blend.energy_consumed_kwh,
                "energy_cost_estimate": blend.energy_consumed_kwh * 0.12,  # Assuming $0.12/kWh
                "cost_per_liter": (
                    blend.material_cost / blend.actual_volume_liters
                    if blend.actual_volume_liters > 0 else 0
                ),
            },

            # Personnel
            "personnel": {
                "created_by": blend.created_by,
                "approved_by": blend.approved_by,
            },

            # Notes
            "notes": blend.notes,
            "hold_reason": blend.hold_reason,
        }

        # Add ingredient details
        for ing in blend.ingredients:
            deviation = 0
            if ing.target_volume_liters > 0:
                deviation = (
                    (ing.actual_volume_liters - ing.target_volume_liters)
                    / ing.target_volume_liters * 100
                )

            report["ingredients"].append({
                "material_code": ing.material_code,
                "material_name": ing.material_name,
                "target_volume": ing.target_volume_liters,
                "actual_volume": ing.actual_volume_liters,
                "target_percentage": ing.target_percentage,
                "actual_percentage": ing.actual_percentage,
                "deviation_percent": deviation,
                "source_tank": ing.source_tank_tag,
                "source_batch": ing.source_batch_number,
                "sequence_order": ing.sequence_order,
                "status": ing.status.value,
                "ai_adjusted": ing.ai_adjusted,
                "unit_cost": ing.unit_cost,
                "total_cost": ing.total_cost,
            })

        return report

    async def generate_traceability_report(
        self,
        blend_id: str,
    ) -> dict[str, Any]:
        """
        Generate full traceability report for a blend.

        Shows complete material genealogy from raw materials to finished product.
        """
        query = (
            select(Blend)
            .options(
                selectinload(Blend.recipe),
                selectinload(Blend.ingredients),
            )
            .where(Blend.id == blend_id)
        )
        result = await self.db.execute(query)
        blend = result.scalar_one_or_none()

        if not blend:
            raise ValueError(f"Blend {blend_id} not found")

        report = {
            "report_type": "Traceability Report",
            "batch_number": blend.batch_number,
            "product_code": blend.recipe.code if blend.recipe else None,
            "product_name": blend.recipe.name if blend.recipe else None,
            "generated_at": datetime.now(timezone.utc).isoformat(),

            # Forward traceability (what was made)
            "finished_product": {
                "batch_number": blend.batch_number,
                "volume_liters": blend.actual_volume_liters,
                "production_date": blend.actual_start,
                "destination_tank": blend.destination_tank_tag,
                "quality_status": "approved" if blend.quality_approved else "pending",
            },

            # Backward traceability (what went in)
            "raw_materials": [],

            # Process traceability
            "process": {
                "blend_tank": blend.blend_tank_tag,
                "recipe_version": blend.recipe.version if blend.recipe else None,
                "ai_optimized": blend.ai_optimized,
                "mixing_parameters": {
                    "speed_rpm": blend.mixing_speed_rpm,
                    "temperature_celsius": blend.mixing_temperature_celsius,
                    "duration_minutes": blend.mixing_duration_minutes,
                },
            },
        }

        # Add raw material details
        for ing in blend.ingredients:
            report["raw_materials"].append({
                "material_code": ing.material_code,
                "material_name": ing.material_name,
                "source_tank": ing.source_tank_tag,
                "source_batch": ing.source_batch_number,
                "quantity_liters": ing.actual_volume_liters or ing.target_volume_liters,
                "percentage": ing.actual_percentage or ing.target_percentage,
            })

        return report

    async def generate_quality_deviation_report(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> dict[str, Any]:
        """Generate report of quality deviations in date range."""
        query = (
            select(Blend)
            .options(
                selectinload(Blend.recipe),
                selectinload(Blend.quality_measurements),
            )
            .where(Blend.created_at >= start_date)
            .where(Blend.created_at <= end_date)
        )
        result = await self.db.execute(query)
        blends = list(result.scalars().all())

        deviations = []
        total_batches = len(blends)
        off_spec_count = 0
        marginal_count = 0

        for blend in blends:
            # Check for off-spec measurements
            for m in blend.quality_measurements:
                if m.status.value == "off_spec":
                    off_spec_count += 1
                    deviations.append({
                        "batch_number": blend.batch_number,
                        "recipe": blend.recipe.code if blend.recipe else None,
                        "status": "off_spec",
                        "measurement_time": m.measurement_time,
                        "details": m.deviations,
                    })
                elif m.status.value == "marginal":
                    marginal_count += 1

        return {
            "report_type": "Quality Deviation Report",
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
            },
            "summary": {
                "total_batches": total_batches,
                "on_spec_count": total_batches - off_spec_count - marginal_count,
                "marginal_count": marginal_count,
                "off_spec_count": off_spec_count,
                "on_spec_rate": (
                    (total_batches - off_spec_count) / total_batches * 100
                    if total_batches > 0 else 100
                ),
            },
            "deviations": deviations,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
