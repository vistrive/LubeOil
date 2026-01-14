"""
Energy consumption tracking and optimization service.

Implements:
- Real-time energy monitoring per batch
- Energy KPI calculation (kWh per 1000L)
- Peak load avoidance
- Energy cost optimization
- Carbon footprint tracking
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession

from lobp.models.blend import Blend, BlendStatus


@dataclass
class EnergyReading:
    """Real-time energy reading."""

    timestamp: datetime
    power_kw: float
    energy_kwh: float  # Cumulative
    equipment_tag: str
    blend_id: str | None = None


@dataclass
class EnergyMetrics:
    """Energy performance metrics."""

    period_start: datetime
    period_end: datetime
    total_kwh: float
    total_volume_liters: float
    kwh_per_1000l: float
    peak_power_kw: float
    average_power_kw: float
    cost_estimate: float
    carbon_kg: float


class EnergyService:
    """Service for energy monitoring and optimization."""

    def __init__(self, db: AsyncSession):
        self.db = db

        # Configuration
        self.electricity_rate = 0.12  # $/kWh
        self.peak_rate_multiplier = 1.5  # Peak hours cost 50% more
        self.carbon_factor = 0.4  # kg CO2 per kWh (varies by grid)

        # Peak hours (typically 2-7 PM)
        self.peak_hours = list(range(14, 19))

        # Baseline consumption by equipment type (kW)
        self.equipment_baseline = {
            "mixer": 15.0,
            "pump": 5.0,
            "heater": 25.0,
            "agitator": 10.0,
        }

    async def record_energy_reading(
        self,
        blend_id: str,
        power_kw: float,
        cumulative_kwh: float,
        equipment_tag: str,
    ) -> dict[str, Any]:
        """
        Record energy reading for a blend.

        Called periodically during blending operations.
        """
        # Update blend energy consumption
        query = select(Blend).where(Blend.id == blend_id)
        result = await self.db.execute(query)
        blend = result.scalar_one_or_none()

        if blend:
            blend.energy_consumed_kwh = cumulative_kwh
            await self.db.commit()

        return {
            "blend_id": blend_id,
            "power_kw": power_kw,
            "cumulative_kwh": cumulative_kwh,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    async def get_blend_energy_metrics(
        self,
        blend_id: str,
    ) -> dict[str, Any]:
        """Get energy metrics for a specific blend."""
        query = select(Blend).where(Blend.id == blend_id)
        result = await self.db.execute(query)
        blend = result.scalar_one_or_none()

        if not blend:
            raise ValueError(f"Blend {blend_id} not found")

        volume = blend.actual_volume_liters or blend.target_volume_liters
        energy = blend.energy_consumed_kwh or 0

        # Calculate KPIs
        kwh_per_1000l = (energy / (volume / 1000)) if volume > 0 else 0

        # Estimate cost
        cost = energy * self.electricity_rate

        # Carbon footprint
        carbon = energy * self.carbon_factor

        return {
            "blend_id": blend_id,
            "batch_number": blend.batch_number,
            "energy_consumed_kwh": energy,
            "volume_liters": volume,
            "kwh_per_1000l": kwh_per_1000l,
            "cost_estimate": cost,
            "carbon_kg": carbon,
            "efficiency_rating": self._rate_efficiency(kwh_per_1000l),
        }

    def _rate_efficiency(self, kwh_per_1000l: float) -> str:
        """Rate energy efficiency."""
        if kwh_per_1000l <= 3.0:
            return "excellent"
        elif kwh_per_1000l <= 4.0:
            return "good"
        elif kwh_per_1000l <= 5.0:
            return "average"
        elif kwh_per_1000l <= 6.0:
            return "below_average"
        else:
            return "poor"

    async def get_daily_energy_summary(
        self,
        date: datetime,
    ) -> EnergyMetrics:
        """Get energy summary for a specific day."""
        start_of_day = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_day = date.replace(hour=23, minute=59, second=59, microsecond=999999)

        # Get all blends for the day
        query = (
            select(Blend)
            .where(Blend.actual_start >= start_of_day)
            .where(Blend.actual_start <= end_of_day)
        )
        result = await self.db.execute(query)
        blends = list(result.scalars().all())

        total_kwh = sum(b.energy_consumed_kwh or 0 for b in blends)
        total_volume = sum(b.actual_volume_liters or 0 for b in blends)

        kwh_per_1000l = (total_kwh / (total_volume / 1000)) if total_volume > 0 else 0

        # Estimate peak power (simplified)
        peak_power = total_kwh / 8 * 1.5 if total_kwh > 0 else 0  # Assuming 8 hour day
        avg_power = total_kwh / 8 if total_kwh > 0 else 0

        cost = total_kwh * self.electricity_rate
        carbon = total_kwh * self.carbon_factor

        return EnergyMetrics(
            period_start=start_of_day,
            period_end=end_of_day,
            total_kwh=total_kwh,
            total_volume_liters=total_volume,
            kwh_per_1000l=kwh_per_1000l,
            peak_power_kw=peak_power,
            average_power_kw=avg_power,
            cost_estimate=cost,
            carbon_kg=carbon,
        )

    async def get_energy_trend(
        self,
        days: int = 30,
    ) -> list[dict[str, Any]]:
        """Get energy consumption trend over time."""
        trend = []
        now = datetime.now(timezone.utc)

        for i in range(days):
            date = now - timedelta(days=i)
            metrics = await self.get_daily_energy_summary(date)
            trend.append({
                "date": date.date().isoformat(),
                "total_kwh": metrics.total_kwh,
                "total_volume": metrics.total_volume_liters,
                "kwh_per_1000l": metrics.kwh_per_1000l,
                "cost": metrics.cost_estimate,
                "carbon_kg": metrics.carbon_kg,
            })

        trend.reverse()  # Oldest first
        return trend

    async def suggest_peak_avoidance(
        self,
        planned_blends: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Suggest schedule adjustments to avoid peak electricity hours.

        Returns modified schedule with cost savings estimate.
        """
        suggestions = []
        now = datetime.now(timezone.utc)

        for blend in planned_blends:
            start_time = blend.get("scheduled_start")
            if isinstance(start_time, str):
                start_time = datetime.fromisoformat(start_time)

            if start_time and start_time.hour in self.peak_hours:
                # Suggest moving to off-peak
                if start_time.hour < 14:
                    new_start = start_time  # Already off-peak
                else:
                    # Move to next morning
                    new_start = (start_time + timedelta(days=1)).replace(hour=6)

                estimated_kwh = blend.get("estimated_kwh", 100)
                savings = estimated_kwh * self.electricity_rate * (self.peak_rate_multiplier - 1)

                suggestions.append({
                    "blend_id": blend.get("blend_id"),
                    "current_start": start_time.isoformat() if start_time else None,
                    "suggested_start": new_start.isoformat(),
                    "reason": "Avoid peak electricity hours (2-7 PM)",
                    "estimated_savings": savings,
                })

        return suggestions

    async def get_energy_benchmark(
        self,
        recipe_id: str,
    ) -> dict[str, Any]:
        """
        Get energy benchmarks for a recipe.

        Based on historical performance.
        """
        query = (
            select(Blend)
            .where(Blend.recipe_id == recipe_id)
            .where(Blend.status == BlendStatus.COMPLETED)
            .where(Blend.energy_consumed_kwh.isnot(None))
            .order_by(Blend.created_at.desc())
            .limit(50)
        )
        result = await self.db.execute(query)
        blends = list(result.scalars().all())

        if not blends:
            return {
                "recipe_id": recipe_id,
                "samples": 0,
                "message": "No historical data available",
            }

        # Calculate statistics
        efficiencies = []
        for b in blends:
            if b.actual_volume_liters and b.actual_volume_liters > 0:
                eff = b.energy_consumed_kwh / (b.actual_volume_liters / 1000)
                efficiencies.append(eff)

        if not efficiencies:
            return {"recipe_id": recipe_id, "samples": 0}

        return {
            "recipe_id": recipe_id,
            "samples": len(efficiencies),
            "average_kwh_per_1000l": np.mean(efficiencies),
            "best_kwh_per_1000l": np.min(efficiencies),
            "worst_kwh_per_1000l": np.max(efficiencies),
            "std_dev": np.std(efficiencies),
            "target_kwh_per_1000l": np.percentile(efficiencies, 25),  # Top quartile as target
        }

    async def estimate_carbon_footprint(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> dict[str, Any]:
        """
        Calculate carbon footprint for a period.

        Includes Scope 1 (direct) and Scope 2 (electricity).
        """
        # Get total energy consumption
        query = (
            select(func.sum(Blend.energy_consumed_kwh))
            .where(Blend.actual_start >= start_date)
            .where(Blend.actual_start <= end_date)
        )
        result = await self.db.execute(query)
        total_kwh = result.scalar() or 0

        # Scope 2: Electricity
        scope2_kg = total_kwh * self.carbon_factor

        # Scope 1: Estimate from heating (natural gas)
        # Simplified estimate: 10% of electricity equivalent
        scope1_kg = scope2_kg * 0.1

        total_kg = scope1_kg + scope2_kg

        # Get production volume for intensity calculation
        volume_query = (
            select(func.sum(Blend.actual_volume_liters))
            .where(Blend.actual_start >= start_date)
            .where(Blend.actual_start <= end_date)
        )
        vol_result = await self.db.execute(volume_query)
        total_volume = vol_result.scalar() or 0

        carbon_intensity = (total_kg / (total_volume / 1000)) if total_volume > 0 else 0

        return {
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
            },
            "scope1_kg_co2": scope1_kg,
            "scope2_kg_co2": scope2_kg,
            "total_kg_co2": total_kg,
            "total_tonnes_co2": total_kg / 1000,
            "production_volume_liters": total_volume,
            "carbon_intensity_kg_per_1000l": carbon_intensity,
            "electricity_kwh": total_kwh,
        }
