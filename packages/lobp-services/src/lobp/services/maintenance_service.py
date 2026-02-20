"""
Predictive maintenance service for equipment health monitoring.

Implements:
- Vibration analysis for rotating equipment
- Runtime-based maintenance scheduling
- Anomaly detection from sensor data
- Maintenance work order generation
- Equipment health scoring
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

import numpy as np
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from lobp.models.equipment import Pump, PumpStatus, Valve


class HealthStatus(str, Enum):
    """Equipment health status levels."""

    EXCELLENT = "excellent"  # 90-100%
    GOOD = "good"  # 70-89%
    FAIR = "fair"  # 50-69%
    POOR = "poor"  # 30-49%
    CRITICAL = "critical"  # 0-29%


class MaintenanceType(str, Enum):
    """Types of maintenance actions."""

    PREVENTIVE = "preventive"
    PREDICTIVE = "predictive"
    CORRECTIVE = "corrective"
    EMERGENCY = "emergency"


@dataclass
class EquipmentHealth:
    """Equipment health assessment."""

    equipment_id: str
    equipment_tag: str
    equipment_type: str
    health_score: float
    health_status: HealthStatus
    risk_factors: list[str]
    predicted_failure_days: int | None
    recommended_actions: list[str]
    last_assessment: datetime


@dataclass
class MaintenanceTask:
    """A maintenance task or work order."""

    task_id: str
    equipment_tag: str
    equipment_type: str
    maintenance_type: MaintenanceType
    priority: int  # 1-5, 5 being highest
    description: str
    due_date: datetime
    estimated_duration_hours: float
    parts_needed: list[dict[str, Any]]
    created_at: datetime
    assigned_to: str | None = None
    completed_at: datetime | None = None


class PredictiveMaintenanceService:
    """Service for predictive maintenance and equipment health monitoring."""

    def __init__(self, db: AsyncSession):
        self.db = db

        # Thresholds for various parameters
        self.vibration_thresholds = {
            "excellent": 2.0,  # mm/s
            "good": 4.0,
            "fair": 7.0,
            "poor": 10.0,
            # Above 10 is critical
        }

        self.temperature_thresholds = {
            "excellent": 60,  # °C
            "good": 70,
            "fair": 75,
            "poor": 80,
        }

        # Maintenance intervals (hours)
        self.maintenance_intervals = {
            "pump_inspection": 2000,
            "pump_overhaul": 10000,
            "seal_replacement": 5000,
            "bearing_replacement": 8000,
            "valve_inspection": 4000,
        }

    async def assess_pump_health(
        self,
        pump_id: str,
    ) -> EquipmentHealth:
        """
        Assess health of a pump based on sensor data.

        Considers:
        - Vibration levels
        - Bearing temperature
        - Runtime hours
        - Performance degradation
        """
        query = select(Pump).where(Pump.id == pump_id)
        result = await self.db.execute(query)
        pump = result.scalar_one_or_none()

        if not pump:
            raise ValueError(f"Pump {pump_id} not found")

        risk_factors = []
        health_scores = []

        # Vibration analysis
        if pump.current_vibration_mm_s is not None:
            vib = pump.current_vibration_mm_s
            if vib <= self.vibration_thresholds["excellent"]:
                vib_score = 100
            elif vib <= self.vibration_thresholds["good"]:
                vib_score = 80
            elif vib <= self.vibration_thresholds["fair"]:
                vib_score = 60
                risk_factors.append(f"Elevated vibration: {vib:.1f} mm/s")
            elif vib <= self.vibration_thresholds["poor"]:
                vib_score = 40
                risk_factors.append(f"High vibration: {vib:.1f} mm/s")
            else:
                vib_score = 20
                risk_factors.append(f"Critical vibration: {vib:.1f} mm/s")
            health_scores.append(vib_score)

        # Temperature analysis
        if pump.bearing_temperature_celsius is not None:
            temp = pump.bearing_temperature_celsius
            if temp <= self.temperature_thresholds["excellent"]:
                temp_score = 100
            elif temp <= self.temperature_thresholds["good"]:
                temp_score = 80
            elif temp <= self.temperature_thresholds["fair"]:
                temp_score = 60
                risk_factors.append(f"Elevated temperature: {temp:.1f}°C")
            elif temp <= self.temperature_thresholds["poor"]:
                temp_score = 40
                risk_factors.append(f"High bearing temperature: {temp:.1f}°C")
            else:
                temp_score = 20
                risk_factors.append(f"Critical temperature: {temp:.1f}°C")
            health_scores.append(temp_score)

        # Runtime-based wear
        hours_since_maintenance = pump.total_run_hours - pump.last_maintenance_hours
        maintenance_due_ratio = hours_since_maintenance / pump.maintenance_interval_hours

        if maintenance_due_ratio < 0.5:
            runtime_score = 100
        elif maintenance_due_ratio < 0.75:
            runtime_score = 80
        elif maintenance_due_ratio < 1.0:
            runtime_score = 60
            risk_factors.append(f"Maintenance due in {int((1 - maintenance_due_ratio) * pump.maintenance_interval_hours)} hours")
        elif maintenance_due_ratio < 1.25:
            runtime_score = 40
            risk_factors.append("Maintenance overdue")
        else:
            runtime_score = 20
            risk_factors.append(f"Maintenance severely overdue ({int(hours_since_maintenance)} hours)")
        health_scores.append(runtime_score)

        # Calculate overall health score
        overall_score = np.mean(health_scores) if health_scores else 50

        # Determine health status
        if overall_score >= 90:
            status = HealthStatus.EXCELLENT
        elif overall_score >= 70:
            status = HealthStatus.GOOD
        elif overall_score >= 50:
            status = HealthStatus.FAIR
        elif overall_score >= 30:
            status = HealthStatus.POOR
        else:
            status = HealthStatus.CRITICAL

        # Predict failure (simple model)
        predicted_failure = None
        if overall_score < 50:
            # Rough estimate: score of 0 = 0 days, score of 50 = 30 days
            predicted_failure = int(overall_score * 0.6)

        # Generate recommendations
        recommendations = []
        if pump.needs_maintenance:
            recommendations.append("Schedule preventive maintenance")
        if pump.vibration_alarm:
            recommendations.append("Investigate vibration source - check alignment and bearings")
        if pump.bearing_temperature_celsius and pump.bearing_temperature_celsius > 75:
            recommendations.append("Check lubrication and cooling")

        return EquipmentHealth(
            equipment_id=pump.id,
            equipment_tag=pump.tag,
            equipment_type="pump",
            health_score=overall_score,
            health_status=status,
            risk_factors=risk_factors,
            predicted_failure_days=predicted_failure,
            recommended_actions=recommendations,
            last_assessment=datetime.now(timezone.utc),
        )

    async def get_all_equipment_health(self) -> list[EquipmentHealth]:
        """Get health status for all monitored equipment."""
        query = select(Pump).where(Pump.status != PumpStatus.MAINTENANCE)
        result = await self.db.execute(query)
        pumps = list(result.scalars().all())

        health_reports = []
        for pump in pumps:
            try:
                health = await self.assess_pump_health(pump.id)
                health_reports.append(health)
            except Exception:
                pass

        # Sort by health score (worst first)
        health_reports.sort(key=lambda h: h.health_score)

        return health_reports

    async def get_maintenance_schedule(
        self,
        days_ahead: int = 30,
    ) -> list[MaintenanceTask]:
        """
        Generate maintenance schedule based on runtime and condition.

        Returns list of maintenance tasks ordered by due date.
        """
        tasks = []
        now = datetime.now(timezone.utc)
        end_date = now + timedelta(days=days_ahead)

        # Get all pumps
        query = select(Pump)
        result = await self.db.execute(query)
        pumps = list(result.scalars().all())

        for pump in pumps:
            # Check if maintenance is due or upcoming
            hours_since = pump.total_run_hours - pump.last_maintenance_hours
            hours_remaining = pump.maintenance_interval_hours - hours_since

            # Estimate when maintenance will be due (assume 8 hours/day operation)
            if hours_remaining > 0:
                days_until_due = hours_remaining / 8
                due_date = now + timedelta(days=days_until_due)
            else:
                due_date = now  # Overdue

            if due_date <= end_date:
                # Determine priority
                if hours_remaining < 0:
                    priority = 5  # Overdue
                    maintenance_type = MaintenanceType.CORRECTIVE
                elif hours_remaining < 200:
                    priority = 4
                    maintenance_type = MaintenanceType.PREVENTIVE
                else:
                    priority = 3
                    maintenance_type = MaintenanceType.PREVENTIVE

                tasks.append(MaintenanceTask(
                    task_id=str(uuid4())[:8],
                    equipment_tag=pump.tag,
                    equipment_type="pump",
                    maintenance_type=maintenance_type,
                    priority=priority,
                    description=f"Scheduled maintenance for {pump.name}",
                    due_date=due_date,
                    estimated_duration_hours=4.0,
                    parts_needed=[
                        {"part": "seal_kit", "quantity": 1},
                        {"part": "lubricant", "quantity": 2, "unit": "liters"},
                    ],
                    created_at=now,
                ))

            # Check for condition-based maintenance
            health = await self.assess_pump_health(pump.id)
            if health.health_status in [HealthStatus.POOR, HealthStatus.CRITICAL]:
                tasks.append(MaintenanceTask(
                    task_id=str(uuid4())[:8],
                    equipment_tag=pump.tag,
                    equipment_type="pump",
                    maintenance_type=MaintenanceType.PREDICTIVE,
                    priority=5 if health.health_status == HealthStatus.CRITICAL else 4,
                    description=f"Condition-based inspection: {', '.join(health.risk_factors[:2])}",
                    due_date=now + timedelta(days=1),
                    estimated_duration_hours=2.0,
                    parts_needed=[],
                    created_at=now,
                ))

        # Sort by priority (desc) then due date
        tasks.sort(key=lambda t: (-t.priority, t.due_date))

        return tasks

    async def detect_anomalies(
        self,
        equipment_tag: str,
        measurements: list[dict[str, float]],
    ) -> list[dict[str, Any]]:
        """
        Detect anomalies in equipment sensor data.

        Uses statistical analysis to identify unusual patterns.
        """
        if len(measurements) < 10:
            return []

        anomalies = []

        for param in ["vibration", "temperature", "current"]:
            values = [m.get(param) for m in measurements if m.get(param) is not None]
            if len(values) < 5:
                continue

            mean = np.mean(values)
            std = np.std(values)

            for i, v in enumerate(values):
                z_score = (v - mean) / std if std > 0 else 0
                if abs(z_score) > 3:  # 3 sigma rule
                    anomalies.append({
                        "timestamp": measurements[i].get("timestamp"),
                        "parameter": param,
                        "value": v,
                        "z_score": z_score,
                        "severity": "high" if abs(z_score) > 4 else "medium",
                    })

        return anomalies

    async def calculate_mtbf(
        self,
        equipment_tag: str,
        history_days: int = 365,
    ) -> dict[str, Any]:
        """
        Calculate Mean Time Between Failures for equipment.

        Uses historical maintenance records.
        """
        # In a full implementation, this would query failure history
        # For now, return estimated values based on equipment type

        query = select(Pump).where(Pump.tag == equipment_tag)
        result = await self.db.execute(query)
        pump = result.scalar_one_or_none()

        if pump:
            # Estimate based on runtime and maintenance interval
            mtbf_hours = pump.maintenance_interval_hours * 0.8
            mttr_hours = 4  # Mean time to repair

            return {
                "equipment_tag": equipment_tag,
                "mtbf_hours": mtbf_hours,
                "mttr_hours": mttr_hours,
                "availability": mtbf_hours / (mtbf_hours + mttr_hours) * 100,
                "based_on": "estimated",
            }

        return {
            "equipment_tag": equipment_tag,
            "error": "Equipment not found",
        }

    async def get_maintenance_kpis(self) -> dict[str, Any]:
        """Get maintenance performance KPIs."""
        health_reports = await self.get_all_equipment_health()

        total = len(health_reports)
        if total == 0:
            return {"total_equipment": 0}

        excellent = sum(1 for h in health_reports if h.health_status == HealthStatus.EXCELLENT)
        good = sum(1 for h in health_reports if h.health_status == HealthStatus.GOOD)
        fair = sum(1 for h in health_reports if h.health_status == HealthStatus.FAIR)
        poor = sum(1 for h in health_reports if h.health_status == HealthStatus.POOR)
        critical = sum(1 for h in health_reports if h.health_status == HealthStatus.CRITICAL)

        avg_score = np.mean([h.health_score for h in health_reports])

        return {
            "total_equipment": total,
            "average_health_score": avg_score,
            "status_distribution": {
                "excellent": excellent,
                "good": good,
                "fair": fair,
                "poor": poor,
                "critical": critical,
            },
            "equipment_at_risk": poor + critical,
            "predicted_failures_30d": sum(
                1 for h in health_reports
                if h.predicted_failure_days and h.predicted_failure_days <= 30
            ),
        }
