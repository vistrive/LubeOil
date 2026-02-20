"""
Digital Twin for Lube Oil Blending Plant simulation.

Implements:
- Virtual plant model
- Real-time state synchronization
- What-if scenario simulation
- Process optimization testing
- Training and validation environment
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any
from uuid import uuid4
import math
import random

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from lobp.models.blend import Blend, BlendStatus
from lobp.models.tank import Tank, TankStatus, TankType
from lobp.models.equipment import Pump, PumpStatus
from lobp.models.recipe import Recipe


class SimulationSpeed(str, Enum):
    """Simulation time multipliers."""

    REAL_TIME = "real_time"  # 1x
    FAST = "fast"  # 10x
    VERY_FAST = "very_fast"  # 60x
    INSTANT = "instant"  # Complete immediately


@dataclass
class TwinTank:
    """Virtual tank state."""

    tag: str
    tank_type: TankType
    capacity_liters: float
    current_level: float
    temperature: float
    material_code: str | None = None
    is_mixing: bool = False
    mixer_speed_rpm: float = 0


@dataclass
class TwinPump:
    """Virtual pump state."""

    tag: str
    status: PumpStatus
    flow_rate_lph: float = 0
    pressure_bar: float = 0
    temperature: float = 25.0
    runtime_hours: float = 0
    power_kw: float = 0


@dataclass
class TwinBlend:
    """Virtual blend operation."""

    id: str
    batch_number: str
    recipe_code: str
    target_volume: float
    current_volume: float = 0
    status: BlendStatus = BlendStatus.QUEUED
    blend_tank: str | None = None
    start_time: datetime | None = None
    ingredients_added: dict[str, float] = field(default_factory=dict)
    quality_predictions: dict[str, float] = field(default_factory=dict)
    energy_consumed: float = 0


@dataclass
class SimulationResult:
    """Result of a simulation run."""

    simulation_id: str
    scenario_name: str
    duration_seconds: float
    blends_completed: int
    total_volume_produced: float
    energy_consumed: float
    quality_issues: int
    equipment_utilization: dict[str, float]
    bottlenecks: list[str]
    recommendations: list[str]


class DigitalTwin:
    """
    Digital Twin for LOBP simulation and optimization.

    Maintains a virtual replica of the plant that can:
    - Sync with real plant state
    - Run simulations at various speeds
    - Test what-if scenarios
    - Validate process changes before implementation
    """

    def __init__(self, db: AsyncSession):
        self.db = db

        # Virtual plant state
        self.tanks: dict[str, TwinTank] = {}
        self.pumps: dict[str, TwinPump] = {}
        self.blends: dict[str, TwinBlend] = {}

        # Simulation parameters
        self.simulation_time: datetime = datetime.now(timezone.utc)
        self.is_running: bool = False
        self.speed: SimulationSpeed = SimulationSpeed.REAL_TIME

        # Physics parameters
        self.ambient_temperature = 25.0
        self.heat_transfer_coeff = 0.1
        self.mixing_efficiency = 0.95
        self.pump_efficiency = 0.85

        # Event log
        self.event_log: list[dict[str, Any]] = []

    async def sync_with_plant(self) -> dict[str, Any]:
        """
        Synchronize digital twin with actual plant state.

        Reads current state from database and updates virtual model.
        """
        # Sync tanks
        tank_query = select(Tank)
        tank_result = await self.db.execute(tank_query)
        tanks = list(tank_result.scalars().all())

        for tank in tanks:
            self.tanks[tank.tag] = TwinTank(
                tag=tank.tag,
                tank_type=tank.tank_type,
                capacity_liters=tank.capacity_liters,
                current_level=tank.current_level_liters,
                temperature=tank.temperature or 25.0,
                material_code=tank.current_material_code,
                is_mixing=tank.status == TankStatus.MIXING,
            )

        # Sync pumps
        pump_query = select(Pump)
        pump_result = await self.db.execute(pump_query)
        pumps = list(pump_result.scalars().all())

        for pump in pumps:
            self.pumps[pump.tag] = TwinPump(
                tag=pump.tag,
                status=pump.status,
                flow_rate_lph=pump.current_flow_rate or 0,
                runtime_hours=pump.runtime_hours or 0,
            )

        # Sync active blends
        blend_query = (
            select(Blend)
            .options(selectinload(Blend.recipe))
            .where(Blend.status.in_([
                BlendStatus.QUEUED,
                BlendStatus.SCHEDULED,
                BlendStatus.IN_PROGRESS,
                BlendStatus.MIXING,
            ]))
        )
        blend_result = await self.db.execute(blend_query)
        blends = list(blend_result.scalars().all())

        for blend in blends:
            self.blends[blend.id] = TwinBlend(
                id=blend.id,
                batch_number=blend.batch_number,
                recipe_code=blend.recipe.code if blend.recipe else "UNKNOWN",
                target_volume=blend.target_volume_liters,
                current_volume=blend.actual_volume_liters or 0,
                status=blend.status,
                blend_tank=blend.blend_tank_tag,
                start_time=datetime.fromisoformat(str(blend.actual_start)) if blend.actual_start else None,
                energy_consumed=blend.energy_consumed_kwh or 0,
            )

        self.simulation_time = datetime.now(timezone.utc)

        return {
            "synced_at": self.simulation_time.isoformat(),
            "tanks": len(self.tanks),
            "pumps": len(self.pumps),
            "active_blends": len(self.blends),
        }

    async def run_simulation(
        self,
        scenario_name: str,
        duration_hours: float,
        speed: SimulationSpeed = SimulationSpeed.FAST,
        modifications: dict[str, Any] | None = None,
    ) -> SimulationResult:
        """
        Run a simulation scenario.

        Args:
            scenario_name: Name for this simulation run
            duration_hours: How long to simulate
            speed: Simulation speed
            modifications: Optional changes to test
        """
        simulation_id = str(uuid4())[:8]
        self.is_running = True
        self.speed = speed
        start_time = self.simulation_time

        # Apply modifications if any
        if modifications:
            self._apply_modifications(modifications)

        # Run simulation loop
        time_step = timedelta(minutes=1)
        end_time = start_time + timedelta(hours=duration_hours)

        blends_completed = 0
        total_volume = 0.0
        total_energy = 0.0
        quality_issues = 0
        equipment_usage: dict[str, float] = {tag: 0 for tag in self.pumps}

        while self.simulation_time < end_time and self.is_running:
            # Update physics
            self._update_temperatures(time_step)
            self._update_transfers(time_step)
            self._update_mixing(time_step)

            # Process blends
            for blend in list(self.blends.values()):
                if blend.status == BlendStatus.QUEUED:
                    # Try to start blend
                    if self._can_start_blend(blend):
                        self._start_blend(blend)

                elif blend.status == BlendStatus.IN_PROGRESS:
                    # Continue adding ingredients
                    progress = self._add_ingredients(blend, time_step)
                    if progress >= 1.0:
                        blend.status = BlendStatus.MIXING
                        self._log_event("blend_filling_complete", blend.batch_number)

                elif blend.status == BlendStatus.MIXING:
                    # Check if mixing complete
                    mix_time = self._calculate_mix_time(blend)
                    if blend.start_time:
                        elapsed = (self.simulation_time - blend.start_time).total_seconds() / 3600
                        if elapsed >= mix_time:
                            # Check quality
                            quality = self._predict_quality(blend)
                            if quality["on_spec"]:
                                blend.status = BlendStatus.COMPLETED
                                blends_completed += 1
                                total_volume += blend.current_volume
                                total_energy += blend.energy_consumed
                                self._log_event("blend_completed", blend.batch_number)
                            else:
                                quality_issues += 1
                                self._log_event("quality_issue", blend.batch_number, quality)

            # Track equipment usage
            for tag, pump in self.pumps.items():
                if pump.status == PumpStatus.RUNNING:
                    equipment_usage[tag] += time_step.total_seconds() / 3600

            # Advance time
            self.simulation_time += time_step

        self.is_running = False

        # Calculate results
        total_hours = duration_hours
        utilization = {
            tag: (hours / total_hours * 100) if total_hours > 0 else 0
            for tag, hours in equipment_usage.items()
        }

        bottlenecks = self._identify_bottlenecks(utilization)
        recommendations = self._generate_recommendations(
            blends_completed, quality_issues, bottlenecks
        )

        return SimulationResult(
            simulation_id=simulation_id,
            scenario_name=scenario_name,
            duration_seconds=duration_hours * 3600,
            blends_completed=blends_completed,
            total_volume_produced=total_volume,
            energy_consumed=total_energy,
            quality_issues=quality_issues,
            equipment_utilization=utilization,
            bottlenecks=bottlenecks,
            recommendations=recommendations,
        )

    def _apply_modifications(self, mods: dict[str, Any]) -> None:
        """Apply scenario modifications."""
        if "tank_capacity" in mods:
            for tag, capacity in mods["tank_capacity"].items():
                if tag in self.tanks:
                    self.tanks[tag].capacity_liters = capacity

        if "pump_flow_rate" in mods:
            for tag, rate in mods["pump_flow_rate"].items():
                if tag in self.pumps:
                    self.pumps[tag].flow_rate_lph = rate

        if "add_blends" in mods:
            for blend_data in mods["add_blends"]:
                blend = TwinBlend(
                    id=str(uuid4()),
                    batch_number=blend_data.get("batch", f"SIM-{uuid4().hex[:6]}"),
                    recipe_code=blend_data.get("recipe", "TEST"),
                    target_volume=blend_data.get("volume", 5000),
                )
                self.blends[blend.id] = blend

        self._log_event("modifications_applied", str(mods))

    def _update_temperatures(self, dt: timedelta) -> None:
        """Update tank temperatures based on heat transfer."""
        hours = dt.total_seconds() / 3600

        for tank in self.tanks.values():
            # Simple heat transfer model
            temp_diff = self.ambient_temperature - tank.temperature
            delta_temp = self.heat_transfer_coeff * temp_diff * hours

            # Mixing adds energy
            if tank.is_mixing:
                delta_temp += 0.5 * hours  # Slight heating from mixing

            tank.temperature += delta_temp

    def _update_transfers(self, dt: timedelta) -> None:
        """Update material transfers between tanks."""
        for pump in self.pumps.values():
            if pump.status == PumpStatus.RUNNING and pump.flow_rate_lph > 0:
                volume = pump.flow_rate_lph * (dt.total_seconds() / 3600)

                # Calculate power consumption
                power = pump.flow_rate_lph * pump.pressure_bar * 0.0001 / self.pump_efficiency
                pump.power_kw = power
                pump.runtime_hours += dt.total_seconds() / 3600

    def _update_mixing(self, dt: timedelta) -> None:
        """Update mixing operations."""
        for tank in self.tanks.values():
            if tank.is_mixing and tank.mixer_speed_rpm > 0:
                # Mixing energy consumption
                power = 0.001 * tank.mixer_speed_rpm * (tank.current_level / 1000)

    def _can_start_blend(self, blend: TwinBlend) -> bool:
        """Check if a blend can be started."""
        # Find available blend tank
        for tank in self.tanks.values():
            if tank.tank_type == TankType.BLEND:
                if tank.current_level < tank.capacity_liters * 0.1:
                    return True
        return False

    def _start_blend(self, blend: TwinBlend) -> None:
        """Start a blend operation."""
        # Find blend tank
        for tag, tank in self.tanks.items():
            if tank.tank_type == TankType.BLEND and tank.current_level < tank.capacity_liters * 0.1:
                blend.blend_tank = tag
                blend.status = BlendStatus.IN_PROGRESS
                blend.start_time = self.simulation_time
                tank.is_mixing = True
                self._log_event("blend_started", blend.batch_number, {"tank": tag})
                return

    def _add_ingredients(self, blend: TwinBlend, dt: timedelta) -> float:
        """Add ingredients to a blend. Returns progress 0-1."""
        # Simulate ingredient addition at typical rate
        rate = 2000  # liters per hour
        volume = rate * (dt.total_seconds() / 3600)

        blend.current_volume = min(
            blend.current_volume + volume,
            blend.target_volume
        )

        # Energy for pumping
        blend.energy_consumed += volume * 0.002  # kWh per liter

        if blend.blend_tank and blend.blend_tank in self.tanks:
            self.tanks[blend.blend_tank].current_level = blend.current_volume

        return blend.current_volume / blend.target_volume

    def _calculate_mix_time(self, blend: TwinBlend) -> float:
        """Calculate required mixing time in hours."""
        # Larger batches need more mixing
        base_time = 0.5
        volume_factor = blend.target_volume / 5000
        return base_time + (0.25 * volume_factor)

    def _predict_quality(self, blend: TwinBlend) -> dict[str, Any]:
        """Predict quality of finished blend."""
        # Simulate quality prediction
        # In real implementation, would use actual soft sensor models

        base_quality = 0.95
        noise = random.gauss(0, 0.02)
        predicted_quality = base_quality + noise

        on_spec = predicted_quality >= 0.93

        return {
            "on_spec": on_spec,
            "predicted_viscosity": 95.5 + random.gauss(0, 1.0),
            "confidence": 0.92,
        }

    def _identify_bottlenecks(
        self,
        utilization: dict[str, float],
    ) -> list[str]:
        """Identify process bottlenecks."""
        bottlenecks = []

        # High utilization indicates bottleneck
        for tag, util in utilization.items():
            if util > 85:
                bottlenecks.append(f"Pump {tag} at {util:.1f}% utilization")

        # Check for tank capacity constraints
        for tag, tank in self.tanks.items():
            if tank.tank_type == TankType.BLEND:
                if tank.current_level > tank.capacity_liters * 0.9:
                    bottlenecks.append(f"Blend tank {tag} near capacity")

        return bottlenecks

    def _generate_recommendations(
        self,
        blends_completed: int,
        quality_issues: int,
        bottlenecks: list[str],
    ) -> list[str]:
        """Generate optimization recommendations."""
        recommendations = []

        if quality_issues > 0:
            recommendations.append(
                f"Review quality control - {quality_issues} issues detected"
            )

        if len(bottlenecks) > 0:
            recommendations.append(
                "Consider adding capacity to bottleneck equipment"
            )

        if blends_completed > 0:
            throughput = blends_completed
            recommendations.append(
                f"Current throughput: {throughput} blends - "
                "evaluate campaign scheduling for improvement"
            )

        return recommendations

    def _log_event(
        self,
        event_type: str,
        subject: str,
        details: Any = None,
    ) -> None:
        """Log simulation event."""
        self.event_log.append({
            "time": self.simulation_time.isoformat(),
            "type": event_type,
            "subject": subject,
            "details": details,
        })

    async def run_what_if(
        self,
        scenario: str,
        parameters: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Run a what-if analysis.

        Common scenarios:
        - equipment_failure: What if a pump fails?
        - capacity_increase: What if we add more tank capacity?
        - demand_spike: What if orders increase 50%?
        - recipe_change: What if we change a recipe formulation?
        """
        # Save current state
        original_state = {
            "tanks": {t: vars(v).copy() for t, v in self.tanks.items()},
            "pumps": {p: vars(v).copy() for p, v in self.pumps.items()},
        }

        results = {}

        if scenario == "equipment_failure":
            equipment = parameters.get("equipment", "P-001")
            failure_duration = parameters.get("duration_hours", 4)

            # Simulate failure
            if equipment in self.pumps:
                self.pumps[equipment].status = PumpStatus.FAILED

            sim_result = await self.run_simulation(
                f"Equipment Failure - {equipment}",
                duration_hours=8,
                speed=SimulationSpeed.INSTANT,
            )

            results = {
                "scenario": scenario,
                "equipment_failed": equipment,
                "impact": {
                    "blends_affected": sim_result.blends_completed,
                    "volume_lost": parameters.get("expected_volume", 10000) - sim_result.total_volume_produced,
                    "recommendations": sim_result.recommendations,
                },
            }

        elif scenario == "capacity_increase":
            tank = parameters.get("tank", "BT-01")
            increase_percent = parameters.get("increase_percent", 25)

            if tank in self.tanks:
                original_capacity = self.tanks[tank].capacity_liters
                new_capacity = original_capacity * (1 + increase_percent / 100)

                # Run with increased capacity
                self.tanks[tank].capacity_liters = new_capacity

                sim_result = await self.run_simulation(
                    f"Capacity Increase - {tank}",
                    duration_hours=24,
                    speed=SimulationSpeed.INSTANT,
                )

                results = {
                    "scenario": scenario,
                    "tank": tank,
                    "original_capacity": original_capacity,
                    "new_capacity": new_capacity,
                    "impact": {
                        "additional_throughput": sim_result.total_volume_produced,
                        "utilization_change": sim_result.equipment_utilization,
                        "bottlenecks": sim_result.bottlenecks,
                    },
                }

        elif scenario == "demand_spike":
            increase_percent = parameters.get("increase_percent", 50)
            additional_blends = parameters.get("additional_blends", 5)

            # Add more blends to queue
            for i in range(additional_blends):
                blend = TwinBlend(
                    id=str(uuid4()),
                    batch_number=f"SPIKE-{i+1}",
                    recipe_code="SAE-10W40",
                    target_volume=5000,
                )
                self.blends[blend.id] = blend

            sim_result = await self.run_simulation(
                "Demand Spike",
                duration_hours=48,
                speed=SimulationSpeed.INSTANT,
            )

            results = {
                "scenario": scenario,
                "additional_blends": additional_blends,
                "impact": {
                    "can_handle": sim_result.blends_completed >= additional_blends,
                    "blends_completed": sim_result.blends_completed,
                    "bottlenecks": sim_result.bottlenecks,
                    "recommendations": sim_result.recommendations,
                },
            }

        # Restore original state
        for tag, state in original_state["tanks"].items():
            if tag in self.tanks:
                for k, v in state.items():
                    setattr(self.tanks[tag], k, v)

        for tag, state in original_state["pumps"].items():
            if tag in self.pumps:
                for k, v in state.items():
                    setattr(self.pumps[tag], k, v)

        return results

    async def validate_process_change(
        self,
        change_type: str,
        change_details: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Validate a proposed process change before implementation.

        Returns risk assessment and expected outcomes.
        """
        await self.sync_with_plant()

        risks = []
        benefits = []
        recommendation = "proceed"

        if change_type == "recipe_modification":
            recipe_code = change_details.get("recipe_code")
            changes = change_details.get("changes", {})

            # Check ingredient availability
            for material, percentage in changes.items():
                if percentage > 50:
                    risks.append(f"High percentage of {material} may affect quality")

            if len(risks) == 0:
                benefits.append("Recipe change appears safe")
                benefits.append("Run small test batch recommended")

        elif change_type == "equipment_change":
            equipment = change_details.get("equipment")
            change = change_details.get("change")

            # Run simulation with change
            sim_result = await self.run_simulation(
                f"Validate {change}",
                duration_hours=24,
                speed=SimulationSpeed.INSTANT,
                modifications={change: change_details.get("parameters", {})},
            )

            if sim_result.quality_issues > 0:
                risks.append("Quality issues detected in simulation")
                recommendation = "review"
            else:
                benefits.append(f"Simulation successful: {sim_result.blends_completed} blends")

        elif change_type == "schedule_change":
            # Validate schedule feasibility
            blends = change_details.get("blends", [])

            # Add to simulation
            for b in blends:
                self.blends[str(uuid4())] = TwinBlend(
                    id=str(uuid4()),
                    batch_number=b.get("batch", "TEST"),
                    recipe_code=b.get("recipe", "TEST"),
                    target_volume=b.get("volume", 5000),
                )

            sim_result = await self.run_simulation(
                "Schedule Validation",
                duration_hours=72,
                speed=SimulationSpeed.INSTANT,
            )

            if len(sim_result.bottlenecks) > 0:
                risks.extend(sim_result.bottlenecks)
                recommendation = "modify_schedule"
            else:
                benefits.append("Schedule is feasible")

        return {
            "change_type": change_type,
            "risks": risks,
            "benefits": benefits,
            "recommendation": recommendation,
            "confidence": 0.85 if len(risks) == 0 else 0.60,
        }

    def get_current_state(self) -> dict[str, Any]:
        """Get current state of digital twin."""
        return {
            "simulation_time": self.simulation_time.isoformat(),
            "is_running": self.is_running,
            "speed": self.speed.value,
            "tanks": {
                tag: {
                    "level": t.current_level,
                    "capacity": t.capacity_liters,
                    "temperature": t.temperature,
                    "mixing": t.is_mixing,
                }
                for tag, t in self.tanks.items()
            },
            "pumps": {
                tag: {
                    "status": p.status.value,
                    "flow_rate": p.flow_rate_lph,
                    "runtime": p.runtime_hours,
                }
                for tag, p in self.pumps.items()
            },
            "active_blends": len([b for b in self.blends.values() if b.status not in [BlendStatus.COMPLETED, BlendStatus.CANCELLED]]),
            "event_count": len(self.event_log),
        }

    def get_event_log(
        self,
        limit: int = 100,
        event_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get simulation event log."""
        events = self.event_log

        if event_type:
            events = [e for e in events if e["type"] == event_type]

        return events[-limit:]
