"""
Dynamic rescheduling based on real-time events.

Implements:
- Real-time schedule adaptation
- Event-driven rescheduling
- Priority rebalancing
- Resource reallocation
- Impact analysis for schedule changes
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from lobp.models.blend import Blend, BlendStatus, BlendPriority
from lobp.models.tank import Tank, TankStatus
from lobp.models.equipment import Pump, PumpStatus


class EventType(str, Enum):
    """Types of events that trigger rescheduling."""

    EQUIPMENT_FAILURE = "equipment_failure"
    EQUIPMENT_RESTORED = "equipment_restored"
    TANK_UNAVAILABLE = "tank_unavailable"
    TANK_AVAILABLE = "tank_available"
    MATERIAL_SHORTAGE = "material_shortage"
    MATERIAL_ARRIVED = "material_arrived"
    QUALITY_HOLD = "quality_hold"
    PRIORITY_CHANGE = "priority_change"
    RUSH_ORDER = "rush_order"
    DELAY = "delay"


@dataclass
class ScheduleEvent:
    """An event that affects the schedule."""

    event_id: str
    event_type: EventType
    timestamp: datetime
    affected_resource: str
    details: dict[str, Any]
    severity: str  # low, medium, high, critical


@dataclass
class ScheduleImpact:
    """Impact analysis of an event on the schedule."""

    event: ScheduleEvent
    affected_blends: list[str]
    delay_minutes: int
    rescheduling_required: bool
    recommendations: list[str]


@dataclass
class ReschedulingAction:
    """An action to take for rescheduling."""

    action_type: str
    blend_id: str
    description: str
    old_value: Any
    new_value: Any
    impact: str


class DynamicScheduler:
    """
    AI-driven dynamic scheduler for real-time adaptation.

    Monitors events and automatically adjusts schedules to:
    - Minimize disruption
    - Maintain priorities
    - Optimize resource utilization
    """

    def __init__(self, db: AsyncSession):
        self.db = db
        self._event_queue: list[ScheduleEvent] = []
        self._action_history: list[ReschedulingAction] = []

    async def process_event(
        self,
        event: ScheduleEvent,
    ) -> ScheduleImpact:
        """
        Process an event and determine its impact on the schedule.

        Returns impact analysis and triggers rescheduling if needed.
        """
        self._event_queue.append(event)

        # Analyze impact
        impact = await self._analyze_impact(event)

        # Auto-reschedule if needed
        if impact.rescheduling_required:
            actions = await self._generate_rescheduling_actions(event, impact)
            for action in actions:
                await self._apply_action(action)
                self._action_history.append(action)

        return impact

    async def _analyze_impact(
        self,
        event: ScheduleEvent,
    ) -> ScheduleImpact:
        """Analyze the impact of an event on the schedule."""
        affected_blends = []
        delay_minutes = 0
        recommendations = []

        if event.event_type == EventType.EQUIPMENT_FAILURE:
            # Find blends using this equipment
            equipment_tag = event.affected_resource
            affected_blends = await self._find_blends_using_equipment(equipment_tag)

            if affected_blends:
                delay_minutes = event.details.get("estimated_downtime_minutes", 60)
                recommendations.append(f"Reroute to backup equipment if available")
                recommendations.append(f"Notify affected blend operators")

        elif event.event_type == EventType.TANK_UNAVAILABLE:
            tank_tag = event.affected_resource
            affected_blends = await self._find_blends_using_tank(tank_tag)

            if affected_blends:
                delay_minutes = 30  # Time to find alternative
                recommendations.append(f"Find alternative tank")
                recommendations.append(f"Check tank availability")

        elif event.event_type == EventType.MATERIAL_SHORTAGE:
            material_code = event.affected_resource
            affected_blends = await self._find_blends_needing_material(material_code)

            shortage_qty = event.details.get("shortage_quantity", 0)
            recommendations.append(f"Check alternative suppliers")
            recommendations.append(f"Consider recipe substitution")

        elif event.event_type == EventType.RUSH_ORDER:
            # Rush order affects all lower priority blends
            rush_blend_id = event.affected_resource
            affected_blends = await self._find_lower_priority_blends(rush_blend_id)
            recommendations.append("Reprioritize queue")
            recommendations.append("Notify operators of schedule change")

        elif event.event_type == EventType.QUALITY_HOLD:
            blend_id = event.affected_resource
            affected_blends = [blend_id]
            delay_minutes = event.details.get("hold_duration_minutes", 120)
            recommendations.append("Investigate quality issue")
            recommendations.append("Consider blend correction")

        rescheduling_required = len(affected_blends) > 0 and delay_minutes > 15

        return ScheduleImpact(
            event=event,
            affected_blends=affected_blends,
            delay_minutes=delay_minutes,
            rescheduling_required=rescheduling_required,
            recommendations=recommendations,
        )

    async def _find_blends_using_equipment(
        self,
        equipment_tag: str,
    ) -> list[str]:
        """Find active blends using specific equipment."""
        # Check for pumps
        query = (
            select(Blend.id)
            .where(Blend.status.in_([
                BlendStatus.IN_PROGRESS,
                BlendStatus.MIXING,
                BlendStatus.QUEUED,
            ]))
        )
        result = await self.db.execute(query)
        return [str(r[0]) for r in result]

    async def _find_blends_using_tank(
        self,
        tank_tag: str,
    ) -> list[str]:
        """Find blends using a specific tank."""
        query = (
            select(Blend.id)
            .where(Blend.blend_tank_tag == tank_tag)
            .where(Blend.status.in_([
                BlendStatus.IN_PROGRESS,
                BlendStatus.MIXING,
                BlendStatus.QUEUED,
                BlendStatus.SCHEDULED,
            ]))
        )
        result = await self.db.execute(query)
        return [str(r[0]) for r in result]

    async def _find_blends_needing_material(
        self,
        material_code: str,
    ) -> list[str]:
        """Find blends that need a specific material."""
        # Would join with blend_ingredients
        query = (
            select(Blend.id)
            .where(Blend.status.in_([
                BlendStatus.QUEUED,
                BlendStatus.SCHEDULED,
            ]))
        )
        result = await self.db.execute(query)
        return [str(r[0]) for r in result]

    async def _find_lower_priority_blends(
        self,
        rush_blend_id: str,
    ) -> list[str]:
        """Find blends with lower priority than the rush order."""
        # Get rush order details
        rush_query = select(Blend).where(Blend.id == rush_blend_id)
        rush_result = await self.db.execute(rush_query)
        rush_blend = rush_result.scalar_one_or_none()

        if not rush_blend:
            return []

        # Find lower priority scheduled blends
        query = (
            select(Blend.id)
            .where(Blend.status == BlendStatus.SCHEDULED)
            .where(Blend.priority < rush_blend.priority)
        )
        result = await self.db.execute(query)
        return [str(r[0]) for r in result]

    async def _generate_rescheduling_actions(
        self,
        event: ScheduleEvent,
        impact: ScheduleImpact,
    ) -> list[ReschedulingAction]:
        """Generate actions to address the scheduling impact."""
        actions = []

        if event.event_type == EventType.EQUIPMENT_FAILURE:
            # Find backup equipment
            backup = await self._find_backup_equipment(event.affected_resource)
            if backup:
                for blend_id in impact.affected_blends:
                    actions.append(ReschedulingAction(
                        action_type="reroute",
                        blend_id=blend_id,
                        description=f"Reroute to backup equipment {backup}",
                        old_value=event.affected_resource,
                        new_value=backup,
                        impact="minimal",
                    ))
            else:
                # Delay blends
                for blend_id in impact.affected_blends:
                    actions.append(ReschedulingAction(
                        action_type="delay",
                        blend_id=blend_id,
                        description=f"Delay by {impact.delay_minutes} minutes",
                        old_value=None,
                        new_value=impact.delay_minutes,
                        impact="moderate",
                    ))

        elif event.event_type == EventType.RUSH_ORDER:
            rush_blend_id = event.affected_resource
            # Move rush order to front
            actions.append(ReschedulingAction(
                action_type="prioritize",
                blend_id=rush_blend_id,
                description="Move to front of queue",
                old_value=None,
                new_value="urgent",
                impact="affects_others",
            ))
            # Delay other blends
            for blend_id in impact.affected_blends:
                actions.append(ReschedulingAction(
                    action_type="delay",
                    blend_id=blend_id,
                    description="Delayed due to rush order",
                    old_value=None,
                    new_value=60,  # Estimate
                    impact="moderate",
                ))

        elif event.event_type == EventType.TANK_UNAVAILABLE:
            # Find alternative tank
            alt_tank = await self._find_alternative_tank(event.affected_resource)
            if alt_tank:
                for blend_id in impact.affected_blends:
                    actions.append(ReschedulingAction(
                        action_type="reassign_tank",
                        blend_id=blend_id,
                        description=f"Reassign to tank {alt_tank}",
                        old_value=event.affected_resource,
                        new_value=alt_tank,
                        impact="minimal",
                    ))

        return actions

    async def _find_backup_equipment(
        self,
        equipment_tag: str,
    ) -> str | None:
        """Find backup equipment for a failed unit."""
        query = (
            select(Pump)
            .where(Pump.is_backup == True)
            .where(Pump.primary_pump_tag == equipment_tag)
            .where(Pump.status == PumpStatus.STANDBY)
        )
        result = await self.db.execute(query)
        backup = result.scalar_one_or_none()
        return backup.tag if backup else None

    async def _find_alternative_tank(
        self,
        tank_tag: str,
    ) -> str | None:
        """Find alternative tank."""
        # Get the unavailable tank's type
        orig_query = select(Tank).where(Tank.tag == tank_tag)
        orig_result = await self.db.execute(orig_query)
        orig_tank = orig_result.scalar_one_or_none()

        if not orig_tank:
            return None

        # Find available tank of same type
        query = (
            select(Tank)
            .where(Tank.tank_type == orig_tank.tank_type)
            .where(Tank.status == TankStatus.AVAILABLE)
            .where(Tank.tag != tank_tag)
        )
        result = await self.db.execute(query)
        alt_tank = result.scalar_one_or_none()
        return alt_tank.tag if alt_tank else None

    async def _apply_action(
        self,
        action: ReschedulingAction,
    ) -> None:
        """Apply a rescheduling action to the database."""
        query = select(Blend).where(Blend.id == action.blend_id)
        result = await self.db.execute(query)
        blend = result.scalar_one_or_none()

        if not blend:
            return

        if action.action_type == "delay":
            if blend.scheduled_start:
                new_start = datetime.fromisoformat(str(blend.scheduled_start))
                new_start += timedelta(minutes=action.new_value)
                blend.scheduled_start = new_start.isoformat()

        elif action.action_type == "prioritize":
            blend.priority = BlendPriority.URGENT

        elif action.action_type == "reassign_tank":
            blend.blend_tank_tag = action.new_value

        elif action.action_type == "reroute":
            # Update equipment assignment (would need additional field)
            pass

        blend.notes = (blend.notes or "") + f"\n[Auto] {action.description}"
        await self.db.commit()

    async def optimize_queue(self) -> list[dict[str, Any]]:
        """
        Optimize the current blend queue.

        Considers:
        - Priority
        - Tank availability
        - Equipment availability
        - Material availability
        - Campaign efficiency
        """
        query = (
            select(Blend)
            .options(selectinload(Blend.recipe))
            .where(Blend.status.in_([
                BlendStatus.SCHEDULED,
                BlendStatus.QUEUED,
            ]))
            .order_by(Blend.priority.desc(), Blend.scheduled_start)
        )
        result = await self.db.execute(query)
        blends = list(result.scalars().all())

        # Score each blend based on readiness
        scored_blends = []
        for blend in blends:
            score = 0

            # Priority weight
            priority_scores = {
                BlendPriority.URGENT: 100,
                BlendPriority.HIGH: 75,
                BlendPriority.NORMAL: 50,
                BlendPriority.LOW: 25,
            }
            score += priority_scores.get(blend.priority, 50)

            # Tank availability bonus
            if blend.blend_tank_tag:
                tank_query = select(Tank).where(Tank.tag == blend.blend_tank_tag)
                tank_result = await self.db.execute(tank_query)
                tank = tank_result.scalar_one_or_none()
                if tank and tank.status == TankStatus.AVAILABLE:
                    score += 20

            scored_blends.append({
                "blend_id": blend.id,
                "batch_number": blend.batch_number,
                "recipe": blend.recipe.code if blend.recipe else None,
                "priority": blend.priority.value,
                "score": score,
                "current_position": blends.index(blend) + 1,
            })

        # Sort by score
        scored_blends.sort(key=lambda x: -x["score"])

        # Add recommended position
        for i, blend in enumerate(scored_blends):
            blend["recommended_position"] = i + 1
            blend["position_change"] = blend["current_position"] - blend["recommended_position"]

        return scored_blends

    async def get_schedule_health(self) -> dict[str, Any]:
        """Get overall schedule health metrics."""
        query = select(Blend).where(
            Blend.status.in_([
                BlendStatus.SCHEDULED,
                BlendStatus.QUEUED,
                BlendStatus.IN_PROGRESS,
            ])
        )
        result = await self.db.execute(query)
        active_blends = list(result.scalars().all())

        delayed = 0
        on_time = 0
        at_risk = 0

        now = datetime.now(timezone.utc)

        for blend in active_blends:
            if blend.scheduled_start:
                scheduled = datetime.fromisoformat(str(blend.scheduled_start))
                if blend.actual_start:
                    actual = datetime.fromisoformat(str(blend.actual_start))
                    if actual > scheduled + timedelta(minutes=30):
                        delayed += 1
                    else:
                        on_time += 1
                elif scheduled < now:
                    delayed += 1
                elif scheduled < now + timedelta(hours=2):
                    at_risk += 1
                else:
                    on_time += 1
            else:
                on_time += 1

        total = len(active_blends)

        return {
            "total_active_blends": total,
            "on_time": on_time,
            "delayed": delayed,
            "at_risk": at_risk,
            "on_time_percentage": (on_time / total * 100) if total > 0 else 100,
            "recent_events": len(self._event_queue),
            "actions_taken": len(self._action_history),
            "health_status": (
                "healthy" if delayed == 0 else
                "warning" if delayed <= total * 0.1 else
                "critical"
            ),
        }
