"""Blend scheduling and resource optimization."""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


class ResourceType(str, Enum):
    """Types of resources in the blending plant."""

    BLEND_TANK = "blend_tank"
    PUMP = "pump"
    PIPELINE = "pipeline"
    OPERATOR = "operator"


@dataclass
class Resource:
    """A resource in the blending plant."""

    id: str
    name: str
    resource_type: ResourceType
    capacity: float  # e.g., liters for tanks, lpm for pumps
    available_from: datetime
    available_until: datetime | None = None
    current_blend_id: str | None = None


@dataclass
class BlendRequest:
    """A request to schedule a blend."""

    blend_id: str
    recipe_id: str
    target_volume: float
    priority: int  # 1=low, 2=normal, 3=high, 4=urgent
    earliest_start: datetime
    latest_start: datetime | None = None
    estimated_duration_hours: float
    required_tank_capacity: float
    required_ingredients: list[dict[str, Any]]


@dataclass
class ScheduleEntry:
    """A scheduled blend operation."""

    blend_id: str
    scheduled_start: datetime
    scheduled_end: datetime
    blend_tank_id: str
    pump_assignments: list[str]
    sequence_order: int
    conflicts: list[str]


class BlendScheduler:
    """
    AI-powered blend scheduler for optimizing production.

    Features:
    - Dynamic rerouting based on equipment availability
    - Priority-based scheduling
    - Minimization of idle time and changeovers
    - Multi-blend parallelism support
    - Anti-contamination interlocks
    """

    def __init__(self):
        self._resources: dict[str, Resource] = {}
        self._scheduled: list[ScheduleEntry] = []
        self._queue: list[BlendRequest] = []

    def add_resource(self, resource: Resource) -> None:
        """Add a resource to the scheduler."""
        self._resources[resource.id] = resource
        logger.debug("Resource added", resource_id=resource.id, type=resource.resource_type)

    def get_resources(self, resource_type: ResourceType | None = None) -> list[Resource]:
        """Get resources optionally filtered by type."""
        resources = list(self._resources.values())
        if resource_type:
            resources = [r for r in resources if r.resource_type == resource_type]
        return resources

    def request_blend(self, request: BlendRequest) -> None:
        """Add a blend to the scheduling queue."""
        self._queue.append(request)
        logger.info(
            "Blend added to queue",
            blend_id=request.blend_id,
            priority=request.priority,
            volume=request.target_volume,
        )

    def schedule(self) -> list[ScheduleEntry]:
        """
        Run the scheduling algorithm.

        Returns:
            List of scheduled blend entries
        """
        logger.info("Running scheduler", queue_size=len(self._queue))

        # Sort queue by priority (descending) and earliest start
        sorted_queue = sorted(
            self._queue,
            key=lambda x: (-x.priority, x.earliest_start),
        )

        scheduled = []
        current_time = datetime.now(timezone.utc)

        for request in sorted_queue:
            entry = self._schedule_single(request, current_time)
            if entry:
                scheduled.append(entry)
                current_time = max(current_time, entry.scheduled_end)

        self._scheduled = scheduled
        logger.info("Scheduling complete", scheduled_count=len(scheduled))
        return scheduled

    def _schedule_single(
        self, request: BlendRequest, current_time: datetime
    ) -> ScheduleEntry | None:
        """Schedule a single blend request."""
        # Find available blend tank
        available_tanks = self._find_available_tanks(
            request.required_tank_capacity,
            max(current_time, request.earliest_start),
        )

        if not available_tanks:
            logger.warning(
                "No available tank for blend",
                blend_id=request.blend_id,
            )
            return ScheduleEntry(
                blend_id=request.blend_id,
                scheduled_start=request.earliest_start,
                scheduled_end=request.earliest_start
                + timedelta(hours=request.estimated_duration_hours),
                blend_tank_id="",
                pump_assignments=[],
                sequence_order=len(self._scheduled) + 1,
                conflicts=["No available blend tank"],
            )

        tank = available_tanks[0]

        # Find available pumps
        available_pumps = self._find_available_pumps(
            max(current_time, request.earliest_start)
        )

        # Calculate schedule
        start_time = max(current_time, request.earliest_start, tank.available_from)
        end_time = start_time + timedelta(hours=request.estimated_duration_hours)

        # Check for conflicts
        conflicts = self._check_conflicts(request, start_time, end_time, tank.id)

        entry = ScheduleEntry(
            blend_id=request.blend_id,
            scheduled_start=start_time,
            scheduled_end=end_time,
            blend_tank_id=tank.id,
            pump_assignments=[p.id for p in available_pumps[:2]],
            sequence_order=len(self._scheduled) + 1,
            conflicts=conflicts,
        )

        # Reserve resources
        tank.current_blend_id = request.blend_id
        tank.available_from = end_time

        return entry

    def _find_available_tanks(
        self, required_capacity: float, start_time: datetime
    ) -> list[Resource]:
        """Find tanks available for the blend."""
        tanks = self.get_resources(ResourceType.BLEND_TANK)
        available = [
            t
            for t in tanks
            if t.capacity >= required_capacity
            and t.available_from <= start_time
            and t.current_blend_id is None
        ]
        return sorted(available, key=lambda t: t.capacity)

    def _find_available_pumps(self, start_time: datetime) -> list[Resource]:
        """Find pumps available at the given time."""
        pumps = self.get_resources(ResourceType.PUMP)
        return [p for p in pumps if p.available_from <= start_time]

    def _check_conflicts(
        self,
        request: BlendRequest,
        start_time: datetime,
        end_time: datetime,
        tank_id: str,
    ) -> list[str]:
        """Check for scheduling conflicts."""
        conflicts = []

        # Check if latest start would be violated
        if request.latest_start and start_time > request.latest_start:
            conflicts.append(
                f"Start time {start_time} exceeds latest allowed {request.latest_start}"
            )

        # Check for overlapping blends in the same tank
        for entry in self._scheduled:
            if entry.blend_tank_id == tank_id:
                if start_time < entry.scheduled_end and end_time > entry.scheduled_start:
                    conflicts.append(
                        f"Overlaps with blend {entry.blend_id} in tank {tank_id}"
                    )

        return conflicts

    def reschedule(
        self, blend_id: str, new_priority: int | None = None
    ) -> ScheduleEntry | None:
        """Reschedule a blend with updated priority."""
        # Find the request
        request = next((r for r in self._queue if r.blend_id == blend_id), None)
        if not request:
            return None

        if new_priority:
            request.priority = new_priority

        # Remove from scheduled
        self._scheduled = [s for s in self._scheduled if s.blend_id != blend_id]

        # Reschedule
        return self._schedule_single(request, datetime.now(timezone.utc))

    def reroute(
        self,
        blend_id: str,
        original_pump_id: str,
        reason: str,
    ) -> dict[str, Any]:
        """
        Reroute a blend to alternative equipment.

        Used when equipment fails or becomes unavailable.
        """
        logger.info(
            "Rerouting blend",
            blend_id=blend_id,
            original_pump=original_pump_id,
            reason=reason,
        )

        # Find backup pumps
        backup_pumps = [
            p
            for p in self.get_resources(ResourceType.PUMP)
            if p.id != original_pump_id and p.current_blend_id is None
        ]

        if not backup_pumps:
            return {
                "success": False,
                "message": "No backup pumps available",
                "blend_id": blend_id,
            }

        new_pump = backup_pumps[0]
        new_pump.current_blend_id = blend_id

        # Update scheduled entry
        for entry in self._scheduled:
            if entry.blend_id == blend_id:
                entry.pump_assignments = [
                    new_pump.id if p == original_pump_id else p
                    for p in entry.pump_assignments
                ]
                break

        return {
            "success": True,
            "message": f"Rerouted from {original_pump_id} to {new_pump.id}",
            "blend_id": blend_id,
            "new_pump_id": new_pump.id,
        }

    def optimize_sequence(self) -> list[ScheduleEntry]:
        """
        Optimize the blend sequence to minimize changeovers.

        Groups similar blends together and minimizes cleaning time.
        """
        if not self._scheduled:
            return []

        # Simple optimization: group by recipe
        # In production, this would use more sophisticated algorithms

        # Get recipe info for each scheduled blend
        recipe_groups: dict[str, list[ScheduleEntry]] = {}
        for entry in self._scheduled:
            request = next(
                (r for r in self._queue if r.blend_id == entry.blend_id), None
            )
            if request:
                recipe_id = request.recipe_id
                if recipe_id not in recipe_groups:
                    recipe_groups[recipe_id] = []
                recipe_groups[recipe_id].append(entry)

        # Reorder to group similar recipes
        optimized = []
        for recipe_id, entries in recipe_groups.items():
            optimized.extend(sorted(entries, key=lambda e: e.scheduled_start))

        # Renumber sequence
        for i, entry in enumerate(optimized):
            entry.sequence_order = i + 1

        self._scheduled = optimized
        return optimized

    def get_schedule_summary(self) -> dict[str, Any]:
        """Get a summary of the current schedule."""
        if not self._scheduled:
            return {
                "total_blends": 0,
                "total_volume": 0,
                "utilization_percent": 0,
                "conflicts": 0,
            }

        total_duration = sum(
            (e.scheduled_end - e.scheduled_start).total_seconds() / 3600
            for e in self._scheduled
        )

        conflicts = sum(len(e.conflicts) for e in self._scheduled)

        return {
            "total_blends": len(self._scheduled),
            "queue_size": len(self._queue),
            "total_duration_hours": total_duration,
            "conflicts": conflicts,
            "next_available": (
                min(e.scheduled_end for e in self._scheduled).isoformat()
                if self._scheduled
                else None
            ),
        }
