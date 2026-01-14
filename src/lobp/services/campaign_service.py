"""
Campaign blending service for optimizing production sequences.

Implements:
- Recipe similarity scoring
- Campaign grouping to minimize changeovers
- Optimal sequence planning
- Cleaning time estimation
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from lobp.models.blend import Blend, BlendStatus
from lobp.models.recipe import Recipe


@dataclass
class BlendSequenceItem:
    """Item in a blend sequence."""

    blend_id: str
    batch_number: str
    recipe_code: str
    recipe_name: str
    target_volume: float
    priority: int
    sequence_order: int
    estimated_start: datetime
    estimated_end: datetime
    changeover_time_minutes: int
    cleaning_required: bool


@dataclass
class Campaign:
    """A group of similar blends to run together."""

    campaign_id: str
    recipe_group: str
    blends: list[BlendSequenceItem]
    total_volume: float
    total_duration_hours: float
    changeover_savings_minutes: int


class CampaignService:
    """Service for campaign blending and sequence optimization."""

    def __init__(self, db: AsyncSession):
        self.db = db

        # Changeover times in minutes based on recipe transitions
        self.changeover_matrix = {
            # (from_group, to_group): minutes
            ("engine_oil", "engine_oil"): 15,  # Same family, quick rinse
            ("engine_oil", "gear_oil"): 45,  # Different family
            ("engine_oil", "atf"): 60,  # Different family, more cleaning
            ("gear_oil", "engine_oil"): 45,
            ("gear_oil", "gear_oil"): 20,
            ("atf", "atf"): 15,
            ("atf", "engine_oil"): 60,
            ("default", "default"): 30,  # Default changeover
        }

        # Cleaning requirements
        self.full_clean_transitions = {
            ("atf", "engine_oil"),
            ("engine_oil", "atf"),
            ("synthetic", "mineral"),
            ("mineral", "synthetic"),
        }

    async def calculate_recipe_similarity(
        self,
        recipe1_id: str,
        recipe2_id: str,
    ) -> float:
        """
        Calculate similarity score between two recipes (0-100).

        Based on:
        - Shared ingredients
        - Similar viscosity grades
        - Same product family
        """
        query = select(Recipe).options(selectinload(Recipe.ingredients))

        result1 = await self.db.execute(
            query.where(Recipe.id == recipe1_id)
        )
        result2 = await self.db.execute(
            query.where(Recipe.id == recipe2_id)
        )

        recipe1 = result1.scalar_one_or_none()
        recipe2 = result2.scalar_one_or_none()

        if not recipe1 or not recipe2:
            return 0.0

        score = 0.0

        # Check ingredient overlap (40% of score)
        ing1_codes = {i.material_code for i in recipe1.ingredients}
        ing2_codes = {i.material_code for i in recipe2.ingredients}

        if ing1_codes and ing2_codes:
            overlap = len(ing1_codes & ing2_codes)
            total = len(ing1_codes | ing2_codes)
            ingredient_score = (overlap / total) * 40 if total > 0 else 0
            score += ingredient_score

        # Check viscosity similarity (30% of score)
        if recipe1.target_viscosity_40c and recipe2.target_viscosity_40c:
            visc_diff = abs(
                recipe1.target_viscosity_40c - recipe2.target_viscosity_40c
            )
            max_visc = max(recipe1.target_viscosity_40c, recipe2.target_viscosity_40c)
            visc_similarity = max(0, 1 - (visc_diff / max_visc)) * 30
            score += visc_similarity

        # Check recipe code prefix (product family) (30% of score)
        if recipe1.code and recipe2.code:
            # Assume product family is first part of code (e.g., "SAE-10W40" -> "SAE")
            family1 = recipe1.code.split("-")[0] if "-" in recipe1.code else recipe1.code[:3]
            family2 = recipe2.code.split("-")[0] if "-" in recipe2.code else recipe2.code[:3]
            if family1 == family2:
                score += 30

        return min(100, score)

    def _get_recipe_group(self, recipe_code: str) -> str:
        """Determine recipe group from code."""
        code_upper = recipe_code.upper()

        if "ATF" in code_upper or "DEXRON" in code_upper:
            return "atf"
        elif "GEAR" in code_upper or "GL-" in code_upper:
            return "gear_oil"
        elif any(x in code_upper for x in ["SAE", "10W", "15W", "20W", "5W"]):
            return "engine_oil"
        else:
            return "default"

    def _get_changeover_time(
        self,
        from_recipe_code: str | None,
        to_recipe_code: str,
    ) -> tuple[int, bool]:
        """
        Get changeover time and cleaning requirement.

        Returns:
            Tuple of (minutes, requires_full_clean)
        """
        if not from_recipe_code:
            return (0, False)  # First blend, no changeover

        from_group = self._get_recipe_group(from_recipe_code)
        to_group = self._get_recipe_group(to_recipe_code)

        key = (from_group, to_group)
        time = self.changeover_matrix.get(key, self.changeover_matrix[("default", "default")])
        full_clean = key in self.full_clean_transitions

        return (time, full_clean)

    async def get_pending_blends(self) -> list[Blend]:
        """Get all blends pending scheduling."""
        query = (
            select(Blend)
            .options(selectinload(Blend.recipe))
            .where(Blend.status.in_([BlendStatus.DRAFT, BlendStatus.SCHEDULED, BlendStatus.QUEUED]))
            .order_by(Blend.priority.desc(), Blend.created_at)
        )
        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def optimize_sequence(
        self,
        blend_ids: list[str] | None = None,
        start_time: datetime | None = None,
    ) -> list[BlendSequenceItem]:
        """
        Optimize blend sequence to minimize changeovers.

        Groups similar recipes together and orders by priority
        within groups.

        Args:
            blend_ids: Optional specific blends to sequence (else all pending)
            start_time: When production starts

        Returns:
            Optimized sequence of blends
        """
        start_time = start_time or datetime.now(timezone.utc)

        # Get blends to sequence
        if blend_ids:
            query = (
                select(Blend)
                .options(selectinload(Blend.recipe))
                .where(Blend.id.in_(blend_ids))
            )
            result = await self.db.execute(query)
            blends = list(result.scalars().all())
        else:
            blends = await self.get_pending_blends()

        if not blends:
            return []

        # Group blends by recipe group
        groups: dict[str, list[Blend]] = {}
        for blend in blends:
            recipe_code = blend.recipe.code if blend.recipe else "UNKNOWN"
            group = self._get_recipe_group(recipe_code)
            if group not in groups:
                groups[group] = []
            groups[group].append(blend)

        # Sort within each group by priority (desc) then volume (desc for efficiency)
        for group in groups.values():
            group.sort(
                key=lambda b: (-b.priority.value if hasattr(b.priority, 'value') else -ord(b.priority[0]),
                               -b.target_volume_liters)
            )

        # Determine optimal group order (minimize transitions)
        # Simple heuristic: start with largest group, prefer same family
        group_order = sorted(groups.keys(), key=lambda g: -len(groups[g]))

        # Build optimized sequence
        sequence = []
        current_time = start_time
        previous_recipe_code = None
        order = 1

        for group in group_order:
            for blend in groups[group]:
                recipe_code = blend.recipe.code if blend.recipe else "UNKNOWN"

                changeover_time, cleaning_required = self._get_changeover_time(
                    previous_recipe_code, recipe_code
                )

                # Add changeover time
                current_time += timedelta(minutes=changeover_time)

                # Estimate blend duration (based on volume)
                blend_duration_hours = self._estimate_blend_duration(
                    blend.target_volume_liters
                )

                estimated_end = current_time + timedelta(hours=blend_duration_hours)

                item = BlendSequenceItem(
                    blend_id=blend.id,
                    batch_number=blend.batch_number,
                    recipe_code=recipe_code,
                    recipe_name=blend.recipe.name if blend.recipe else "Unknown",
                    target_volume=blend.target_volume_liters,
                    priority=blend.priority.value if hasattr(blend.priority, 'value') else 2,
                    sequence_order=order,
                    estimated_start=current_time,
                    estimated_end=estimated_end,
                    changeover_time_minutes=changeover_time,
                    cleaning_required=cleaning_required,
                )
                sequence.append(item)

                current_time = estimated_end
                previous_recipe_code = recipe_code
                order += 1

        return sequence

    def _estimate_blend_duration(self, volume_liters: float) -> float:
        """Estimate blend duration in hours based on volume."""
        # Base time + time per 1000L
        base_hours = 0.5
        hours_per_1000l = 0.3
        return base_hours + (volume_liters / 1000) * hours_per_1000l

    async def create_campaigns(
        self,
        blend_ids: list[str] | None = None,
    ) -> list[Campaign]:
        """
        Group blends into campaigns for efficient production.

        Returns list of campaigns with estimated savings.
        """
        sequence = await self.optimize_sequence(blend_ids)

        if not sequence:
            return []

        campaigns = []
        current_campaign_blends = []
        current_group = None
        campaign_number = 1

        for item in sequence:
            item_group = self._get_recipe_group(item.recipe_code)

            if current_group is None:
                current_group = item_group

            if item_group != current_group:
                # Save current campaign
                if current_campaign_blends:
                    campaigns.append(self._create_campaign(
                        f"C{campaign_number:03d}",
                        current_group,
                        current_campaign_blends,
                    ))
                    campaign_number += 1

                # Start new campaign
                current_campaign_blends = [item]
                current_group = item_group
            else:
                current_campaign_blends.append(item)

        # Don't forget last campaign
        if current_campaign_blends:
            campaigns.append(self._create_campaign(
                f"C{campaign_number:03d}",
                current_group,
                current_campaign_blends,
            ))

        return campaigns

    def _create_campaign(
        self,
        campaign_id: str,
        recipe_group: str,
        blends: list[BlendSequenceItem],
    ) -> Campaign:
        """Create a campaign from a list of blend sequence items."""
        total_volume = sum(b.target_volume for b in blends)

        if blends:
            duration = (blends[-1].estimated_end - blends[0].estimated_start).total_seconds() / 3600
        else:
            duration = 0

        # Calculate changeover savings
        # Without campaign: each blend needs full changeover (assume 45 min)
        # With campaign: only first blend and intra-campaign changeovers
        without_campaign = len(blends) * 45
        with_campaign = sum(b.changeover_time_minutes for b in blends)
        savings = without_campaign - with_campaign

        return Campaign(
            campaign_id=campaign_id,
            recipe_group=recipe_group,
            blends=blends,
            total_volume=total_volume,
            total_duration_hours=duration,
            changeover_savings_minutes=max(0, savings),
        )

    async def get_sequence_summary(
        self,
        blend_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Get summary of optimized sequence."""
        sequence = await self.optimize_sequence(blend_ids)
        campaigns = await self.create_campaigns(blend_ids)

        if not sequence:
            return {
                "total_blends": 0,
                "total_volume": 0,
                "campaigns": [],
            }

        total_changeover = sum(s.changeover_time_minutes for s in sequence)
        total_volume = sum(s.target_volume for s in sequence)

        return {
            "total_blends": len(sequence),
            "total_volume": total_volume,
            "total_changeover_minutes": total_changeover,
            "total_cleaning_stops": sum(1 for s in sequence if s.cleaning_required),
            "estimated_start": sequence[0].estimated_start.isoformat(),
            "estimated_end": sequence[-1].estimated_end.isoformat(),
            "campaigns": [
                {
                    "campaign_id": c.campaign_id,
                    "recipe_group": c.recipe_group,
                    "blend_count": len(c.blends),
                    "total_volume": c.total_volume,
                    "duration_hours": c.total_duration_hours,
                    "changeover_savings_minutes": c.changeover_savings_minutes,
                }
                for c in campaigns
            ],
            "total_savings_minutes": sum(c.changeover_savings_minutes for c in campaigns),
        }
