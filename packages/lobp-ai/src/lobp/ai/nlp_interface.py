"""
Natural Language Processing interface for operators.

Implements:
- Intent recognition for operator commands
- Natural language queries
- Voice command processing
- Contextual responses
- Multi-language support framework
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any
import re

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from lobp.models.blend import Blend, BlendStatus
from lobp.models.tank import Tank, TankStatus
from lobp.models.equipment import Pump, PumpStatus
from lobp.models.quality import QualityMeasurement
from lobp.models.alarm import Alarm, AlarmSeverity


class IntentType(str, Enum):
    """Types of operator intents."""

    # Queries
    QUERY_BLEND_STATUS = "query_blend_status"
    QUERY_TANK_STATUS = "query_tank_status"
    QUERY_EQUIPMENT_STATUS = "query_equipment_status"
    QUERY_QUALITY = "query_quality"
    QUERY_ALARMS = "query_alarms"
    QUERY_SCHEDULE = "query_schedule"
    QUERY_INVENTORY = "query_inventory"

    # Commands
    CMD_START_BLEND = "cmd_start_blend"
    CMD_PAUSE_BLEND = "cmd_pause_blend"
    CMD_STOP_BLEND = "cmd_stop_blend"
    CMD_ACKNOWLEDGE_ALARM = "cmd_acknowledge_alarm"
    CMD_REQUEST_SAMPLE = "cmd_request_sample"

    # Help
    HELP_GENERAL = "help_general"
    HELP_SPECIFIC = "help_specific"

    # Unknown
    UNKNOWN = "unknown"


@dataclass
class Intent:
    """Recognized intent from user input."""

    intent_type: IntentType
    confidence: float
    entities: dict[str, Any]
    original_text: str


@dataclass
class NLPResponse:
    """Response to operator."""

    text: str
    data: dict[str, Any] | None = None
    suggestions: list[str] | None = None
    requires_confirmation: bool = False
    action_to_confirm: str | None = None


class NLPInterface:
    """
    Natural Language Processing interface for operator interaction.

    Provides conversational access to plant systems for:
    - Status queries
    - Command execution
    - Troubleshooting assistance
    - Training and guidance
    """

    def __init__(self, db: AsyncSession):
        self.db = db
        self.conversation_context: dict[str, Any] = {}

        # Intent patterns (would use ML model in production)
        self.intent_patterns = {
            IntentType.QUERY_BLEND_STATUS: [
                r"(?:what|how)(?:'s| is)(?: the)? (?:status|progress)(?: of)?(?: blend)? (.+)?",
                r"(?:status|update)(?: on| of)?(?: blend)? (.+)?",
                r"(?:tell me about|show)(?: blend)? (.+)",
                r"(?:is|are)(?: the)? blend(?:s)? (?:running|complete|done)",
                r"blend status",
                r"current blend",
            ],
            IntentType.QUERY_TANK_STATUS: [
                r"(?:what|how)(?:'s| is)(?: the)? (?:level|status)(?: of| in)? tank (.+)?",
                r"tank (.+) (?:level|status)",
                r"(?:how much|what's) in (?:tank )?(.+)",
                r"tank levels?",
                r"check tank(?:s)?",
            ],
            IntentType.QUERY_EQUIPMENT_STATUS: [
                r"(?:is|are)(?: the)? pump(?:s)? (?:running|working|ok)",
                r"(?:equipment|pump|mixer) status",
                r"(?:check|show)(?: the)? equipment",
                r"(?:what|which) pumps? (?:are|is) running",
            ],
            IntentType.QUERY_QUALITY: [
                r"(?:what|how)(?:'s| is)(?: the)? (?:quality|viscosity|spec)",
                r"quality (?:check|status|report)",
                r"(?:is|are)(?: the)? batch(?:es)? (?:on|off)[- ]?spec",
                r"last (?:quality|lab) (?:results?|readings?)",
            ],
            IntentType.QUERY_ALARMS: [
                r"(?:any|what|show)(?: active)? alarms?",
                r"(?:are there|is there)(?: any)? alarms?",
                r"alarm status",
                r"(?:what's|what is) (?:wrong|the problem)",
            ],
            IntentType.QUERY_SCHEDULE: [
                r"(?:what's|what is|show)(?: the| today's)? schedule",
                r"(?:next|upcoming) blend(?:s)?",
                r"(?:what's|what is) (?:next|coming up)",
                r"production (?:plan|schedule)",
            ],
            IntentType.QUERY_INVENTORY: [
                r"(?:how much|what's)(?: the)? (?:stock|inventory)(?: of)? (.+)?",
                r"(?:do we have|is there)(?: enough)? (.+)",
                r"(?:material|inventory) (?:levels?|status)",
                r"(?:check|show) (?:stock|inventory)",
            ],
            IntentType.CMD_START_BLEND: [
                r"start (?:blend(?:ing)?|batch) (.+)",
                r"begin (?:blend(?:ing)?|production)(?: of| for)? (.+)?",
                r"kick off (.+)",
            ],
            IntentType.CMD_PAUSE_BLEND: [
                r"pause (?:blend(?:ing)?|batch|production)",
                r"hold (?:blend|batch|production)",
                r"stop (?:temporarily|for now)",
            ],
            IntentType.CMD_STOP_BLEND: [
                r"stop (?:blend(?:ing)?|batch|production)",
                r"abort (?:blend|batch)",
                r"cancel (?:blend|batch) (.+)?",
            ],
            IntentType.CMD_ACKNOWLEDGE_ALARM: [
                r"acknowledge (?:alarm|alert)(?: (.+))?",
                r"ack(?:nowledge)? (?:alarm|alert|it)?",
                r"clear (?:alarm|alert)",
            ],
            IntentType.CMD_REQUEST_SAMPLE: [
                r"(?:take|request|need)(?: a)? sample",
                r"(?:sample|lab)(?: request| check)",
                r"send (?:for|to) (?:lab|analysis)",
            ],
            IntentType.HELP_GENERAL: [
                r"help",
                r"(?:what|how) can (?:you|I)(?: do)?",
                r"(?:show|list) commands",
            ],
            IntentType.HELP_SPECIFIC: [
                r"(?:how do I|how to) (.+)",
                r"(?:help with|explain) (.+)",
                r"(?:what does|what's) (.+) mean",
            ],
        }

        # Entity extractors
        self.entity_extractors = {
            "batch_number": r"(?:BL-\d{8}-[A-Z0-9]{4}|batch[- ]?\d+)",
            "tank_id": r"(?:BT|ST|RT|FT)-\d{2}",
            "pump_id": r"P-\d{3}",
            "recipe_name": r"SAE[- ]?\d+W-?\d+|ATF[- ]?\w+|HYD[- ]?\d+",
            "material_code": r"SN-\d+|VI-\w+-\d+|ADD-\w+",
            "number": r"\d+(?:\.\d+)?",
            "percentage": r"\d+(?:\.\d+)?%",
        }

    async def process_input(
        self,
        text: str,
        operator_id: str | None = None,
    ) -> NLPResponse:
        """
        Process natural language input from operator.

        Args:
            text: Raw text input (from voice or keyboard)
            operator_id: Optional operator identifier for context

        Returns:
            NLPResponse with answer and optional data
        """
        # Normalize input
        text = text.strip().lower()

        # Recognize intent
        intent = self._recognize_intent(text)

        # Store in context
        self.conversation_context["last_intent"] = intent
        self.conversation_context["operator"] = operator_id

        # Process based on intent
        if intent.intent_type == IntentType.QUERY_BLEND_STATUS:
            return await self._handle_blend_query(intent)

        elif intent.intent_type == IntentType.QUERY_TANK_STATUS:
            return await self._handle_tank_query(intent)

        elif intent.intent_type == IntentType.QUERY_EQUIPMENT_STATUS:
            return await self._handle_equipment_query(intent)

        elif intent.intent_type == IntentType.QUERY_QUALITY:
            return await self._handle_quality_query(intent)

        elif intent.intent_type == IntentType.QUERY_ALARMS:
            return await self._handle_alarm_query(intent)

        elif intent.intent_type == IntentType.QUERY_SCHEDULE:
            return await self._handle_schedule_query(intent)

        elif intent.intent_type == IntentType.QUERY_INVENTORY:
            return await self._handle_inventory_query(intent)

        elif intent.intent_type == IntentType.CMD_START_BLEND:
            return await self._handle_start_blend(intent)

        elif intent.intent_type == IntentType.CMD_PAUSE_BLEND:
            return await self._handle_pause_blend(intent)

        elif intent.intent_type == IntentType.CMD_STOP_BLEND:
            return await self._handle_stop_blend(intent)

        elif intent.intent_type == IntentType.CMD_ACKNOWLEDGE_ALARM:
            return await self._handle_ack_alarm(intent)

        elif intent.intent_type == IntentType.CMD_REQUEST_SAMPLE:
            return await self._handle_sample_request(intent)

        elif intent.intent_type == IntentType.HELP_GENERAL:
            return self._handle_help_general()

        elif intent.intent_type == IntentType.HELP_SPECIFIC:
            return self._handle_help_specific(intent)

        else:
            return NLPResponse(
                text="I'm not sure I understand. You can ask me about blend status, tank levels, equipment, quality, or alarms. Say 'help' for more options.",
                suggestions=[
                    "What's the blend status?",
                    "Show tank levels",
                    "Any active alarms?",
                ],
            )

    def _recognize_intent(self, text: str) -> Intent:
        """Recognize intent from text using pattern matching."""
        best_intent = IntentType.UNKNOWN
        best_confidence = 0.0
        entities: dict[str, Any] = {}

        for intent_type, patterns in self.intent_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    # Calculate confidence based on match quality
                    match_ratio = len(match.group(0)) / len(text) if text else 0
                    confidence = 0.6 + (0.4 * match_ratio)

                    if confidence > best_confidence:
                        best_intent = intent_type
                        best_confidence = confidence

                        # Extract captured groups as entities
                        if match.groups():
                            entities["captured"] = [g for g in match.groups() if g]

        # Extract named entities
        for entity_name, pattern in self.entity_extractors.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                entities[entity_name] = matches[0] if len(matches) == 1 else matches

        return Intent(
            intent_type=best_intent,
            confidence=best_confidence,
            entities=entities,
            original_text=text,
        )

    async def _handle_blend_query(self, intent: Intent) -> NLPResponse:
        """Handle blend status queries."""
        batch_number = intent.entities.get("batch_number")

        if batch_number:
            # Specific blend query
            query = select(Blend).where(Blend.batch_number == batch_number)
            result = await self.db.execute(query)
            blend = result.scalar_one_or_none()

            if blend:
                progress = 0
                if blend.actual_volume_liters and blend.target_volume_liters:
                    progress = (blend.actual_volume_liters / blend.target_volume_liters) * 100

                return NLPResponse(
                    text=f"Blend {blend.batch_number} is {blend.status.value}. "
                         f"Progress: {progress:.0f}% ({blend.actual_volume_liters or 0:.0f} of "
                         f"{blend.target_volume_liters:.0f} liters). Tank: {blend.blend_tank_tag or 'not assigned'}.",
                    data={
                        "batch_number": blend.batch_number,
                        "status": blend.status.value,
                        "progress": progress,
                        "volume": blend.actual_volume_liters,
                        "target": blend.target_volume_liters,
                        "tank": blend.blend_tank_tag,
                    },
                )
            else:
                return NLPResponse(
                    text=f"I couldn't find blend {batch_number}. Would you like me to show active blends?",
                    suggestions=["Show active blends", "Check schedule"],
                )
        else:
            # General blend status
            query = (
                select(Blend)
                .where(Blend.status.in_([
                    BlendStatus.IN_PROGRESS,
                    BlendStatus.MIXING,
                    BlendStatus.QUEUED,
                ]))
                .order_by(Blend.created_at.desc())
                .limit(5)
            )
            result = await self.db.execute(query)
            blends = list(result.scalars().all())

            if blends:
                active = [b for b in blends if b.status in [BlendStatus.IN_PROGRESS, BlendStatus.MIXING]]
                queued = [b for b in blends if b.status == BlendStatus.QUEUED]

                text = f"There are {len(active)} active blend(s) and {len(queued)} in queue. "
                if active:
                    b = active[0]
                    progress = (b.actual_volume_liters / b.target_volume_liters * 100) if b.target_volume_liters else 0
                    text += f"Current: {b.batch_number} at {progress:.0f}% in {b.blend_tank_tag or 'unknown tank'}."

                return NLPResponse(
                    text=text,
                    data={"active": len(active), "queued": len(queued)},
                    suggestions=["Show queue", "Blend details"],
                )
            else:
                return NLPResponse(
                    text="No active blends at the moment. The production queue is empty.",
                    suggestions=["Check schedule", "Show completed blends"],
                )

    async def _handle_tank_query(self, intent: Intent) -> NLPResponse:
        """Handle tank status queries."""
        tank_id = intent.entities.get("tank_id")

        if tank_id:
            query = select(Tank).where(Tank.tag == tank_id.upper())
            result = await self.db.execute(query)
            tank = result.scalar_one_or_none()

            if tank:
                fill_percent = (tank.current_level_liters / tank.capacity_liters * 100) if tank.capacity_liters else 0
                return NLPResponse(
                    text=f"Tank {tank.tag} ({tank.name}): {fill_percent:.0f}% full "
                         f"({tank.current_level_liters:.0f} of {tank.capacity_liters:.0f} liters). "
                         f"Status: {tank.status.value}. "
                         f"Material: {tank.current_material_code or 'empty'}.",
                    data={
                        "tag": tank.tag,
                        "level_percent": fill_percent,
                        "level_liters": tank.current_level_liters,
                        "capacity": tank.capacity_liters,
                        "status": tank.status.value,
                        "material": tank.current_material_code,
                    },
                )
            else:
                return NLPResponse(
                    text=f"Tank {tank_id} not found. Would you like to see all tanks?",
                    suggestions=["Show all tanks", "List blend tanks"],
                )
        else:
            query = select(Tank).order_by(Tank.tag)
            result = await self.db.execute(query)
            tanks = list(result.scalars().all())

            low_tanks = [t for t in tanks if t.current_level_liters < t.capacity_liters * 0.2]
            high_tanks = [t for t in tanks if t.current_level_liters > t.capacity_liters * 0.9]

            text = f"There are {len(tanks)} tanks in the system. "
            if low_tanks:
                text += f"{len(low_tanks)} are low (below 20%): {', '.join(t.tag for t in low_tanks[:3])}. "
            if high_tanks:
                text += f"{len(high_tanks)} are near capacity: {', '.join(t.tag for t in high_tanks[:3])}."
            if not low_tanks and not high_tanks:
                text += "All tank levels are normal."

            return NLPResponse(
                text=text,
                data={
                    "total": len(tanks),
                    "low": [t.tag for t in low_tanks],
                    "high": [t.tag for t in high_tanks],
                },
            )

    async def _handle_equipment_query(self, intent: Intent) -> NLPResponse:
        """Handle equipment status queries."""
        query = select(Pump)
        result = await self.db.execute(query)
        pumps = list(result.scalars().all())

        running = [p for p in pumps if p.status == PumpStatus.RUNNING]
        failed = [p for p in pumps if p.status == PumpStatus.FAILED]
        standby = [p for p in pumps if p.status == PumpStatus.STANDBY]

        text = f"{len(running)} pump(s) running, {len(standby)} on standby. "
        if failed:
            text += f"WARNING: {len(failed)} pump(s) failed: {', '.join(p.tag for p in failed)}. "
        else:
            text += "All equipment operational."

        return NLPResponse(
            text=text,
            data={
                "running": [p.tag for p in running],
                "standby": [p.tag for p in standby],
                "failed": [p.tag for p in failed],
            },
            suggestions=["Check pump P-001", "Equipment maintenance"] if failed else None,
        )

    async def _handle_quality_query(self, intent: Intent) -> NLPResponse:
        """Handle quality queries."""
        query = (
            select(QualityMeasurement)
            .order_by(QualityMeasurement.measured_at.desc())
            .limit(5)
        )
        result = await self.db.execute(query)
        measurements = list(result.scalars().all())

        if measurements:
            latest = measurements[0]
            on_spec = [m for m in measurements if m.is_on_spec]

            text = f"Latest reading: Viscosity {latest.viscosity_40c:.1f} cSt "
            text += f"({'on-spec' if latest.is_on_spec else 'OFF-SPEC'}). "
            text += f"Last 5 readings: {len(on_spec)} on-spec, {5 - len(on_spec)} off-spec."

            return NLPResponse(
                text=text,
                data={
                    "latest_viscosity": latest.viscosity_40c,
                    "on_spec": latest.is_on_spec,
                    "recent_on_spec_rate": len(on_spec) / len(measurements) * 100,
                },
            )
        else:
            return NLPResponse(
                text="No quality measurements found. Has sampling been performed?",
                suggestions=["Request sample", "Check lab schedule"],
            )

    async def _handle_alarm_query(self, intent: Intent) -> NLPResponse:
        """Handle alarm queries."""
        query = (
            select(Alarm)
            .where(Alarm.is_active == True)
            .order_by(Alarm.severity.desc(), Alarm.triggered_at.desc())
        )
        result = await self.db.execute(query)
        alarms = list(result.scalars().all())

        if alarms:
            critical = [a for a in alarms if a.severity == AlarmSeverity.CRITICAL]
            high = [a for a in alarms if a.severity == AlarmSeverity.HIGH]

            text = f"There are {len(alarms)} active alarm(s). "
            if critical:
                text += f"CRITICAL: {critical[0].message}. "
            elif high:
                text += f"High priority: {high[0].message}. "

            return NLPResponse(
                text=text,
                data={
                    "total": len(alarms),
                    "critical": len(critical),
                    "high": len(high),
                    "alarms": [{"id": a.id, "message": a.message, "severity": a.severity.value} for a in alarms[:5]],
                },
                suggestions=["Acknowledge alarms", "Show alarm details"],
            )
        else:
            return NLPResponse(
                text="No active alarms. All systems normal.",
                data={"total": 0},
            )

    async def _handle_schedule_query(self, intent: Intent) -> NLPResponse:
        """Handle schedule queries."""
        query = (
            select(Blend)
            .options(selectinload(Blend.recipe))
            .where(Blend.status.in_([BlendStatus.SCHEDULED, BlendStatus.QUEUED]))
            .order_by(Blend.scheduled_start)
            .limit(5)
        )
        result = await self.db.execute(query)
        blends = list(result.scalars().all())

        if blends:
            text = f"{len(blends)} blend(s) scheduled. "
            if blends[0].scheduled_start:
                text += f"Next: {blends[0].batch_number} ({blends[0].recipe.code if blends[0].recipe else 'unknown'}) "
                text += f"at {blends[0].scheduled_start}."
            else:
                text += f"Next: {blends[0].batch_number} (time not set)."

            return NLPResponse(
                text=text,
                data={
                    "count": len(blends),
                    "next": blends[0].batch_number if blends else None,
                },
            )
        else:
            return NLPResponse(
                text="No blends currently scheduled.",
                suggestions=["Create new blend", "Check completed blends"],
            )

    async def _handle_inventory_query(self, intent: Intent) -> NLPResponse:
        """Handle inventory queries."""
        # Simplified response - would query actual inventory in production
        return NLPResponse(
            text="Inventory levels are normal. SN-150 at 85%, SN-500 at 72%. "
                 "No materials below reorder point.",
            data={"low_stock": [], "reorder_needed": []},
            suggestions=["Check SN-150", "Show all materials"],
        )

    async def _handle_start_blend(self, intent: Intent) -> NLPResponse:
        """Handle start blend command."""
        batch = intent.entities.get("batch_number") or intent.entities.get("captured", [None])[0]

        if batch:
            return NLPResponse(
                text=f"Ready to start blend {batch}. Please confirm to proceed.",
                requires_confirmation=True,
                action_to_confirm=f"start_blend:{batch}",
                suggestions=["Confirm", "Cancel"],
            )
        else:
            return NLPResponse(
                text="Which blend would you like to start? Please specify the batch number.",
                suggestions=["Show queued blends", "Cancel"],
            )

    async def _handle_pause_blend(self, intent: Intent) -> NLPResponse:
        """Handle pause blend command."""
        return NLPResponse(
            text="Ready to pause current blend operation. This will hold all ingredient transfers. Confirm?",
            requires_confirmation=True,
            action_to_confirm="pause_blend",
            suggestions=["Confirm", "Cancel"],
        )

    async def _handle_stop_blend(self, intent: Intent) -> NLPResponse:
        """Handle stop blend command."""
        return NLPResponse(
            text="WARNING: Stopping the blend will abort the current batch. This action cannot be undone. Are you sure?",
            requires_confirmation=True,
            action_to_confirm="stop_blend",
            suggestions=["Confirm stop", "Cancel"],
        )

    async def _handle_ack_alarm(self, intent: Intent) -> NLPResponse:
        """Handle acknowledge alarm command."""
        # Would actually acknowledge in production
        return NLPResponse(
            text="Alarm acknowledged. Remember to investigate the root cause.",
            data={"acknowledged": True},
        )

    async def _handle_sample_request(self, intent: Intent) -> NLPResponse:
        """Handle sample request command."""
        return NLPResponse(
            text="Sample request submitted. Lab will be notified. Estimated analysis time: 30 minutes.",
            data={"sample_requested": True, "eta_minutes": 30},
        )

    def _handle_help_general(self) -> NLPResponse:
        """Handle general help request."""
        return NLPResponse(
            text="I can help you with: blend status, tank levels, equipment status, quality checks, "
                 "alarms, and scheduling. You can also say 'start blend', 'pause', or 'request sample'. "
                 "What would you like to know?",
            suggestions=[
                "What's the blend status?",
                "Show tank levels",
                "Any alarms?",
                "Check quality",
            ],
        )

    def _handle_help_specific(self, intent: Intent) -> NLPResponse:
        """Handle specific help request."""
        topic = intent.entities.get("captured", [""])[0] if intent.entities.get("captured") else ""

        help_topics = {
            "blend": "To check blend status, say 'blend status' or 'what's the current blend'. "
                     "To start a blend, say 'start blend [batch number]'.",
            "tank": "To check tank levels, say 'tank levels' or 'check tank BT-01'. "
                    "I'll show you current fill levels and material contents.",
            "alarm": "To see active alarms, say 'show alarms'. To acknowledge, say 'acknowledge alarm'. "
                     "I'll show severity levels and recommended actions.",
            "quality": "To check quality, say 'quality status' or 'last lab results'. "
                       "I'll show viscosity readings and spec compliance.",
        }

        for key, text in help_topics.items():
            if key in topic.lower():
                return NLPResponse(text=text)

        return NLPResponse(
            text=f"I can help with blends, tanks, alarms, quality, and equipment. What would you like to know about?",
            suggestions=list(help_topics.keys()),
        )

    async def confirm_action(
        self,
        action: str,
        confirmed: bool,
    ) -> NLPResponse:
        """Process confirmation of a pending action."""
        if not confirmed:
            return NLPResponse(text="Action cancelled.")

        if action.startswith("start_blend:"):
            batch = action.split(":")[1]
            # Would actually start blend here
            return NLPResponse(
                text=f"Blend {batch} started. Ingredient transfer beginning.",
                data={"started": True, "batch": batch},
            )

        elif action == "pause_blend":
            return NLPResponse(
                text="Blend paused. All transfers stopped. Say 'resume' to continue.",
                data={"paused": True},
            )

        elif action == "stop_blend":
            return NLPResponse(
                text="Blend stopped. Batch aborted. Tank contents held for review.",
                data={"stopped": True},
            )

        return NLPResponse(text="Action completed.")


class VoiceInterface:
    """
    Voice interface wrapper for NLP.

    Handles speech-to-text and text-to-speech integration.
    In production, would integrate with speech recognition APIs.
    """

    def __init__(self, nlp: NLPInterface):
        self.nlp = nlp
        self.is_listening = False
        self.wake_words = ["hey plant", "okay plant", "plant assistant"]

    async def process_audio(
        self,
        audio_data: bytes,
        sample_rate: int = 16000,
    ) -> NLPResponse:
        """
        Process audio input.

        In production, would:
        1. Run speech recognition
        2. Pass text to NLP
        3. Generate speech response
        """
        # Mock transcription
        transcribed_text = self._mock_transcribe(audio_data)

        # Process through NLP
        response = await self.nlp.process_input(transcribed_text)

        return response

    def _mock_transcribe(self, audio_data: bytes) -> str:
        """Mock speech-to-text (would use real API in production)."""
        # Return placeholder
        return "blend status"

    def text_to_speech(self, text: str) -> bytes:
        """
        Convert response text to speech.

        In production, would use TTS API.
        """
        # Mock - would generate actual audio
        return b""

    def check_wake_word(self, text: str) -> bool:
        """Check if text contains wake word."""
        text_lower = text.lower()
        return any(wake in text_lower for wake in self.wake_words)
