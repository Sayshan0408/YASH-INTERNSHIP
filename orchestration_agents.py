"""
agents/orchestration/home_orchestrator.py  — Tier 3
Central planning agent. Translates high-level goals into domain commands.
"""
import logging
from agents.base_agent import BaseAgent
from bus.message_bus import Message

logger = logging.getLogger(__name__)

# Intent → list of domain commands
INTENT_PLANS: dict[str, list[dict]] = {
    "good_morning": [
        {"topic": "cmd.lighting.scene",  "payload": {"scene": "morning"}},
        {"topic": "cmd.climate.comfort", "payload": {"action": "set_comfort"}},
    ],
    "good_night": [
        {"topic": "cmd.lighting.scene",  "payload": {"scene": "off"}},
        {"topic": "cmd.climate.eco",     "payload": {"action": "set_eco"}},
        {"topic": "cmd.security.arm",    "payload": {"action": "arm"}},
    ],
    "leaving_home": [
        {"topic": "cmd.lighting.scene",  "payload": {"scene": "off"}},
        {"topic": "cmd.climate.eco",     "payload": {"action": "set_eco"}},
        {"topic": "cmd.security.arm",    "payload": {"action": "arm"}},
        {"topic": "cmd.ev.ev_01",        "payload": {"charging": True, "rate_kw": 7.4}},
    ],
    "arriving_home": [
        {"topic": "cmd.lighting.scene",  "payload": {"scene": "evening"}},
        {"topic": "cmd.climate.comfort", "payload": {"action": "set_comfort"}},
        {"topic": "cmd.security.disarm", "payload": {"action": "disarm"}},
    ],
    "movie_mode": [
        {"topic": "cmd.lighting.scene",  "payload": {"scene": "movie"}},
        {"topic": "cmd.climate.eco",     "payload": {"action": "set_eco"}},
    ],
}


class HomeOrchestrator(BaseAgent):
    def __init__(self, bus, policy_agent=None):
        super().__init__("orchestration.home", bus)
        self.policy = policy_agent

    @property
    def publishes(self):
        return [f"cmd.{domain}.#" for domain in
                ("lighting", "climate", "security", "energy", "ev", "appliance")]

    @property
    def subscribes(self):
        return [
            "intent.#",
            "presence.away",
            "presence.home",
            "domain.alert.#",
            "learning.routine.proposed",
        ]

    async def handle_message(self, msg: Message) -> None:
        if msg.topic.startswith("intent"):
            intent = msg.topic.split(".")[-1]
            await self._execute_plan(intent, msg.payload)

        elif msg.topic == "presence.away":
            await self._execute_plan("leaving_home", {})

        elif msg.topic == "presence.home":
            await self._execute_plan("arriving_home", {})

        elif msg.topic.startswith("domain.alert"):
            self.log(f"Alert received: {msg.payload}")

        elif msg.topic == "learning.routine.proposed":
            self.log(f"Routine suggestion: {msg.payload.get('suggestion')}")

    async def _execute_plan(self, intent: str, context: dict) -> None:
        plan = INTENT_PLANS.get(intent)
        if not plan:
            self.log(f"No plan for intent '{intent}'")
            return

        self.log(f"Executing plan: {intent} ({len(plan)} steps)")
        for step in plan:
            topic = step["topic"]
            payload = {**step["payload"], **context}

            # Policy check before dispatch
            if self.policy:
                allowed, reason = self.policy.is_allowed(topic, payload)
                if not allowed:
                    self.log(f"Step blocked by policy: {reason}")
                    continue

            await self.publish(topic, payload)


"""
agents/orchestration/user_intent_agent.py  — Tier 3
Parses natural language / voice commands into structured intents.
"""
from agents.base_agent import BaseAgent
from bus.message_bus import Message as Msg2

KEYWORD_INTENTS = {
    "morning":    "good_morning",
    "wake":       "good_morning",
    "night":      "good_night",
    "sleep":      "good_night",
    "leaving":    "leaving_home",
    "goodbye":    "leaving_home",
    "home":       "arriving_home",
    "arrived":    "arriving_home",
    "movie":      "movie_mode",
    "cinema":     "movie_mode",
}


class UserIntentAgent(BaseAgent):
    def __init__(self, bus):
        super().__init__("orchestration.user_intent", bus)

    @property
    def publishes(self): return ["intent.#"]
    @property
    def subscribes(self): return ["user.command"]

    async def handle_message(self, msg: Msg2) -> None:
        text = msg.payload.get("text", "").lower()
        intent = self._parse(text)
        if intent:
            await self.publish(f"intent.{intent}", {"raw": text, "source": msg.payload.get("source", "app")})
            self.log(f"Intent: '{intent}' from '{text}'")
        else:
            self.log(f"Unrecognised command: '{text}'")

    def _parse(self, text: str) -> str | None:
        for keyword, intent in KEYWORD_INTENTS.items():
            if keyword in text:
                return intent
        return None

    async def send_command(self, text: str, source: str = "api") -> None:
        """Convenience: inject a command programmatically."""
        await self.bus.publish("user.command", {"text": text, "source": source}, sender=self.name)
