"""
agents/orchestration/policy_agent.py  — Tier 3
Enforces hard constraints. Other agents MUST check policy before acting.
"""
import logging
from agents.base_agent import BaseAgent
from bus.message_bus import Message

logger = logging.getLogger(__name__)

# Each rule: {"topic": pattern, "condition": fn(payload) -> bool, "block_reason": str}
DEFAULT_RULES = [
    {
        "id": "no_unlock_away",
        "description": "Never unlock doors when away mode is active",
        "trigger_topic": "cmd.lock.#",
        "blocks_if": lambda p, ctx: p.get("action") == "unlock" and ctx.get("away", False),
        "reason": "Unlock blocked: home is in away mode",
    },
    {
        "id": "no_heat_windows_open",
        "description": "Do not run heat when windows are open",
        "trigger_topic": "cmd.climate.#",
        "blocks_if": lambda p, ctx: p.get("action") == "heat" and ctx.get("windows_open", False),
        "reason": "Heat blocked: windows are open",
    },
]


class PolicyAgent(BaseAgent):
    def __init__(self, bus):
        super().__init__("orchestration.policy", bus)
        self.rules = DEFAULT_RULES
        self.context: dict = {"away": False, "windows_open": False}

    @property
    def publishes(self): return ["policy.decision", "policy.violation"]
    @property
    def subscribes(self): return [
        "presence.away", "presence.home",
        "policy.check.#",
    ]

    async def handle_message(self, msg: Message) -> None:
        if msg.topic == "presence.away":
            self.context["away"] = True
        elif msg.topic == "presence.home":
            self.context["away"] = False
        elif msg.topic.startswith("policy.check"):
            await self._evaluate(msg)

    async def _evaluate(self, msg: Message) -> None:
        original_topic = msg.payload.get("topic", "")
        payload = msg.payload.get("payload", {})

        for rule in self.rules:
            pattern = rule["trigger_topic"].rstrip("#")
            if original_topic.startswith(pattern):
                if rule["blocks_if"](payload, self.context):
                    await self.publish("policy.violation", {
                        "rule_id": rule["id"],
                        "reason": rule["reason"],
                        "blocked_topic": original_topic,
                    })
                    self.log(f"BLOCKED [{rule['id']}]: {rule['reason']}")
                    return

        await self.publish("policy.decision", {
            "allowed": True,
            "topic": original_topic,
            "payload": payload,
        })

    def is_allowed(self, topic: str, payload: dict) -> tuple[bool, str]:
        """Synchronous check for use by orchestrator before dispatching."""
        for rule in self.rules:
            pattern = rule["trigger_topic"].rstrip("#")
            if topic.startswith(pattern) and rule["blocks_if"](payload, self.context):
                return False, rule["reason"]
        return True, ""


"""
agents/orchestration/learning_agent.py  — Tier 3
Observes patterns and proposes automation routines.
"""
from collections import defaultdict
from agents.base_agent import BaseAgent
from bus.message_bus import Message as Msg2


class LearningAgent(BaseAgent):
    def __init__(self, bus):
        super().__init__("orchestration.learning", bus)
        self._event_counts: dict = defaultdict(int)
        self._proposed: set[str] = set()

    @property
    def publishes(self): return ["learning.routine.proposed"]
    @property
    def subscribes(self): return ["domain.presence.state", "domain.lighting.state", "domain.climate.state"]

    async def handle_message(self, msg: Msg2) -> None:
        key = f"{msg.topic}:{str(msg.payload)[:40]}"
        self._event_counts[key] += 1

        # Propose a routine after observing the same pattern 5+ times
        for k, count in self._event_counts.items():
            if count >= 5 and k not in self._proposed:
                self._proposed.add(k)
                await self.publish("learning.routine.proposed", {
                    "pattern": k,
                    "occurrences": count,
                    "suggestion": f"Automate: {k.split(':')[0]}",
                })
                self.log(f"Proposed routine for: {k}")
