"""
agents/device/smart_lock.py  — Tier 1
"""

from agents.base_agent import BaseAgent
from bus.message_bus import Message


class SmartLockAgent(BaseAgent):
    def __init__(self, bus, device_id: str = "lock_01", location: str = "front_door"):
        super().__init__(f"lock.{device_id}", bus)
        self.device_id = device_id
        self.location = location
        self.state = "locked"
        self.audit_log: list[dict] = []

    @property
    def publishes(self):
        return [f"sensor.lock.{self.location}"]

    @property
    def subscribes(self):
        return [f"device.{self.device_id}.lock_state", f"cmd.lock.{self.device_id}"]

    async def handle_message(self, msg: Message) -> None:
        if "lock_state" in msg.topic:
            self.state = msg.payload["value"]
            await self._publish_state()
        elif msg.topic.startswith("cmd.lock"):
            action = msg.payload.get("action")
            user = msg.payload.get("user", "system")
            if action in ("lock", "unlock"):
                self.state = "locked" if action == "lock" else "unlocked"
                entry = {"action": action, "user": user, "state": self.state}
                self.audit_log.append(entry)
                await self._publish_state(user=user)

    async def _publish_state(self, user: str = "sensor") -> None:
        await self.publish(f"sensor.lock.{self.location}", {
            "device_id": self.device_id,
            "location": self.location,
            "state": self.state,
            "user": user,
        })
        self.log(f"{self.location} is {self.state}")
