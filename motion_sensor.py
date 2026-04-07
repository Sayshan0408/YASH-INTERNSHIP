"""
agents/device/motion_sensor.py  — Tier 1 device agent
"""

import asyncio
from agents.base_agent import BaseAgent
from bus.message_bus import Message


class MotionSensorAgent(BaseAgent):
    def __init__(self, bus, device_id: str = "motion_01", zone: str = "living_room"):
        super().__init__(f"motion.{device_id}", bus)
        self.device_id = device_id
        self.zone = zone
        self.motion_detected = False
        self._clear_task = None

    @property
    def publishes(self):
        return [f"sensor.motion.{self.zone}"]

    @property
    def subscribes(self):
        return [f"device.{self.device_id}.motion"]

    async def handle_message(self, msg: Message) -> None:
        detected = bool(msg.payload.get("value", False))
        if detected and not self.motion_detected:
            self.motion_detected = True
            await self._publish_motion(True)
        if detected:
            # Reset clear timer on each positive reading
            if self._clear_task:
                self._clear_task.cancel()
            self._clear_task = self._spawn(self._auto_clear())

    async def _auto_clear(self) -> None:
        await asyncio.sleep(30)  # 30s timeout
        self.motion_detected = False
        await self._publish_motion(False)

    async def _publish_motion(self, active: bool) -> None:
        await self.publish(f"sensor.motion.{self.zone}", {
            "device_id": self.device_id,
            "zone": self.zone,
            "motion": active,
        })
        self.log(f"zone={self.zone} motion={'DETECTED' if active else 'cleared'}")
