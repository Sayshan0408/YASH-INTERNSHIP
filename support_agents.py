"""
agents/domain/notification_agent.py  — Tier 2
Routes domain-level alerts to users via push / SMS / in-home display.
"""
import logging
from agents.base_agent import BaseAgent
from bus.message_bus import Message

logger = logging.getLogger(__name__)


class NotificationAgent(BaseAgent):
    def __init__(self, bus):
        super().__init__("domain.notifications", bus)
        self.channels = ["push", "log"]  # extend with SMS, email, etc.

    @property
    def publishes(self): return []
    @property
    def subscribes(self): return ["domain.alert.#", "domain.security.state"]

    async def handle_message(self, msg: Message) -> None:
        if "alert" in msg.topic:
            await self._send(msg.payload)

    async def _send(self, payload: dict) -> None:
        # In production: call push service, Twilio, etc.
        logger.warning("[ALERT] %s", payload)
        self.log(f"Sent alert: {payload}")


"""
agents/domain/wellness_agent.py  — Tier 2
Monitors air quality and suggests environmental adjustments.
"""
from agents.base_agent import BaseAgent
from bus.message_bus import Message as Msg2


class WellnessAgent(BaseAgent):
    CO2_THRESHOLD = 1000  # ppm
    PM25_THRESHOLD = 12   # µg/m³

    def __init__(self, bus):
        super().__init__("domain.wellness", bus)

    @property
    def publishes(self): return ["domain.alert.wellness", "cmd.climate.#"]
    @property
    def subscribes(self): return ["sensor.air_quality"]

    async def handle_message(self, msg: Msg2) -> None:
        co2 = msg.payload.get("co2_ppm", 0)
        pm25 = msg.payload.get("pm25", 0)

        if co2 > self.CO2_THRESHOLD:
            await self.publish("domain.alert.wellness", {
                "type": "high_co2", "value": co2, "action": "ventilate"
            })
            await self.publish("cmd.climate.ventilate", {"action": "ventilate"})

        if pm25 > self.PM25_THRESHOLD:
            await self.publish("domain.alert.wellness", {
                "type": "high_pm25", "value": pm25, "action": "purify"
            })


"""
agents/domain/appliance_agent.py  — Tier 2
Manages smart appliances (dishwasher, washer, dryer).
Defers high-draw appliances to off-peak hours.
"""
from agents.base_agent import BaseAgent
from bus.message_bus import Message as Msg3
import asyncio


class ApplianceAgent(BaseAgent):
    PEAK_HOURS = range(17, 21)  # 5–9 PM

    def __init__(self, bus):
        super().__init__("domain.appliances", bus)
        self.deferred: list[dict] = []

    @property
    def publishes(self): return ["domain.appliances.state"]
    @property
    def subscribes(self): return ["cmd.appliance.#", "domain.energy.state"]

    async def handle_message(self, msg: Msg3) -> None:
        if msg.topic.startswith("cmd.appliance"):
            appliance = msg.payload.get("appliance")
            action = msg.payload.get("action")
            from datetime import datetime
            hour = datetime.now().hour
            if action == "start" and hour in self.PEAK_HOURS:
                self.deferred.append(msg.payload)
                self.log(f"Deferred {appliance} start to off-peak")
            else:
                self.log(f"Starting {appliance} now")
