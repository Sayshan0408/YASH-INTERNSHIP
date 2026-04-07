"""
agents/domain/climate_agent.py  — Tier 2
Aggregates temperature/humidity sensors, controls HVAC.
"""

from agents.base_agent import BaseAgent
from bus.message_bus import Message


class ClimateAgent(BaseAgent):
    def __init__(self, bus):
        super().__init__("domain.climate", bus)
        self.temperatures: dict[str, float] = {}
        self.setpoint = 22.0
        self.mode = "auto"

    @property
    def publishes(self): return ["domain.climate.state", "cmd.device.#"]
    @property
    def subscribes(self): return [
        "sensor.temperature.#",
        "sensor.air_quality",
        "cmd.climate.#",
    ]

    async def handle_message(self, msg: Message) -> None:
        if msg.topic.startswith("sensor.temperature"):
            device_id = msg.payload.get("device_id", "unknown")
            self.temperatures[device_id] = msg.payload["temperature"]
            await self._update_state()

        elif msg.topic.startswith("cmd.climate"):
            action = msg.payload.get("action")
            if action == "set_eco":
                self.setpoint = max(self.setpoint - 2, 18.0)
                self.log(f"Eco mode: setpoint → {self.setpoint}°C")
                await self._push_setpoints()
            elif action == "set_comfort":
                self.setpoint = 22.0
                await self._push_setpoints()
            elif "setpoint" in msg.payload:
                self.setpoint = float(msg.payload["setpoint"])
                await self._push_setpoints()

    async def _update_state(self) -> None:
        if not self.temperatures:
            return
        avg = sum(self.temperatures.values()) / len(self.temperatures)
        await self.publish("domain.climate.state", {
            "avg_temp": round(avg, 2),
            "setpoint": self.setpoint,
            "mode": self.mode,
            "zones": self.temperatures,
        })

    async def _push_setpoints(self) -> None:
        for device_id in self.temperatures:
            await self.publish(f"cmd.device.{device_id}", {
                "setpoint": self.setpoint,
                "mode": self.mode,
            })
