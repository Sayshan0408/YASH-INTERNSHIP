"""
agents/device/thermostat.py  — Tier 1 device agent
Local control loop for a smart thermostat.
"""

from agents.base_agent import BaseAgent
from bus.message_bus import Message


class ThermostatAgent(BaseAgent):
    def __init__(self, bus, device_id: str = "therm_01"):
        super().__init__(f"thermostat.{device_id}", bus)
        self.device_id = device_id
        self.temperature: float = 21.0
        self.setpoint: float = 22.0
        self.mode: str = "auto"  # auto | heat | cool | off

    @property
    def publishes(self):
        return [f"sensor.temperature.{self.device_id}"]

    @property
    def subscribes(self):
        return [
            f"device.{self.device_id}.temperature",
            f"cmd.thermostat.{self.device_id}",
        ]

    async def handle_message(self, msg: Message) -> None:
        if "temperature" in msg.topic:
            self.temperature = msg.payload["value"]
            await self._evaluate_control_loop()
        elif msg.topic.startswith("cmd.thermostat"):
            await self._apply_command(msg.payload)

    async def _evaluate_control_loop(self) -> None:
        delta = self.setpoint - self.temperature
        action = "none"
        if abs(delta) > 0.5:
            action = "heat" if delta > 0 else "cool"

        await self.publish(f"sensor.temperature.{self.device_id}", {
            "device_id": self.device_id,
            "temperature": self.temperature,
            "setpoint": self.setpoint,
            "mode": self.mode,
            "action": action,
        })
        self.log(f"temp={self.temperature}°C setpoint={self.setpoint}°C action={action}")

    async def _apply_command(self, cmd: dict) -> None:
        if "setpoint" in cmd:
            self.setpoint = float(cmd["setpoint"])
            self.log(f"setpoint updated to {self.setpoint}°C")
        if "mode" in cmd:
            self.mode = cmd["mode"]
            self.log(f"mode set to {self.mode}")
