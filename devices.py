"""
agents/device/energy_meter.py  — Tier 1
"""
from agents.base_agent import BaseAgent
from bus.message_bus import Message


class EnergyMeterAgent(BaseAgent):
    def __init__(self, bus, device_id: str = "meter_01"):
        super().__init__(f"energy_meter.{device_id}", bus)
        self.device_id = device_id
        self.power_w: float = 0.0
        self.daily_kwh: float = 0.0

    @property
    def publishes(self): return ["sensor.energy.consumption"]
    @property
    def subscribes(self): return [f"device.{self.device_id}.power"]

    async def handle_message(self, msg: Message) -> None:
        self.power_w = msg.payload.get("value", 0.0)
        self.daily_kwh += self.power_w / 3_600_000
        await self.publish("sensor.energy.consumption", {
            "power_w": self.power_w, "daily_kwh": round(self.daily_kwh, 4)
        })


"""
agents/device/camera.py  — Tier 1
"""
from agents.base_agent import BaseAgent
from bus.message_bus import Message as Msg2


class CameraAgent(BaseAgent):
    def __init__(self, bus, device_id: str = "cam_01", location: str = "front"):
        super().__init__(f"camera.{device_id}", bus)
        self.device_id = device_id
        self.location = location

    @property
    def publishes(self): return [f"sensor.camera.{self.location}"]
    @property
    def subscribes(self): return [f"device.{self.device_id}.motion_detected"]

    async def handle_message(self, msg: Msg2) -> None:
        await self.publish(f"sensor.camera.{self.location}", {
            "device_id": self.device_id,
            "location": self.location,
            "event": "motion",
            "value": msg.payload.get("value"),
        })
        self.log(f"motion event at {self.location}")


"""
agents/device/smart_bulb.py  — Tier 1
"""
from agents.base_agent import BaseAgent
from bus.message_bus import Message as Msg3


class SmartBulbAgent(BaseAgent):
    def __init__(self, bus, device_id: str = "bulb_01", room: str = "living_room"):
        super().__init__(f"bulb.{device_id}", bus)
        self.device_id = device_id
        self.room = room
        self.brightness = 100
        self.color_temp = 4000
        self.on = True

    @property
    def publishes(self): return [f"sensor.bulb.{self.room}"]
    @property
    def subscribes(self): return [
        f"device.{self.device_id}.brightness",
        f"cmd.bulb.{self.device_id}",
    ]

    async def handle_message(self, msg: Msg3) -> None:
        if "brightness" in msg.topic:
            self.brightness = msg.payload.get("value", self.brightness)
        elif msg.topic.startswith("cmd.bulb"):
            self.brightness = msg.payload.get("brightness", self.brightness)
            self.color_temp = msg.payload.get("color_temp", self.color_temp)
            self.on = msg.payload.get("on", self.on)
        await self.publish(f"sensor.bulb.{self.room}", {
            "device_id": self.device_id,
            "room": self.room,
            "on": self.on,
            "brightness": self.brightness,
            "color_temp": self.color_temp,
        })


"""
agents/device/air_quality.py  — Tier 1
"""
from agents.base_agent import BaseAgent
from bus.message_bus import Message as Msg4


class AirQualityAgent(BaseAgent):
    def __init__(self, bus, device_id: str = "aq_01"):
        super().__init__(f"air_quality.{device_id}", bus)
        self.device_id = device_id
        self.co2_ppm = 400.0
        self.voc_ppb = 50.0
        self.pm25 = 5.0

    @property
    def publishes(self): return ["sensor.air_quality"]
    @property
    def subscribes(self): return [f"device.{self.device_id}.#"]

    async def handle_message(self, msg: Msg4) -> None:
        etype = msg.payload.get("event_type", "")
        val = msg.payload.get("value", 0)
        if "co2" in etype:  self.co2_ppm = val
        elif "voc" in etype: self.voc_ppb = val
        elif "pm" in etype:  self.pm25 = val
        await self.publish("sensor.air_quality", {
            "co2_ppm": self.co2_ppm, "voc_ppb": self.voc_ppb, "pm25": self.pm25,
            "quality": "good" if self.co2_ppm < 1000 else "poor",
        })


"""
agents/device/solar_inverter.py  — Tier 1
"""
from agents.base_agent import BaseAgent
from bus.message_bus import Message as Msg5


class SolarInverterAgent(BaseAgent):
    def __init__(self, bus, device_id: str = "solar_01"):
        super().__init__(f"solar.{device_id}", bus)
        self.device_id = device_id
        self.generation_w = 0.0

    @property
    def publishes(self): return ["sensor.energy.solar"]
    @property
    def subscribes(self): return [f"device.{self.device_id}.generation_w"]

    async def handle_message(self, msg: Msg5) -> None:
        self.generation_w = msg.payload.get("value", 0.0)
        await self.publish("sensor.energy.solar", {
            "generation_w": self.generation_w,
            "status": "generating" if self.generation_w > 0 else "idle",
        })


"""
agents/device/ev_charger.py  — Tier 1
"""
from agents.base_agent import BaseAgent
from bus.message_bus import Message as Msg6


class EVChargerAgent(BaseAgent):
    def __init__(self, bus, device_id: str = "ev_01"):
        super().__init__(f"ev.{device_id}", bus)
        self.device_id = device_id
        self.soc_pct = 0.0
        self.charging = False
        self.max_charge_rate_kw = 7.4

    @property
    def publishes(self): return ["sensor.ev.state"]
    @property
    def subscribes(self): return [
        f"device.{self.device_id}.soc_pct",
        f"cmd.ev.{self.device_id}",
    ]

    async def handle_message(self, msg: Msg6) -> None:
        if "soc_pct" in msg.topic:
            self.soc_pct = msg.payload.get("value", self.soc_pct)
        elif msg.topic.startswith("cmd.ev"):
            self.charging = msg.payload.get("charging", self.charging)
            self.max_charge_rate_kw = msg.payload.get("rate_kw", self.max_charge_rate_kw)
        await self.publish("sensor.ev.state", {
            "soc_pct": self.soc_pct,
            "charging": self.charging,
            "max_rate_kw": self.max_charge_rate_kw,
        })


"""
agents/device/doorbell.py  — Tier 1
"""
from agents.base_agent import BaseAgent
from bus.message_bus import Message as Msg7


class DoorbellAgent(BaseAgent):
    def __init__(self, bus, device_id: str = "bell_01"):
        super().__init__(f"doorbell.{device_id}", bus)
        self.device_id = device_id

    @property
    def publishes(self): return ["sensor.doorbell.ring"]
    @property
    def subscribes(self): return [f"device.{self.device_id}.#"]

    async def handle_message(self, msg: Msg7) -> None:
        etype = msg.payload.get("event_type", "")
        await self.publish("sensor.doorbell.ring", {
            "device_id": self.device_id,
            "event": etype,
            "value": msg.payload.get("value"),
        })
        self.log(f"event: {etype}")
