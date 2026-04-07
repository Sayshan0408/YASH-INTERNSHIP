"""
agents/domain/security_agent.py  — Tier 2
"""
from agents.base_agent import BaseAgent
from bus.message_bus import Message


class SecurityAgent(BaseAgent):
    def __init__(self, bus):
        super().__init__("domain.security", bus)
        self.armed = False
        self.locks: dict[str, str] = {}
        self.camera_events: list[dict] = []

    @property
    def publishes(self): return ["domain.security.state", "domain.alert.security"]
    @property
    def subscribes(self): return ["sensor.lock.#", "sensor.camera.#", "cmd.security.#"]

    async def handle_message(self, msg: Message) -> None:
        if msg.topic.startswith("sensor.lock"):
            loc = msg.payload.get("location", "?")
            self.locks[loc] = msg.payload.get("state", "unknown")
            await self._publish_state()

        elif msg.topic.startswith("sensor.camera") and msg.payload.get("event") == "motion":
            self.camera_events.append(msg.payload)
            if self.armed:
                await self.publish("domain.alert.security", {
                    "alert": "motion_while_armed",
                    "source": msg.payload.get("location"),
                })

        elif msg.topic.startswith("cmd.security"):
            action = msg.payload.get("action")
            if action == "arm":
                self.armed = True
                await self._lock_all()
            elif action == "disarm":
                self.armed = False
            await self._publish_state()

    async def _lock_all(self) -> None:
        for loc in self.locks:
            await self.publish(f"cmd.lock.{loc}", {"action": "lock", "user": "security_agent"})

    async def _publish_state(self) -> None:
        await self.publish("domain.security.state", {
            "armed": self.armed,
            "locks": self.locks,
        })


"""
agents/domain/lighting_agent.py  — Tier 2
"""
from agents.base_agent import BaseAgent
from bus.message_bus import Message as Msg2

SCENES = {
    "evening":  {"brightness": 60, "color_temp": 2700},
    "morning":  {"brightness": 80, "color_temp": 4000},
    "movie":    {"brightness": 20, "color_temp": 2200},
    "off":      {"brightness": 0,  "on": False},
    "full":     {"brightness": 100,"color_temp": 5000},
}


class LightingAgent(BaseAgent):
    def __init__(self, bus):
        super().__init__("domain.lighting", bus)
        self.scene = "full"
        self.bulbs: list[str] = []

    @property
    def publishes(self): return ["domain.lighting.state", "cmd.bulb.#"]
    @property
    def subscribes(self): return ["sensor.bulb.#", "cmd.lighting.#", "sensor.motion.#"]

    async def handle_message(self, msg: Msg2) -> None:
        if msg.topic.startswith("sensor.bulb"):
            device_id = msg.payload.get("device_id")
            if device_id and device_id not in self.bulbs:
                self.bulbs.append(device_id)

        elif msg.topic.startswith("cmd.lighting"):
            scene = msg.payload.get("scene")
            if scene in SCENES:
                self.scene = scene
                await self._apply_scene(scene)

        elif msg.topic.startswith("sensor.motion"):
            if msg.payload.get("motion") and self.scene == "off":
                await self._apply_scene("evening")

    async def _apply_scene(self, scene: str) -> None:
        config = SCENES.get(scene, {})
        for bulb_id in self.bulbs:
            await self.publish(f"cmd.bulb.{bulb_id}", config)
        await self.publish("domain.lighting.state", {"scene": scene, "bulbs": len(self.bulbs)})
        self.log(f"scene → {scene}")


"""
agents/domain/energy_agent.py  — Tier 2
Monitors total consumption, solar generation, EV. Negotiates with climate agent.
"""
from agents.base_agent import BaseAgent
from bus.message_bus import Message as Msg3


class EnergyAgent(BaseAgent):
    PEAK_THRESHOLD_W = 5000.0

    def __init__(self, bus):
        super().__init__("domain.energy", bus)
        self.consumption_w = 0.0
        self.solar_w = 0.0
        self.ev_soc = 0.0

    @property
    def publishes(self): return ["domain.energy.state", "cmd.climate.#", "cmd.ev.#"]
    @property
    def subscribes(self): return [
        "sensor.energy.consumption",
        "sensor.energy.solar",
        "sensor.ev.state",
    ]

    async def handle_message(self, msg: Msg3) -> None:
        if "consumption" in msg.topic:
            self.consumption_w = msg.payload.get("power_w", 0)
        elif "solar" in msg.topic:
            self.solar_w = msg.payload.get("generation_w", 0)
        elif "ev" in msg.topic:
            self.ev_soc = msg.payload.get("soc_pct", self.ev_soc)

        net = self.consumption_w - self.solar_w
        if net > self.PEAK_THRESHOLD_W:
            # Ask climate to back off
            await self.publish("cmd.climate.eco", {"action": "set_eco"})
            # Throttle EV charging
            await self.publish("cmd.ev.ev_01", {"charging": True, "rate_kw": 3.7})

        await self.publish("domain.energy.state", {
            "consumption_w": self.consumption_w,
            "solar_w": self.solar_w,
            "net_w": round(net, 2),
            "ev_soc_pct": self.ev_soc,
        })


"""
agents/domain/presence_agent.py  — Tier 2
Fuses motion data to determine whole-home occupancy.
"""
from agents.base_agent import BaseAgent
from bus.message_bus import Message as Msg4
import asyncio


class PresenceAgent(BaseAgent):
    AWAY_TIMEOUT = 300  # seconds of no motion → away

    def __init__(self, bus):
        super().__init__("domain.presence", bus)
        self.occupied = True
        self.active_zones: set[str] = set()
        self._away_task = None

    @property
    def publishes(self): return ["domain.presence.state", "presence.away", "presence.home"]
    @property
    def subscribes(self): return ["sensor.motion.#"]

    async def handle_message(self, msg: Msg4) -> None:
        zone = msg.payload.get("zone", "unknown")
        motion = msg.payload.get("motion", False)

        if motion:
            self.active_zones.add(zone)
            if not self.occupied:
                self.occupied = True
                await self.publish("presence.home", {"zones": list(self.active_zones)})
                self.log("occupancy: HOME")
            if self._away_task:
                self._away_task.cancel()
            self._away_task = self._spawn(self._schedule_away())
        else:
            self.active_zones.discard(zone)

        await self.publish("domain.presence.state", {
            "occupied": self.occupied,
            "active_zones": list(self.active_zones),
        })

    async def _schedule_away(self) -> None:
        await asyncio.sleep(self.AWAY_TIMEOUT)
        self.occupied = False
        self.active_zones.clear()
        await self.publish("presence.away", {"reason": "no_motion_timeout"})
        self.log("occupancy: AWAY")
