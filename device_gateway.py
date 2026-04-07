"""
gateway/device_gateway.py
Protocol bridge between physical devices and the message bus.
Abstracts Zigbee / Z-Wave / Matter / Wi-Fi / BLE into a unified event format.
"""

import asyncio
import logging
import random
from datetime import datetime, timezone

from bus.message_bus import MessageBus

logger = logging.getLogger(__name__)

# Simulated raw device events (replace with real protocol drivers)
SIMULATED_EVENTS = [
    {"protocol": "zigbee", "device_id": "therm_01", "type": "temperature", "value": 22.4},
    {"protocol": "zigbee", "device_id": "motion_01", "type": "motion", "value": True},
    {"protocol": "zwave",  "device_id": "lock_01",   "type": "lock_state", "value": "locked"},
    {"protocol": "wifi",   "device_id": "cam_01",    "type": "motion_detected", "value": True},
    {"protocol": "matter", "device_id": "bulb_01",   "type": "brightness", "value": 80},
    {"protocol": "wifi",   "device_id": "solar_01",  "type": "generation_w", "value": 3200},
    {"protocol": "wifi",   "device_id": "ev_01",     "type": "soc_pct", "value": 62},
]


class DeviceGateway:
    """
    Receives raw protocol frames, normalises them, and publishes to the bus
    on topic: device.<device_id>.<event_type>

    Also accepts command messages from domain agents on:
    topic: cmd.device.<device_id>
    """

    def __init__(self, bus: MessageBus) -> None:
        self.bus = bus
        self._running = False

    async def start(self) -> None:
        self._running = True
        self.bus.subscribe("cmd.device.#", self._handle_command)
        asyncio.create_task(self._simulate_device_events())
        logger.info("[Gateway] started — simulating device events")

    async def stop(self) -> None:
        self._running = False

    async def _simulate_device_events(self) -> None:
        """Emit simulated device events on a random interval."""
        while self._running:
            event = random.choice(SIMULATED_EVENTS).copy()
            # Add small noise to numeric values
            if isinstance(event["value"], (int, float)):
                event["value"] = round(event["value"] + random.uniform(-0.5, 0.5), 2)

            topic = f"device.{event['device_id']}.{event['type']}"
            payload = {
                "device_id": event["device_id"],
                "protocol": event["protocol"],
                "event_type": event["type"],
                "value": event["value"],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            await self.bus.publish(topic, payload, sender="gateway")
            await asyncio.sleep(random.uniform(1.0, 3.0))

    async def _handle_command(self, msg) -> None:
        """Receive a command from a domain agent and forward to the physical device."""
        parts = msg.topic.split(".")
        device_id = parts[2] if len(parts) > 2 else "unknown"
        logger.info("[Gateway] → device %s: %s", device_id, msg.payload)
        # In production: translate payload to Zigbee/Z-Wave/Matter frame and transmit
