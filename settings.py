"""
config/settings.py
Central configuration for the smart home A2A system.
"""

from dataclasses import dataclass, field


@dataclass
class MQTTConfig:
    host: str = "localhost"
    port: int = 1883
    username: str = ""
    password: str = ""
    tls: bool = False


@dataclass
class DeviceConfig:
    thermostat_id: str = "therm_01"
    motion_sensor_id: str = "motion_01"
    lock_id: str = "lock_01"
    energy_meter_id: str = "meter_01"
    camera_id: str = "cam_01"
    bulb_ids: list = field(default_factory=lambda: ["bulb_01", "bulb_02"])
    air_quality_id: str = "aq_01"
    solar_id: str = "solar_01"
    ev_id: str = "ev_01"
    doorbell_id: str = "bell_01"


@dataclass
class Settings:
    mqtt: MQTTConfig = field(default_factory=MQTTConfig)
    devices: DeviceConfig = field(default_factory=DeviceConfig)
    log_level: str = "INFO"
    simulation_mode: bool = True  # Use in-process bus instead of MQTT broker


settings = Settings()
