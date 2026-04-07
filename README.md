## 🏠 Smart Home IoT — Agent-to-Agent (A2A) Automation System
A production-ready multi-agent smart home automation framework built on an async pub-sub architecture. 23 independent agents span three tiers — from physical device control to high-level goal orchestration — communicating entirely through a message bus without direct coupling.

## Tier 1 — Device Agents (11 agents)
Thermostat · Motion Sensor · Smart Lock · Security Camera · Smart Bulb · Solar Panel · EV Charger · Doorbell · Air Quality Sensor · Occupancy Sensor · Smart Plug
Each device agent owns a tight local control loop — reads sensor, applies setpoint logic, emits normalised events. Keeps working even when the bus is down (local-first fallback).

## ✨ Key Design Decisions
Async pub-sub, never direct RPC
Agents subscribe to topics rather than calling each other. Adding a new agent never requires touching existing code — it just subscribes to the relevant topics.
Local-first fallback
Device agents hold their last setpoint when the bus is unavailable. Domain agents cache recent device state for resilience.
Agent capability manifests
Each domain agent publishes a machine-readable manifest of accepted commands and emitted events. The orchestrator builds action plans dynamically from these — no hardcoded domain knowledge.
Policy as a synchronous gatekeeper
The only place agents communicate directly. PolicyAgent.is_allowed() blocks before any command leaves the orchestrator — preventing unsafe cross-domain side effects.

## Project Structure
smart-home-a2a/
├── main.py                  # Entry point — boots all agents and runs demo
├── requirements.txt
├── core/
│   ├── message_bus.py       # Async pub-sub bus (topic routing)
│   ├── base_agent.py        # BaseAgent with subscribe/publish helpers
│   └── device_gateway.py    # Simulated hardware event source
├── tier3_orchestration/
│   ├── user_intent.py
│   ├── orchestrator.py
│   ├── policy.py
│   └── learning.py
├── tier2_domain/
│   ├── climate.py
│   ├── security.py
│   ├── lighting.py
│   ├── energy.py
│   ├── presence.py
│   ├── appliances.py
│   ├── wellness.py
│   └── notifications.py
└── tier1_devices/
    ├── thermostat.py
    ├── motion_sensor.py
    ├── smart_lock.py
    ├── camera.py
    ├── smart_bulb.py
    ├── solar_panel.py
    ├── ev_charger.py
    └── doorbell.py
