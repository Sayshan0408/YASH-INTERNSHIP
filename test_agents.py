"""
tests/test_agents.py
Basic unit tests for the A2A smart home system.
"""

import asyncio
import pytest
from bus.message_bus import MessageBus, Message
from agents.device.thermostat import ThermostatAgent
from agents.domain.climate_agent import ClimateAgent
from agents.orchestration.policy_and_learning import PolicyAgent
from agents.orchestration.orchestration_agents import UserIntentAgent, HomeOrchestrator


# ── Helpers ────────────────────────────────────────────────────────────────

def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ── Message bus ────────────────────────────────────────────────────────────

def test_bus_exact_topic():
    bus = MessageBus()
    received = []

    async def handler(msg):
        received.append(msg)

    bus.subscribe("test.topic", handler)
    run(bus.publish("test.topic", {"v": 1}, "tester"))
    run(bus._dispatch(await_queue(bus)))
    assert len(received) == 1


def await_queue(bus):
    return bus._queue.get_nowait()


def test_bus_wildcard_topic():
    bus = MessageBus()
    received = []

    async def handler(msg):
        received.append(msg)

    bus.subscribe("sensor.#", handler)

    async def go():
        await bus.publish("sensor.temperature.room1", {}, "test")
        msg = await bus._queue.get()
        await bus._dispatch(msg)

    run(go())
    assert len(received) == 1


def test_bus_no_match():
    bus = MessageBus()
    received = []

    async def handler(msg):
        received.append(msg)

    bus.subscribe("other.topic", handler)

    async def go():
        await bus.publish("sensor.temperature", {}, "test")
        msg = await bus._queue.get()
        await bus._dispatch(msg)

    run(go())
    assert len(received) == 0


# ── Thermostat device agent ────────────────────────────────────────────────

def test_thermostat_publishes_on_temperature_event():
    bus = MessageBus()
    published = []

    async def capture(msg):
        published.append(msg)

    bus.subscribe("sensor.temperature.#", capture)

    async def go():
        agent = ThermostatAgent(bus)
        await agent.start()
        # Simulate a raw device event
        await bus.publish("device.therm_01.temperature", {"value": 19.0}, "gateway")
        msg = await bus._queue.get()
        await bus._dispatch(msg)
        # Allow the handler to publish
        await asyncio.sleep(0.05)
        if not bus._queue.empty():
            msg2 = await bus._queue.get()
            await bus._dispatch(msg2)

    run(go())
    assert any("temperature" in m.topic for m in published)


# ── Policy agent ───────────────────────────────────────────────────────────

def test_policy_blocks_unlock_when_away():
    bus = MessageBus()
    policy = PolicyAgent(bus)
    policy.context["away"] = True
    allowed, reason = policy.is_allowed("cmd.lock.front_door", {"action": "unlock"})
    assert not allowed
    assert "away" in reason.lower()


def test_policy_allows_unlock_when_home():
    bus = MessageBus()
    policy = PolicyAgent(bus)
    policy.context["away"] = False
    allowed, reason = policy.is_allowed("cmd.lock.front_door", {"action": "unlock"})
    assert allowed


def test_policy_allows_lock_when_away():
    bus = MessageBus()
    policy = PolicyAgent(bus)
    policy.context["away"] = True
    allowed, _ = policy.is_allowed("cmd.lock.front_door", {"action": "lock"})
    assert allowed


# ── User intent parsing ────────────────────────────────────────────────────

def test_user_intent_parses_morning():
    bus = MessageBus()
    agent = UserIntentAgent(bus)
    assert agent._parse("good morning") == "good_morning"


def test_user_intent_parses_leaving():
    bus = MessageBus()
    agent = UserIntentAgent(bus)
    assert agent._parse("I am leaving home now") == "leaving_home"


def test_user_intent_unknown_returns_none():
    bus = MessageBus()
    agent = UserIntentAgent(bus)
    assert agent._parse("play some jazz please") is None


# ── Orchestrator ───────────────────────────────────────────────────────────

def test_orchestrator_executes_known_plan():
    bus = MessageBus()
    published = []

    async def capture(msg):
        published.append(msg)

    bus.subscribe("cmd.#", capture)

    async def go():
        policy = PolicyAgent(bus)
        orch = HomeOrchestrator(bus, policy_agent=policy)
        await orch.start()
        await orch._execute_plan("good_morning", {})
        await asyncio.sleep(0.05)
        # Drain the queue
        while not bus._queue.empty():
            msg = await bus._queue.get()
            await bus._dispatch(msg)

    run(go())
    topics = [m.topic for m in published]
    assert any("lighting" in t for t in topics)
    assert any("climate" in t for t in topics)


if __name__ == "__main__":
    import sys
    # Run tests manually if pytest not available
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
