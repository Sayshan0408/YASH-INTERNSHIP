"""
agents/base_agent.py
Abstract base for all agents across all three tiers.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any

from bus.message_bus import Message, MessageBus

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Every agent has:
      - a unique name
      - a reference to the shared message bus
      - a capability manifest (topics it publishes and subscribes to)
      - a start/stop lifecycle
    """

    def __init__(self, name: str, bus: MessageBus) -> None:
        self.name = name
        self.bus = bus
        self._running = False
        self._tasks: list[asyncio.Task] = []

    # ------------------------------------------------------------------ #
    # Capability manifest                                                  #
    # ------------------------------------------------------------------ #

    @property
    def publishes(self) -> list[str]:
        """Topics this agent publishes. Override in subclass."""
        return []

    @property
    def subscribes(self) -> list[str]:
        """Topics this agent subscribes to. Override in subclass."""
        return []

    def capability_manifest(self) -> dict[str, Any]:
        return {
            "agent": self.name,
            "publishes": self.publishes,
            "subscribes": self.subscribes,
        }

    # ------------------------------------------------------------------ #
    # Lifecycle                                                            #
    # ------------------------------------------------------------------ #

    async def start(self) -> None:
        self._running = True
        self._register_subscriptions()
        await self.on_start()
        logger.info("[%s] started", self.name)

    async def stop(self) -> None:
        self._running = False
        for task in self._tasks:
            task.cancel()
        await self.on_stop()
        logger.info("[%s] stopped", self.name)

    async def on_start(self) -> None:
        """Override for agent-specific startup logic."""

    async def on_stop(self) -> None:
        """Override for agent-specific teardown logic."""

    def _register_subscriptions(self) -> None:
        for topic in self.subscribes:
            self.bus.subscribe(topic, self.handle_message)

    # ------------------------------------------------------------------ #
    # Messaging                                                            #
    # ------------------------------------------------------------------ #

    async def publish(self, topic: str, payload: dict) -> None:
        await self.bus.publish(topic, payload, sender=self.name)

    @abstractmethod
    async def handle_message(self, msg: Message) -> None:
        """Handle an inbound message from the bus."""

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _spawn(self, coro) -> asyncio.Task:
        task = asyncio.create_task(coro)
        self._tasks.append(task)
        return task

    def log(self, msg: str) -> None:
        logger.info("[%s] %s", self.name, msg)
