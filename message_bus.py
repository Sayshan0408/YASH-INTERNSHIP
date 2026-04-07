"""
bus/message_bus.py
Async pub-sub message bus — the A2A communication backbone.
All agents communicate exclusively through this bus.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine
from uuid import uuid4

logger = logging.getLogger(__name__)


@dataclass
class Message:
    topic: str
    payload: dict[str, Any]
    sender: str
    message_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, data: str) -> "Message":
        return cls(**json.loads(data))


# Type alias for async message handlers
Handler = Callable[[Message], Coroutine[Any, Any, None]]


class MessageBus:
    """
    In-process async pub-sub bus.
    Topics support wildcard segments: 'device.#' matches any device subtopic.
    In production, swap the internal queue for MQTT broker calls.
    """

    def __init__(self) -> None:
        self._subscribers: dict[str, list[Handler]] = {}
        self._queue: asyncio.Queue[Message] = asyncio.Queue()
        self._running = False

    # ------------------------------------------------------------------ #
    # Subscription                                                         #
    # ------------------------------------------------------------------ #

    def subscribe(self, topic: str, handler: Handler) -> None:
        """Subscribe handler to a topic. Use '#' as a trailing wildcard."""
        self._subscribers.setdefault(topic, []).append(handler)
        logger.debug("Subscribed %s to topic '%s'", handler.__qualname__, topic)

    def unsubscribe(self, topic: str, handler: Handler) -> None:
        handlers = self._subscribers.get(topic, [])
        if handler in handlers:
            handlers.remove(handler)

    # ------------------------------------------------------------------ #
    # Publishing                                                           #
    # ------------------------------------------------------------------ #

    async def publish(self, topic: str, payload: dict, sender: str) -> None:
        msg = Message(topic=topic, payload=payload, sender=sender)
        await self._queue.put(msg)
        logger.debug("[BUS] %s → %s", sender, topic)

    # ------------------------------------------------------------------ #
    # Dispatch loop                                                        #
    # ------------------------------------------------------------------ #

    async def start(self) -> None:
        self._running = True
        logger.info("Message bus started")
        while self._running:
            try:
                msg = await asyncio.wait_for(self._queue.get(), timeout=0.1)
                await self._dispatch(msg)
                self._queue.task_done()
            except asyncio.TimeoutError:
                continue

    async def stop(self) -> None:
        self._running = False
        logger.info("Message bus stopped")

    async def _dispatch(self, msg: Message) -> None:
        handlers: list[Handler] = []
        for pattern, subs in self._subscribers.items():
            if self._matches(pattern, msg.topic):
                handlers.extend(subs)

        if not handlers:
            logger.debug("[BUS] No subscribers for topic '%s'", msg.topic)
            return

        await asyncio.gather(*(h(msg) for h in handlers), return_exceptions=True)

    @staticmethod
    def _matches(pattern: str, topic: str) -> bool:
        """Match topic against a pattern where '#' is a trailing wildcard."""
        if pattern == topic:
            return True
        if pattern.endswith("#"):
            prefix = pattern[:-1]
            return topic.startswith(prefix)
        return False
