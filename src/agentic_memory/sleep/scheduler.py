"""Background scheduler for sleep-time memory consolidation.

Uses asyncio tasks (not OS threads or Celery) for single-process deployment.
Runs periodic consolidation cycles for each registered agent.
"""

from __future__ import annotations

import asyncio
import contextlib

from agentic_memory.schemas.consolidation import ConsolidationResult
from agentic_memory.sleep.consolidator import Consolidator
from agentic_memory.utils.logging import get_logger

logger = get_logger("scheduler")


class SleepScheduler:
    """Background scheduler for memory consolidation.

    Registers agents and runs a periodic consolidation loop for each one.
    Designed for single-process deployment. For multi-process production,
    subclass and dispatch to external task queues.
    """

    def __init__(
        self,
        consolidator: Consolidator,
        interval_seconds: int = 300,
    ) -> None:
        self._consolidator = consolidator
        self._interval = interval_seconds
        self._tasks: dict[str, asyncio.Task] = {}
        self._running = False

    async def start(self) -> None:
        """Enable the scheduler (tasks will be created for registered agents)."""
        self._running = True
        logger.info("sleep scheduler started (interval=%ds)", self._interval)

    async def stop(self) -> None:
        """Cancel all running consolidation tasks."""
        self._running = False
        for agent_id, task in self._tasks.items():
            task.cancel()
            logger.debug("cancelled consolidation task for agent '%s'", agent_id)
        # Wait for cancellation to propagate.
        if self._tasks:
            await asyncio.gather(*self._tasks.values(), return_exceptions=True)
        self._tasks.clear()
        logger.info("sleep scheduler stopped")

    async def register_agent(self, agent_id: str) -> None:
        """Register an agent for periodic consolidation."""
        if agent_id in self._tasks:
            logger.warning("agent '%s' already registered", agent_id)
            return
        task = asyncio.create_task(self._consolidation_loop(agent_id))
        self._tasks[agent_id] = task
        logger.info("registered agent '%s' for consolidation", agent_id)

    async def unregister_agent(self, agent_id: str) -> None:
        """Stop consolidation for an agent."""
        task = self._tasks.pop(agent_id, None)
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            logger.info("unregistered agent '%s' from consolidation", agent_id)

    async def run_once(self, agent_id: str) -> ConsolidationResult:
        """Manually trigger a single consolidation cycle."""
        return await self._consolidator.consolidate(agent_id)

    async def _consolidation_loop(self, agent_id: str) -> None:
        """Main loop: sleep then consolidate, repeat until cancelled."""
        try:
            while self._running:
                await asyncio.sleep(self._interval)
                if not self._running:
                    break
                try:
                    result = await self._consolidator.consolidate(agent_id)
                    logger.info(
                        "consolidated agent '%s': scored=%d, pruned=%d, summaries=%d",
                        agent_id, result.nodes_scored, result.nodes_pruned,
                        result.summaries_created,
                    )
                except Exception as exc:
                    logger.error(
                        "consolidation failed for agent '%s': %s", agent_id, exc
                    )
        except asyncio.CancelledError:
            pass
