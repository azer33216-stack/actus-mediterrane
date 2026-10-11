"""Auto-discover all workflow classes in the `workflows` package and start a worker."""
# ruff: noqa: E402

import asyncio
import contextlib
import importlib
import inspect
import pkgutil
import signal
import sys

from dotenv import load_dotenv

load_dotenv(override=True)

import mistralai.workflows as mistralai_workflows
from mistralai.workflows.core.definition.workflow_definition import (
    get_workflow_definition,
)


def discover_workflows() -> list[type]:
    """Scan the `workflows` package and return all workflow classes."""
    discovered = []
    package = importlib.import_module("workflows")

    for _, modname, ispkg in pkgutil.iter_modules(package.__path__, prefix="workflows."):
        if ispkg:
            continue
        module = importlib.import_module(modname)
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if hasattr(obj, "__workflows_workflow_def"):
                discovered.append(obj)

    return discovered


async def main() -> None:
    discovered = discover_workflows()

    if not discovered:
        print("No workflows discovered in the `workflows` package.")
        sys.exit(1)

    names = [get_workflow_definition(wf).name for wf in discovered]
    print(f"Discovered {len(discovered)} workflow(s): {', '.join(names)}")

    worker = asyncio.create_task(mistralai_workflows.run_worker(discovered))

    # As PID 1 in a container, Python discards SIGTERM unless a handler is installed,
    # so the worker would be SIGKILLed on redeploy. Cancelling the task is what makes
    # Temporal drain in-flight activities before exiting.
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):  # Windows event loops
            loop.add_signal_handler(sig, worker.cancel)

    # A CancelledError here is the expected end of a clean shutdown; any other
    # exception is a real worker failure and propagates.
    with contextlib.suppress(asyncio.CancelledError):
        await worker


if __name__ == "__main__":
    asyncio.run(main())
