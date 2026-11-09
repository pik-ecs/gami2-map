"""Reads the per-task settings from tasks_settings.py in the project root (kept outside src so it's easy
to find and edit; see that file for what each entry holds)."""

import importlib.util

from gami2_map.paths import TASKS_PATH


def _load_tasks() -> dict:
    spec = importlib.util.spec_from_file_location("project_tasks", TASKS_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TASKS


def task_config(task: str) -> dict:
    """The TASKS entry for `task`, read fresh from tasks_settings.py."""
    tasks = _load_tasks()
    if task not in tasks:
        raise KeyError(f"No entry for task {task!r} in tasks_settings.py; known: {list(tasks)}")
    return tasks[task]


def data_config(task: str) -> dict:
    """The task's "data" entry: what build-labels reads (files, columns, relabels, filter,
    splits)."""
    return task_config(task)["data"]


def model_settings(task: str, model_key: str) -> dict:
    """The task's "models" entry for `model_key` ({} if none): trials, params, search."""
    return task_config(task).get("models", {}).get(model_key, {})
