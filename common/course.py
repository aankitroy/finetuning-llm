"""Load course module manifests (``modules/m*/module.toml``)."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

_BETTER = {"lower", "higher"}
STEP_KINDS = {"lesson", "exercises", "answers", "train", "checkpoint"}


@dataclass(frozen=True)
class Module:
    id: str
    title: str
    part: str
    order: int
    summary: str
    est_minutes: int
    next_start: str
    next_finish: str
    next_why: str
    artifacts: list[str]
    metrics: list[dict]
    path: Path
    steps: list[dict] = field(default_factory=list)


def _parse(data: dict, path: Path) -> Module:
    nxt = data["next"]
    if not isinstance(nxt, dict):
        raise ValueError("[next] must be a table")
    raw_metrics = data.get("metrics", [])
    if not isinstance(raw_metrics, list) or not all(isinstance(m, dict) for m in raw_metrics):
        raise ValueError("metrics must be written as [[metrics]] tables")
    artifacts = data.get("artifacts", {})
    if not isinstance(artifacts, dict) or not isinstance(artifacts.get("any", []), list):
        raise ValueError('artifacts must be an [artifacts] table with any = ["..."]')
    raw_steps = data.get("steps", [])
    if not isinstance(raw_steps, list) or not all(isinstance(st, dict) for st in raw_steps):
        raise ValueError("steps must be written as [[steps]] tables")
    steps = []
    for st in raw_steps:
        if st.get("kind") not in STEP_KINDS:
            raise ValueError(f"step kind {st.get('kind')!r} must be one of {sorted(STEP_KINDS)}")
        steps.append(
            {
                "kind": st["kind"],
                "title": str(st["title"]),
                "detail": str(st.get("detail", "")),
                "command": str(st["command"]) if st.get("command") else None,
                "file": str(st["file"]) if st.get("file") else None,
                "metric": str(st["metric"]) if st.get("metric") else None,
            }
        )
    metrics = []
    for m in raw_metrics:
        if m.get("better") not in _BETTER:
            raise ValueError(f"metric {m.get('key')!r}: better must be 'lower' or 'higher'")
        metrics.append({"key": str(m["key"]), "label": str(m["label"]), "better": m["better"]})
    return Module(
        id=str(data["id"]),
        title=str(data["title"]),
        part=str(data["part"]),
        order=int(data["order"]),
        summary=str(data["summary"]),
        est_minutes=int(data["est_minutes"]),
        next_start=str(nxt["start"]),
        next_finish=str(nxt["finish"]),
        next_why=str(nxt["why"]),
        artifacts=[str(p) for p in artifacts.get("any", [])],
        metrics=metrics,
        path=path.parent,
        steps=steps,
    )


def load_modules(root: Path) -> tuple[list[Module], list[str]]:
    """Return valid modules sorted by ``order`` plus human-readable warnings."""
    modules: list[Module] = []
    warnings: list[str] = []
    for manifest in sorted(Path(root).glob("modules/m*/module.toml")):
        name = manifest.parent.name
        try:
            with manifest.open("rb") as f:
                modules.append(_parse(tomllib.load(f), manifest))
        except tomllib.TOMLDecodeError as e:
            warnings.append(f"{name}: invalid module.toml ({e})")
        except KeyError as e:
            warnings.append(f"{name}: module.toml missing required key {e}")
        except (ValueError, TypeError) as e:
            warnings.append(f"{name}: module.toml {e}")
    modules.sort(key=lambda m: m.order)
    return modules, warnings
