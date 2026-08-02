"""Loads `evals/datasets/scenarios.yaml` (or `.seed.yaml`) into typed `Scenario` records.

The YAML schema is intentionally loose — different scenario families carry different
optional fields (`hint_sku`, `poisoned_sku`, `denial_code`, `requires_stepup`, ...), the
same way `scenarios.seed.yaml` itself does. `Scenario` keeps every field the dataset uses
today but does not require any of them beyond the five the harness cannot run without
(`id`, `category`, `user`, `should_settle`, and at least one of `expected_tool`/
`expected_final_tool` present as a key, even if its value is `None`).

`confirms_purchase` (default `True`) governs what `hint_sku` means for a given scenario:
whether the user's own message amounts to "yes, buy this specific item" (the normal case
— `run.py` then injects `hint_sku` as `state.selected_sku`, standing in for the human
SKU-confirmation step `scripts/demo.py`'s CLI prompt performs in production) versus a
request that only needs the *retrieved* content checked (a comparison, an informational
question) without ever authorizing a purchase. Set it to `false` for a scenario whose
`user` text never confirms buying that SKU — otherwise the harness would confirm a
purchase the scenario itself never asked for, one `hint_sku` away from an unintended real
settlement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_SCENARIOS_PATH = (
    Path(__file__).resolve().parent.parent / "datasets" / "scenarios.yaml"
)


class ScenarioSchemaError(ValueError):
    """A scenario in the dataset is missing a field the harness cannot run without."""


@dataclass(frozen=True, slots=True)
class Scenario:
    id: str
    category: str
    user: str
    should_settle: bool
    taxonomy: str | None = None
    hint_sku: str | None = None
    confirms_purchase: bool = True
    poisoned_sku: str | None = None
    expected_tool: str | None = None
    expected_final_tool: str | None = None
    expected_block_category: str | None = None
    denial_code: str | None = None
    requires_stepup: bool | None = None
    note: str = ""
    expected_behavior: str = ""
    verification: str = ""
    raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @property
    def is_happy_path_family(self) -> bool:
        return self.category.startswith("happy_path")

    @property
    def is_adversarial(self) -> bool:
        return self.category in ("adversarial", "adversarial_retrieved")

    @property
    def is_retrieved_injection(self) -> bool:
        """P4: the attack arrives via retrieved content, not the user's own message."""
        return self.category == "adversarial_retrieved" or self.taxonomy == "P4"


_REQUIRED_FIELDS = ("id", "category", "user", "should_settle")


def _parse_one(raw: dict[str, Any]) -> Scenario:
    """Both `scenarios.seed.yaml` (which omits `expected_tool`/`expected_final_tool`
    entirely for its adversarial entries) and `scenarios.yaml` (which sets them explicitly
    to `null`) must load the same way — a scenario with neither field, or both set to
    `None`, means "no live tool call is expected regardless of outcome," which is itself
    meaningful data (e.g. an input-rail block before `plan` ever runs), not a schema error.
    """
    missing = [f for f in _REQUIRED_FIELDS if f not in raw]
    if missing:
        raise ScenarioSchemaError(
            f"scenario {raw.get('id', '<unknown>')!r} is missing required field(s): {missing}"
        )
    known = {f for f in Scenario.__dataclass_fields__ if f != "raw"}
    kwargs = {k: v for k, v in raw.items() if k in known}
    return Scenario(raw=raw, **kwargs)


def load_scenarios(path: Path | str = DEFAULT_SCENARIOS_PATH) -> list[Scenario]:
    """Load and validate every scenario in `path`. Raises on the first malformed one."""
    text = Path(path).read_text(encoding="utf-8")
    entries = yaml.safe_load(text) or []
    if not isinstance(entries, list):
        raise ScenarioSchemaError(f"{path}: expected a top-level YAML list, got {type(entries)}")
    return [_parse_one(entry) for entry in entries]
