"""Gradio front-end for the same purchase graph `scripts/demo.py` drives from the CLI.

Same graph, same composition root (`payagent.mcp_server.__main__.build_deps_from_env`), same
two points that need a human because the graph will not fabricate them:

- SKU selection, once `retrieve` returns candidates — never inferred from retrieved prose
  (invariant I5).
- Step-up resolution, when `confirm` pauses with `status="awaiting_step_up"`. The "Approve"
  button below simulates the separate WebAuthn channel by calling
  `StepUpVerifier.issue_challenge`/`resolve_challenge` directly — never by feeding anything
  back through a tool argument, which is the P5 evasion the requirement exists to stop.

`deps`/`graph` are built once, lazily, on the first request and shared across the whole app
(a single local operator, same as the CLI demo — not a multi-tenant deployment). Per-visitor
progress (the current `PurchaseState` and the transcript so far) lives in a `gr.State`, so two
browser tabs each drive their own purchase through the one shared `deps`.

Usage:
    uv run python scripts/demo_gradio.py

Requires the same things `scripts/demo.py` does: Qdrant reachable at `QDRANT_URL` with the
catalog/policies collections ingested, and `OPENROUTER_API_KEY` configured.
"""

from __future__ import annotations

import os
import sys
from typing import Any
from uuid import uuid4

import gradio as gr
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from payagent.graph.graph import PurchaseGraph, build_graph
from payagent.graph.state import PurchaseState
from payagent.guardrails.untrusted import unwrap_untrusted_text
from payagent.mcp_server.__main__ import build_deps_from_env
from payagent.mcp_server.deps import ToolDeps
from payagent.observability.logging import configure_logging
from payagent.observability.tracing import configure_tracing, open_session_span

_deps: ToolDeps | None = None
_graph: PurchaseGraph | None = None


def _build_llm() -> ChatOpenAI:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY not configured. Copy .env.example to .env and fill it in."
        )
    return ChatOpenAI(
        model=os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        temperature=0,
    )


def _get_graph() -> tuple[ToolDeps, PurchaseGraph]:
    """Build `deps`/`graph` once, lazily. Not at import time, so the app still launches (and
    shows a clear error in the transcript) if Qdrant or the LLM key aren't configured yet."""
    global _deps, _graph
    if _deps is None or _graph is None:
        deps = build_deps_from_env()
        llm = _build_llm()
        _deps = deps
        _graph = build_graph(llm, deps.retriever, deps)
    return _deps, _graph


# --------------------------------------------------------------------------------- reporting


def _denial_line(denial: dict[str, Any] | None) -> str | None:
    if denial is None:
        return None
    ref = f" [{denial['policy_ref']}]" if denial.get("policy_ref") else ""
    return f"**Deny** ({denial['code']}){ref}: {denial['reason']}"


def _report_lines(step: str, state: PurchaseState) -> list[str]:
    """Same reporting shape as `scripts/demo.py`'s `_report`, as Markdown lines instead of prints."""
    lines = [f"### {step}", f"status: `{state.status}`"]

    if step == "plan" and state.status == "answered_directly":
        lines.append(state.direct_response or "")
        return lines

    if step == "retrieve":
        lines.append(
            f"{len(state.retrieved_chunks)} chunk(s), confidence={state.retrieval_confidence}"
        )
        return lines

    if step == "quote":
        if state.status == "quote_failed":
            error = state.quote["error"]
            lines.append(f"**Deny** ({error['code']}): {error['message']}")
        else:
            data = state.quote["data"]
            lines.append(
                f"**Allow**: {data['quantity']} x {data['sku']} = "
                f"{data['amount_cents']} {data['currency']} (quote_id={data['quote_id']})"
            )
        return lines

    if step in ("mandate", "confirm"):
        denial_line = _denial_line(state.policy_decision)
        lines.append(denial_line if denial_line else "**Allow**")
        return lines

    if step == "settle":
        if state.status == "settled":
            data = state.settlement_result["data"]
            lines.append(
                f"**Allow**: settlement_id={data['settlement_id']} "
                f"{data['amount_cents']} {data['currency']}"
            )
        else:
            error = state.settlement_result["error"]
            denial_line = _denial_line(error.get("denial"))
            lines.append(denial_line or f"**Deny** ({error['code']}): {error['message']}")
        return lines

    return lines


def _render(log: list[str]) -> str:
    return "\n\n".join(log)


# ------------------------------------------------------------------------------------ events
#
# `plan` and `retrieve` each make at least one real network call (an LLM turn; `retrieve` also
# embeds the query and searches Qdrant), so one `on_start` run is commonly 10-30+ seconds. A
# plain function that only returns once at the end leaves the page looking frozen for all of
# that — every handler below is a *generator* instead: it yields a "thinking" placeholder
# immediately, then yields the real result once it has one. Gradio renders each yield as it
# arrives, so the click has visible effect right away rather than the page looking unresponsive
# until the whole run finishes.
#
# Every handler yields/returns the same 9-tuple, matching `start_outputs` in `build_ui`:
#   (log_md, sku_radio, quantity_number, confirm_btn, stepup_note, approve_btn, decline_btn,
#    start_btn, session)
# `start_btn` is disabled the moment any handler starts working and re-enabled on every final
# yield, so a second click can't fire a concurrent run against the same session state.

_HIDE_SELECTION = (gr.update(visible=False), gr.update(visible=False), gr.update(visible=False))
_HIDE_STEP_UP = (gr.update(visible=False), gr.update(visible=False), gr.update(visible=False))
_SHOW_STEP_UP = (gr.update(visible=True), gr.update(visible=True), gr.update(visible=True))
_KEEP_SELECTION = (gr.update(), gr.update(), gr.update())
_KEEP_STEP_UP = (gr.update(), gr.update(), gr.update())


def _thinking(log: list[str], note: str = "Thinking...") -> tuple:
    """First yield of every handler: show a placeholder and disable `start_btn`, leaving every
    other control exactly as it was (a no-arg `gr.update()` changes nothing)."""
    return (
        _render([*log, f"_{note}_"]),
        *_KEEP_SELECTION,
        *_KEEP_STEP_UP,
        gr.update(interactive=False),
        None,
    )


def _done(log: list[str], selection: tuple, step_up: tuple, session: dict | None) -> tuple:
    """Last yield of every handler: the real result, with `start_btn` re-enabled."""
    return (_render(log), *selection, *step_up, gr.update(interactive=True), session)


def on_start(request: str, session: dict | None):
    yield _thinking([])

    try:
        deps, graph = _get_graph()
    except Exception as exc:
        yield _done([f"**Setup error:** {exc}"], _HIDE_SELECTION, _HIDE_STEP_UP, session)
        return

    if not request or not request.strip():
        yield _done(["Type a request first."], _HIDE_SELECTION, _HIDE_STEP_UP, session)
        return

    # One browser-side purchase attempt is one session (CLAUDE.md: one session = one
    # purchase). Generated once here and threaded through `session["session_id"]` to later
    # handlers (each Gradio event handler is its own call, so `open_session_span`'s context
    # can't stay open across handlers the way `run_guarded`'s single `with` block does for the
    # CLI/harness paths) — re-entered fresh, with the same `session_id`, in every handler below.
    session_id = str(uuid4())
    state = PurchaseState(user_request=request)
    log: list[str] = []

    with open_session_span("demo_gradio.on_start", session_id=session_id):
        state = graph.step("plan", state)
        log += _report_lines("plan", state)
        if state.status == "answered_directly":
            yield _done(
                log,
                _HIDE_SELECTION,
                _HIDE_STEP_UP,
                {"state": state, "log": log, "session_id": session_id},
            )
            return

        yield _thinking(log, note="Searching the catalog...")

        state = graph.step("retrieve", state)
        log += _report_lines("retrieve", state)
        catalog_hits = [c for c in state.retrieved_chunks if c.source == "catalog"]
        if not catalog_hits:
            log.append("Nothing found in the catalog for this request.")
            yield _done(
                log,
                _HIDE_SELECTION,
                _HIDE_STEP_UP,
                {"state": state, "log": log, "session_id": session_id},
            )
            return

        # Gradio's Radio accepts (label, value) pairs: the shopper sees the product, the code
        # still only ever gets the chunk_id back — the unwrapped prose is display-only (I5).
        choices = [
            (f"{c.chunk_id} — {unwrap_untrusted_text(c.text)}", c.chunk_id) for c in catalog_hits
        ]
        selection = (
            gr.update(visible=True, choices=choices, value=choices[0][1]),
            gr.update(visible=True),
            gr.update(visible=True),
        )
        yield _done(
            log,
            selection,
            _HIDE_STEP_UP,
            {"state": state, "log": log, "session_id": session_id},
        )


def _advance_confirm_then_settle(
    deps: ToolDeps, graph: PurchaseGraph, state: PurchaseState, log: list[str], session_id: str
) -> tuple:
    """Shared tail: run `confirm`, then either pause for step-up or fall through to `settle`."""
    with open_session_span("demo_gradio.confirm_then_settle", session_id=session_id):
        state = graph.step("confirm", state)
        log += _report_lines("confirm", state)

        if state.status == "awaiting_step_up":
            return _done(
                log,
                _HIDE_SELECTION,
                _SHOW_STEP_UP,
                {"state": state, "log": log, "session_id": session_id},
            )
        if state.status != "in_progress":
            return _done(
                log,
                _HIDE_SELECTION,
                _HIDE_STEP_UP,
                {"state": state, "log": log, "session_id": session_id},
            )

        state = graph.step("settle", state)
        log += _report_lines("settle", state)
        return _done(
            log,
            _HIDE_SELECTION,
            _HIDE_STEP_UP,
            {"state": state, "log": log, "session_id": session_id},
        )


def on_confirm_selection(sku: str, quantity: float, session: dict):
    state: PurchaseState = session["state"]
    log: list[str] = session["log"]
    session_id: str = session["session_id"]
    yield _thinking(log)

    deps, graph = _get_graph()
    state = state.model_copy(
        update={"selected_sku": sku, "selected_quantity": int(quantity or 1)}
    )

    with open_session_span("demo_gradio.on_confirm_selection", session_id=session_id):
        state = graph.step("quote", state)
        log += _report_lines("quote", state)
        if state.status != "in_progress":
            yield _done(
                log,
                _HIDE_SELECTION,
                _HIDE_STEP_UP,
                {"state": state, "log": log, "session_id": session_id},
            )
            return

        state = graph.step("mandate", state)
        log += _report_lines("mandate", state)
        if state.status != "in_progress":
            yield _done(
                log,
                _HIDE_SELECTION,
                _HIDE_STEP_UP,
                {"state": state, "log": log, "session_id": session_id},
            )
            return

    yield _advance_confirm_then_settle(deps, graph, state, log, session_id)


def on_approve_step_up(session: dict):
    state: PurchaseState = session["state"]
    log: list[str] = session["log"]
    session_id: str = session["session_id"]
    yield _thinking(log, note="Checking step-up...")

    deps, graph = _get_graph()
    # The separate channel — never a field on `state`, never a tool argument (P5).
    now = deps.clock()
    challenge = deps.step_up.issue_challenge(now=now)
    deps.step_up.resolve_challenge(challenge.challenge_id, now=now)
    log.append("_Step-up approved via the out-of-band channel._")

    yield _advance_confirm_then_settle(deps, graph, state, log, session_id)


def on_decline_step_up(session: dict) -> tuple:
    """Declining just logs it — the buttons stay up, since the purchase is still paused and
    approving later is still possible (unlike a terminal denial). No network call, so this
    stays a plain function rather than a generator."""
    state: PurchaseState = session["state"]
    log: list[str] = session["log"]
    session_id: str = session["session_id"]
    log.append("_Step-up declined. The purchase remains paused._")
    return _done(
        log,
        _HIDE_SELECTION,
        _SHOW_STEP_UP,
        {"state": state, "log": log, "session_id": session_id},
    )


# -------------------------------------------------------------------------------------- UI


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="PayAgent Guard — demo") as demo:
        gr.Markdown("# PayAgent Guard\nOne purchase, driven through the real purchase graph.")

        request_box = gr.Textbox(label="What do you want to buy?", placeholder="fone bluetooth")
        start_btn = gr.Button("Start", variant="primary")

        log_md = gr.Markdown()

        with gr.Row():
            sku_radio = gr.Radio(label="Pick a product", choices=[], visible=False)
            quantity_number = gr.Number(
                label="Quantity", value=1, precision=0, minimum=1, visible=False
            )
        confirm_btn = gr.Button("Confirm selection", visible=False)

        stepup_note = gr.Markdown("**Step-up (WebAuthn/passkey) required.**", visible=False)
        with gr.Row():
            approve_btn = gr.Button("Approve", visible=False)
            decline_btn = gr.Button("Decline", visible=False)

        session = gr.State(value=None)

        outputs = [
            log_md,
            sku_radio,
            quantity_number,
            confirm_btn,
            stepup_note,
            approve_btn,
            decline_btn,
            start_btn,
            session,
        ]
        start_btn.click(on_start, inputs=[request_box, session], outputs=outputs)
        confirm_btn.click(
            on_confirm_selection, inputs=[sku_radio, quantity_number, session], outputs=outputs
        )
        approve_btn.click(on_approve_step_up, inputs=[session], outputs=outputs)
        decline_btn.click(on_decline_step_up, inputs=[session], outputs=outputs)

    return demo


def main() -> None:
    load_dotenv()
    configure_logging(stream=sys.stderr)
    configure_tracing()
    build_ui().queue().launch()


if __name__ == "__main__":
    main()
