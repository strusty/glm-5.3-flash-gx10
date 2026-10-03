"""A fail-closed glm47 tool parser, registered as `glm47_failclosed`.

Loaded with `--tool-parser-plugin` and selected with
`--tool-call-parser glm47_failclosed`, so no upstream file is forked or mounted.

## Why

The forward pass is not bit-reproducible at M >> 1 on this hardware (see
../../dev/TOPK-CORRUPTION.md), and the token it lands on is occasionally part of a tool
call. Upstream handles a corrupted *name* by dropping the call: the turn ends
`stop` with no content and no `tool_calls`, which clients report as a dead
endpoint. It does not check argument *keys* at all, so a call carrying a garbage
key is emitted as though it were fine, the client echoes it into history, the
chat template re-renders it, and the model imitates its own corruption until a
turn is unparseable.

Containment, not a cure: nothing here stops the first bad token, it only keeps
one from poisoning the rest of the session.

Approach and both of its corrections come from
NNNtrance/GLM-5.3-Flash-EXL3-DGX-Spark, issue #7 and PR #11.

## What it does

Validates at TOOL_CALL_END, once the whole call is buffered
(`stream_arg_deltas=False`): the name must be shaped like a tool name and resolve
to one the request offered, the arguments must parse to an object, and every key
must appear in that tool's schema.

A call that fails is **re-offered with a sentinel argument key** rather than
dropped or turned into text. Surfacing it as content clears the tool slot, and an
agent loop reads that as a finished turn and stops -- the bug behind PR #11. A
call no client can accept is a retryable error instead, so the loop continues.
The sentinel's value names the specific mistake and lists the valid keys, because
an unexplained rejection invites the model to repeat it.

Name resolution takes an exact match first, then the longest offered name the
emitted one starts with, so `bash</arg_key>...` resolves to `bash` while
`bash1635` does not.

## Nested and abandoned calls

Two shapes pass the checks above with valid keys and still carry the wrong call:

- A call opened inside the think block. GLM's `(REASONING, TOOL_START)` transition
  ends reasoning implicitly, so a `<tool_call>` written mid-thought opens a call
  whose argument then swallows the rest of the reasoning, the `</think>` and the
  call written after it. Upstream executes it with the reasoning as its argument.
- An abandoned call: `bash<arg_key>command</arg_key><arg_value># note\nweb_search
  <arg_key>query...`, where the model switched tools without closing the first.

At call end the text after `</think><tool_call>NAME<arg_key>`, or an offered tool
name directly followed by `<arg_key>` inside an argument value, is checked as a
call of its own. When it passes, the outer call is refused with a reason that
spells out the recovered call, so the model re-issues it in one step. Recovered
text is never executed: it could be a file's content, and GLM's format cannot
carry literal `<arg_key>` or `</tool_call>` inside an argument anyway.
`GLM47_REPAIR=0` turns this off.

## Streaming

Arguments are buffered until the call ends, so a long call sends nothing while it
is written; clients with an idle timeout drop the stream. While a call is
buffered, an empty argument delta goes out every `GLM47_KEEPALIVE_S` seconds
(default 5, 0 turns it off). A refused call whose name was already streamed is
re-offered with arguments only: clients that concatenate deltas (the openai
SDK's stream helper) would otherwise read the name twice.

`GLM47_FAIL_CLOSED=0` restores upstream behaviour. `GLM47_REJECT_AS_TEXT=1`
surfaces refusals as content instead, which is the pre-#11 behaviour and ends
agent loops.
"""

from __future__ import annotations

import json
import os
import re
import time

# Moved from vllm.entrypoints.openai.engine.protocol on vLLM main; the old
# path is why this plugin failed to register on the 2026-09-22 nightly.
from vllm.entrypoints.generate.base.protocol import (
    DeltaFunctionCall,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.logger import init_logger
from vllm.parser.engine.adapters import make_adapters
from vllm.parser.glm47_moe import Glm47MoeParser
from vllm.tool_parsers.abstract_tool_parser import ToolParserManager
from vllm.tool_parsers.utils import find_tool_name, find_tool_properties

logger = init_logger(__name__)

# Tool and argument names as these schemas spell them: a leading letter or
# underscore, then word characters, dots or dashes. Narrower than the JSON that
# reaches us, which is the point.
NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]{0,63}$")

# No real schema declares this, so a client that validates arguments must reject
# the call, and one that does not gets a message saying what happened.
SENTINEL_KEY = "_rejected_by_server"


def _flag(name: str, default: str) -> bool:
    return os.environ.get(name, default) not in ("0", "false", "False")


ENABLED = _flag("GLM47_FAIL_CLOSED", "1")
REJECT_AS_TEXT = _flag("GLM47_REJECT_AS_TEXT", "0")
REPAIR = _flag("GLM47_REPAIR", "1")
KEEPALIVE_S = float(os.environ.get("GLM47_KEEPALIVE_S", "5"))

ARG_KEY = "<arg_key>"
NAME_CHAR = re.compile(r"[A-Za-z0-9_.-]")
# A call opened inside the think block: the reasoning close, a new call, its first key.
NESTED_RE = re.compile(r"</think>\s*<tool_call>\s*([A-Za-z_][A-Za-z0-9_.-]{0,63})\s*<arg_key>")


class Glm47FailClosedParser(Glm47MoeParser):
    """Glm47MoeParser that refuses any call it cannot fully validate."""

    def __init__(self, tokenizer, tools=None, **kwargs) -> None:
        super().__init__(tokenizer, tools, **kwargs)
        # An argument delta cannot be withdrawn once sent, and the decision needs
        # the whole argument object.
        self._stream_arg_deltas = False
        # Sentinel arguments of refused calls by call id. A non-streaming result
        # is rebuilt from each slot's raw arguments, so it needs these too.
        self._refused: dict[str, str] = {}
        # When each buffered slot last sent anything, for the keep-alive.
        self._last_emit: dict[int, float] = {}

    def _reset(self, *args, **kwargs) -> None:
        super()._reset(*args, **kwargs)
        self._refused = {}
        self._last_emit = {}

    def _handle_arg_chunk(self, event, deltas) -> None:
        super()._handle_arg_chunk(event, deltas)
        if KEEPALIVE_S <= 0 or self._stream_arg_deltas:
            return
        idx = event.tool_index
        if not (0 <= idx < len(self._tool_slots)) or not self._tool_slots[idx].name_sent:
            return
        now = time.monotonic()
        last = self._last_emit.get(idx)
        if last is None or any(d.index == idx for d in deltas):
            self._last_emit[idx] = now
            return
        if now - last >= KEEPALIVE_S:
            self._last_emit[idx] = now
            deltas.append(DeltaToolCall(index=idx, function=DeltaFunctionCall(arguments="")))

    def _json_for(self, name: str, raw: str) -> str | None:
        converter = self._arg_converter
        if converter is None:
            return None
        try:
            text = converter(raw, False)
        except (json.JSONDecodeError, ValueError, TypeError):
            return None
        return self._fix_arg_types(text, name) if text else text

    def _recover(self, idx: int) -> tuple[str, str, str] | None:
        """(kind, name, arguments JSON) of a valid call nested in this slot's arguments."""
        raw = self._tool_slots[idx].args or ""
        found = None
        m = None
        for m in NESTED_RE.finditer(raw):
            pass  # the last one is the call the model finished
        if m is not None:
            found = ("nested", m.group(1), raw[m.end() - len(ARG_KEY):])
        else:
            best = None
            for name in self._offered_names():
                start = 0
                while (j := raw.find(name + ARG_KEY, start)) != -1:
                    start = j + 1
                    if j == 0 or NAME_CHAR.match(raw[j - 1]):
                        continue  # the call's own first key, or the tail of a longer name
                    if raw.rfind("<arg_value>", 0, j) <= raw.rfind("</arg_value>", 0, j):
                        continue  # not inside an argument value
                    if best is None or j < best[0]:
                        best = (j, name)
                    break
            if best is not None:
                found = ("abandoned", best[1], raw[best[0] + len(best[1]):])
        if found is None:
            return None
        kind, name, args_raw = found
        if args_raw.endswith("</tool_call>"):
            args_raw = args_raw[: -len("</tool_call>")]
        args_json = self._json_for(name, args_raw)
        if args_json is None or self._reject_reason(name, args_json) is not None:
            return None
        return kind, name, args_json

    def _args_json(self, idx: int) -> str:
        """The slot's arguments as the client would receive them.

        GLM writes arguments as <arg_key>/<arg_value> pairs and the slot keeps
        that raw text; the engine's converter turns it into JSON. Text the
        converter cannot handle is returned raw, and fails as "not JSON".
        """
        slot = self._tool_slots[idx]
        converter = self._arg_converter
        if converter is None:
            return slot.args
        try:
            text = converter(slot.args, False)
        except (json.JSONDecodeError, ValueError, TypeError):
            return slot.args
        return self._fix_arg_types(text, slot.name) if text and slot.name else text

    def _build_extracted_result(self, *deltas):
        result = super()._build_extracted_result(*deltas)
        calls = list(result.tool_calls or [])
        for call in calls:
            if call.id in self._refused:
                call.function.arguments = self._refused[call.id]
        # Upstream drops a call whose name no tool matches; re-offer it like the
        # streaming path does, so the turn is not left empty.
        seen = {call.id for call in calls}
        for slot in self._tool_slots:
            if slot.id in self._refused and slot.id not in seen:
                calls.append(ToolCall(id=slot.id, function=FunctionCall(name=slot.name,
                                                                         arguments=self._refused[slot.id])))
        if len(calls) != len(result.tool_calls or []):
            result = ExtractedToolCallInformation(tools_called=True, tool_calls=calls, content=result.content)
        return result

    def _offered_names(self) -> list[str]:
        names: list[str] = []
        for tool in self._tools or []:
            function = getattr(tool, "function", None)
            name = getattr(function, "name", None)
            if isinstance(name, str) and name:
                names.append(name)
        return names

    def _resolve_name(self, emitted: str) -> str | None:
        """The offered tool this name means, or None.

        Exact match first. Otherwise the longest offered name the emitted one
        starts with, which recovers a name whose tail is corrupted markup without
        matching a different tool that merely shares a prefix.
        """
        if find_tool_name(self._tools, emitted):
            return emitted
        candidates = [n for n in self._offered_names() if emitted.startswith(n)]
        if not candidates:
            return None
        best = max(candidates, key=len)
        tail = emitted[len(best):]
        if not tail:
            return best
        # What follows decides: a name character means a different tool
        # (`bash1635`), anything else means corruption after the real name
        # (`bash</arg_key>`).
        return None if re.match(r"[A-Za-z0-9_.-]", tail) else best

    def _reject_reason(self, name: str, args_json: str) -> str | None:
        """Why this call cannot be trusted, or None if it can."""
        if not NAME_RE.match(name):
            return f"tool name {name!r} is not shaped like a tool name"
        if self._tools and not find_tool_name(self._tools, name):
            return f"tool name {name!r} is not one this request offered"
        text = (args_json or "").strip()
        if not text:
            return None
        try:
            args = json.loads(text)
        except (json.JSONDecodeError, ValueError) as exc:
            return f"arguments for {name!r} are not JSON: {exc}"
        if not isinstance(args, dict):
            return f"arguments for {name!r} are {type(args).__name__}, not an object"
        properties = find_tool_properties(self._tools, name) or {}
        for key in args:
            if not isinstance(key, str) or not NAME_RE.match(key):
                return f"argument key {key!r} is not shaped like a key"
            # No declared properties means any key is unverifiable rather than
            # wrong, so it passes.
            if properties and key not in properties:
                valid = ", ".join(sorted(properties)) or "none"
                return (
                    f"argument key {key!r} is not in the schema for {name!r}; "
                    f"valid keys are: {valid}"
                )
        return None

    def _handle_tool_end(self, event, deltas) -> None:
        idx = event.tool_index
        if not ENABLED or not (0 <= idx < len(self._tool_slots)):
            super()._handle_tool_end(event, deltas)
            return

        slot = self._tool_slots[idx]
        emitted = (slot.name or self._try_extract_name(idx) or "").strip()
        resolved = self._resolve_name(emitted)
        was_sent = slot.name_sent
        recovered = self._recover(idx) if REPAIR else None
        if recovered is not None:
            kind, rec_name, rec_args = recovered
            how = ("was opened inside your reasoning, so its arguments swallowed the rest "
                   "of the reasoning and the call you wrote after it" if kind == "nested" else
                   "was left unfinished when you started another call inside its arguments")
            reason = (f"nothing was run: this {resolved or emitted or 'tool'} call {how}. The call "
                      f"you meant appears to be {rec_name} with arguments {rec_args}; issue it on its "
                      f"own. (To write tool-call markup literally inside an argument, split the "
                      f"tags, e.g. '</th' + 'ink>'.)")
        else:
            # Converting here leaves the slot and the converter's state alone.
            reason = self._reject_reason(resolved or emitted, self._args_json(idx))
        if reason is None:
            if resolved and resolved != emitted:
                slot.name = resolved
            super()._handle_tool_end(event, deltas)
            return

        # Drain this slot's converter so its pending text cannot surface inside
        # the next call.
        self._flush_arg_converter(idx)
        logger.warning("glm47 fail-closed: %s", reason)

        if REJECT_AS_TEXT:
            # With no tool delta for this event the engine flushes deferred
            # content as ordinary content. This ends agent loops; see PR #11.
            self._deferred_content += f"\n[tool call refused: {reason}]\n"
            slot.name_sent = True
            slot.streamed_json = slot.args or ""
            return

        # Re-offer under the resolved name where there is one, so the client
        # recognises the tool and rejects it on the arguments.
        name = resolved or emitted if NAME_RE.match(emitted) else (resolved or "unknown_tool")
        slot.name = name
        slot.name_sent = True
        self._ensure_tool_id(slot, name)
        arguments = json.dumps({SENTINEL_KEY: reason}, ensure_ascii=False)
        slot.streamed_json = arguments
        self._refused[slot.id] = arguments
        if was_sent:
            # The client already has this call's name and id: arguments only, or a
            # client that concatenates deltas reads the name twice.
            deltas.append(DeltaToolCall(index=idx, function=DeltaFunctionCall(arguments=arguments)))
        else:
            deltas.append(
                DeltaToolCall(
                    index=idx,
                    id=slot.id,
                    type="function",
                    function=DeltaFunctionCall(name=name, arguments=arguments),
                )
            )


_REASONING_ADAPTER, _TOOL_ADAPTER = make_adapters(Glm47FailClosedParser)


@ToolParserManager.register_module("glm47_failclosed")
class Glm47FailClosedToolParser(_TOOL_ADAPTER):  # type: ignore[valid-type, misc]
    supports_required_and_named = False
    structural_tag_model = "glm_4_7"
