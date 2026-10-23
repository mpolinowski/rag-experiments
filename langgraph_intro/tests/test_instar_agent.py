"""Tests for 05v2_langgraph_react_instar_agent.py.

Camera-touching tests are read-only (param.cgi GETs against the local
camera). Write behavior is verified against a fake device, and the full
propose -> approve/reject -> execute graph flow runs against a scripted
fake LLM so no writes can ever reach the real camera.
"""

import importlib.util
from pathlib import Path

import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "langgraph_intro"
    / "05v2_langgraph_react_instar_agent.py"
)

_spec = importlib.util.spec_from_file_location("instar_agent_v2", MODULE_PATH)
module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(module)


# --------------------------------------------------------------------
# Fixture data shaped like the real camera's getmdattr response
# --------------------------------------------------------------------

AREA_GEOMETRY = {
    1: {"x": "0", "y": "0", "w": "960", "h": "1080",
        "polyX": "0,960,960,0", "polyY": "0,0,1080,1080"},
    2: {"x": "960", "y": "0", "w": "960", "h": "1080",
        "polyX": "960,1920,1920,960", "polyY": "0,0,1080,1080"},
    3: {"x": "1920", "y": "0", "w": "960", "h": "1080",
        "polyX": "1920,2880,2880,1920", "polyY": "0,0,1080,1080"},
    4: {"x": "2880", "y": "0", "w": "960", "h": "1080",
        "polyX": "2880,3840,3840,2880", "polyY": "0,0,1080,1080"},
    5: {"x": "0", "y": "1080", "w": "960", "h": "1080",
        "polyX": "0,960,960,0", "polyY": "1080,1080,2160,2160"},
    6: {"x": "960", "y": "1080", "w": "960", "h": "1080",
        "polyX": "960,1920,1920,960", "polyY": "1080,1080,2160,2160"},
    7: {"x": "1920", "y": "1080", "w": "960", "h": "1080",
        "polyX": "1920,2880,2880,1920", "polyY": "1080,1080,2160,2160"},
    8: {"x": "2880", "y": "1080", "w": "960", "h": "1080",
        "polyX": "2880,3840,3840,2880", "polyY": "1080,1080,2160,2160"},
}


def make_mdattr_body(enables: dict[int, str]) -> str:
    lines = ['cmd="getmdattr";']
    for area in range(1, 9):
        geom = AREA_GEOMETRY[area]
        lines.append(f'enable[{area}]="{enables[area]}";')
        lines.append(f'rect[{area}]="1";')
        lines.append(f's[{area}]="80";')
        lines.append(f'x[{area}]="{geom["x"]}";')
        lines.append(f'y[{area}]="{geom["y"]}";')
        lines.append(f'w[{area}]="{geom["w"]}";')
        lines.append(f'h[{area}]="{geom["h"]}";')
        lines.append(f'polyX[{area}]="{geom["polyX"]}";')
        lines.append(f'polyY[{area}]="{geom["polyY"]}";')
        lines.append(f'name[{area}]="";')
    lines.append('response="200";')
    return "\n".join(lines)


ALL_DISABLED = make_mdattr_body({n: "0" for n in range(1, 9)})
AREA1_ENABLED = make_mdattr_body(
    {n: ("1" if n == 1 else "0") for n in range(1, 9)}
)


class FakeDevice:
    """Stands in for the camera. Records every request it receives."""

    def __init__(self, before_body: str, after_body: str):
        self.before_body = before_body
        self.after_body = after_body
        self.calls: list[dict] = []
        self._gets = 0

    def __call__(self, path: str, params: dict) -> str:
        self.calls.append({"path": path, "params": dict(params)})
        if params.get("cmd") == "getmdattr":
            self._gets += 1
            # First read = current state, read-back = state after write.
            return self.before_body if self._gets == 1 else self.after_body
        return '{"setmdattr":"ok"}'

    @property
    def writes(self) -> list[dict]:
        return [c for c in self.calls if c["params"].get("cmd") == "setmdattr"]

    @property
    def reads(self) -> list[dict]:
        return [c for c in self.calls if c["params"].get("cmd") == "getmdattr"]


def make_fake_llm():
    """Scripted LLM: read camera, propose area 1, wait for the decision.

    Message order follows the graph flow:
      1. investigate   (tool call)
      2. propose       (tool call)
      3. plain answer  (no tool calls + pending_plan set -> approval node)
      4. plain answer  (after the user approves/rejects)
    """
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage

    return GenericFakeChatModel(messages=iter([
        AIMessage(
            content="",
            tool_calls=[{
                "name": "investigate_device",
                "args": {"command_id": "get_motion"},
                "id": "call_read",
            }],
        ),
        AIMessage(
            content="",
            tool_calls=[{
                "name": "propose_device_changes",
                "args": {
                    "changes_json": (
                        '[{"command_id": "set_motion", "area": 1,'
                        ' "params": {"enable": "1"}}]'
                    )
                },
                "id": "call_propose",
            }],
        ),
        AIMessage(content="I prepared a plan for motion area 1. Please review."),
        AIMessage(content="Acknowledged - proceeding according to your decision."),
    ]))


def config_for(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


# --------------------------------------------------------------------
# parse_mdattr
# --------------------------------------------------------------------

def test_parse_mdattr_parses_all_areas_and_ignores_response_line():
    parsed = module.parse_mdattr(ALL_DISABLED)

    assert sorted(parsed) == [str(n) for n in range(1, 9)]
    area1 = parsed["1"]
    assert area1["enable"] == "0"
    assert area1["rect"] == "1"
    assert area1["s"] == "80"
    assert area1["x"] == "0"
    assert area1["y"] == "0"
    assert area1["w"] == "960"
    assert area1["h"] == "1080"
    assert area1["polyX"] == "0,960,960,0"
    assert area1["polyY"] == "0,0,1080,1080"
    assert area1["name"] == ""
    # response="200" must not be picked up as an area entry
    assert "response" not in area1


# --------------------------------------------------------------------
# validate_changes
# --------------------------------------------------------------------

def test_validate_changes_accepts_a_valid_enable_change():
    valid = module.validate_changes([
        {"command_id": "set_motion", "area": 3, "params": {"enable": "0"}}
    ])
    assert valid == [
        {"command_id": "set_motion", "area": "3", "params": {"enable": "0"}}
    ]


@pytest.mark.parametrize("bad_changes,error_fragment", [
    ([{"command_id": "set_motion", "area": 9, "params": {"enable": "0"}}], "area must be 1-8"),
    ([{"command_id": "set_motion", "area": 1, "params": {"sensitivity": "50"}}], "unsupported fields"),
    ([{"command_id": "set_motion", "area": 1, "params": {"polyX": "0,0,1,1"}}], "unsupported fields"),
    ([{"command_id": "set_motion", "area": 1, "params": {"s": "150"}}], "invalid value for s"),
    ([{"command_id": "set_motion", "area": 1, "params": {"enable": "2"}}], "invalid value for enable"),
    ([{"command_id": "set_motion", "area": 1, "params": {}}], "at least one param"),
    ([{"command_id": "set_motion"}], "area"),
    ([{"command_id": "set_firmware", "area": 1, "params": {"enable": "0"}}], "not allowed"),
    ([{"command_id": "set_motion", "area": 2, "params": {"enable": "0"}},
      {"command_id": "set_motion", "area": 2, "params": {"s": "10"}}], "duplicate"),
    ("not a list", "JSON array"),
])
def test_validate_changes_rejects_invalid_input(bad_changes, error_fragment):
    with pytest.raises(ValueError, match=error_fragment):
        module.validate_changes(bad_changes)


def test_validate_changes_rejects_bool_values():
    # bool passes isinstance(x, int) in Python, so it must be rejected
    # explicitly: JSON true would otherwise validate as "True"/"1".
    with pytest.raises(ValueError, match="invalid value type"):
        module.validate_changes([
            {"command_id": "set_motion", "area": 1, "params": {"enable": True}}
        ])


# --------------------------------------------------------------------
# _build_set_motion_params: full field set, merged with current state
# --------------------------------------------------------------------

def test_build_set_motion_params_merges_overrides_over_current_state():
    current = module.parse_mdattr(ALL_DISABLED)
    params = module._build_set_motion_params(
        "1", current, {"enable": "1"}
    )

    assert params == {
        "cmd": "setmdattr",
        "area": "1",
        "enable": "1",  # override applied
        "rect": "1",    # kept from current state
        "s": "80",
        "x": "0",
        "y": "0",
        "w": "960",
        "h": "1080",
        "polyX": "0,960,960,0",
        "polyY": "0,0,1080,1080",
        "name": "",
    }


def test_build_set_motion_params_rejects_unknown_area():
    current = module.parse_mdattr(ALL_DISABLED)
    with pytest.raises(ValueError, match="not found"):
        module._build_set_motion_params("9", current, {"enable": "0"})


# --------------------------------------------------------------------
# _device_request guards
# --------------------------------------------------------------------

def test_device_request_rejects_base_url_with_credentials(monkeypatch):
    monkeypatch.setattr(
        module, "DEVICE_BASE_URL", "http://user:pass@192.168.2.224/x?q=1"
    )
    with pytest.raises(ValueError, match="Invalid INSTAR_DEVICE_URL"):
        module._device_request("/param.cgi", {"cmd": "getnetattr"})


def test_device_request_rejects_bad_cgi_path(monkeypatch):
    with pytest.raises(ValueError, match="Invalid configured CGI path"):
        module._device_request("param.cgi", {"cmd": "x"})


def test_device_request_treats_empty_body_as_failure(monkeypatch):
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            # Mimics the camera's unauthenticated response: 200 + empty body
            self.send_response(200)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        monkeypatch.setattr(
            module, "DEVICE_BASE_URL", f"http://127.0.0.1:{server.server_address[1]}"
        )
        monkeypatch.setattr(module, "DEVICE_USERNAME", "")
        monkeypatch.setattr(module, "DEVICE_PASSWORD", "")
        with pytest.raises(RuntimeError, match="empty response"):
            module._device_request("/param.cgi", {"cmd": "getnetattr"})
    finally:
        server.shutdown()
        server.server_close()


# --------------------------------------------------------------------
# propose_device_changes tool (no device access)
# --------------------------------------------------------------------

def test_propose_rejects_invalid_plan_and_writes_nothing():
    result = module.propose_device_changes.invoke({
        "changes_json": '[{"command_id": "set_motion", "area": 1, "params": {"sensitivity": "50"}}]'
    })
    data = __import__("json").loads(result)
    assert data["ok"] is False
    assert "sensitivity" in data["error"]
    assert "plan_id" not in data


def test_propose_returns_awaiting_approval_and_no_side_effects():
    result = module.propose_device_changes.invoke({
        "changes_json": '[{"command_id": "set_motion", "area": 1, "params": {"enable": "1"}}]'
    })
    data = __import__("json").loads(result)
    assert data["ok"] is True
    assert data["status"] == "awaiting_user_approval"
    assert data["plan"]["changes"][0]["area"] == "1"


# --------------------------------------------------------------------
# Actor node: never writes without a matching approval
# --------------------------------------------------------------------

def test_actor_without_plan_performs_no_calls(monkeypatch):
    fake = FakeDevice(ALL_DISABLED, AREA1_ENABLED)
    monkeypatch.setattr(module, "_device_request", fake)

    result = module.actor_node({"pending_plan": None})

    assert fake.calls == []
    content = result["messages"][0].content
    assert "No pending plan exists" in content


def test_actor_with_unmatched_approval_performs_no_calls(monkeypatch):
    fake = FakeDevice(ALL_DISABLED, AREA1_ENABLED)
    monkeypatch.setattr(module, "_device_request", fake)

    plan = {
        "plan_id": "abcdef123456",
        "changes": [
            {"command_id": "set_motion", "area": "1", "params": {"enable": "1"}}
        ],
        "created_at": __import__("time").time(),
    }
    state = {
        "pending_plan": plan,
        "approved_plan_id": "deadbeef0000",  # different plan!
        "approval_status": "approved",
    }

    result = module.actor_node(state)

    assert fake.calls == []
    assert "matching approval" in result["messages"][0].content


# --------------------------------------------------------------------
# approval_node decision handling
# --------------------------------------------------------------------

def test_approval_node_rejects_payload_with_wrong_plan_id():
    plan = {
        "plan_id": "abcdef123456",
        "changes": [
            {"command_id": "set_motion", "area": "1", "params": {"enable": "1"}}
        ],
        "created_at": __import__("time").time(),
    }

    # interrupt() is patched to behave as if the CLI resumed with a
    # decision referencing a different plan.
    orig_interrupt = module.interrupt
    module.interrupt = lambda payload: {"decision": "approve", "plan_id": "ffffffffffff"}
    try:
        result = module.approval_node({"pending_plan": plan})
    finally:
        module.interrupt = orig_interrupt

    assert result["approval_status"] == "rejected"
    assert result["approved_plan_id"] is None


# --------------------------------------------------------------------
# End-to-end graph flow with fake LLM + fake device
# --------------------------------------------------------------------

def test_graph_approval_flow_approve_writes_and_verifies(monkeypatch):
    import json as _json

    from langgraph.types import Command

    fake = FakeDevice(ALL_DISABLED, AREA1_ENABLED)
    monkeypatch.setattr(module, "_device_request", fake)
    llm = make_fake_llm()
    monkeypatch.setattr(module, "llm", llm)

    state = {
        "messages": [
            module.SystemMessage(content=module.SYSTEM_PROMPT),
            module.HumanMessage(content="Please enable motion detection in alarm area 1."),
        ],
        "pending_plan": None,
        "approved_plan_id": None,
        "approval_status": None,
    }

    # 1) invoke -> graph must PAUSE at approval, before any write.
    result = module.graph.invoke(state, config=config_for("e2e-approve"))
    payload = module.get_interrupt_payload(result)
    assert payload is not None, "expected the graph to interrupt for approval"
    assert payload["changes"] == [
        {"command_id": "set_motion", "area": "1", "params": {"enable": "1"}}
    ]
    # Only the read happened so far - no set command at all.
    assert fake.writes == []
    assert len(fake.reads) == 1

    # 2) resume with approval of the exact plan.
    result = module.graph.invoke(
        Command(resume={"decision": "approve", "plan_id": payload["plan_id"]}),
        config=config_for("e2e-approve"),
    )
    assert module.get_interrupt_payload(result) is None

    # 3) exactly one write, full field set, merged with the current state.
    assert len(fake.writes) == 1
    write = fake.writes[0]["params"]
    area1 = module.parse_mdattr(ALL_DISABLED)["1"]
    assert write == {
        "cmd": "setmdattr",
        "area": "1",
        **{f: area1[f] for f in module.MD_ATTR_FIELDS},
        "enable": "1",  # the one override
    }

    # 4) reads: investigate (1) + actor current-state read (2) + actor
    #    read-back verification (3).
    assert len(fake.reads) == 3

    # 5) the actor reported a verified success into the conversation.
    actor_messages = [
        m.content for m in result["messages"]
        if "Device execution result" in getattr(m, "content", "")
    ]
    assert actor_messages, "actor result message missing"
    reported = _json.loads(
        actor_messages[0].split(": ", 1)[1]
    )
    assert reported["ok"] is True
    assert reported["verification"]["per_area"]["1"] is True


def test_graph_approval_flow_reject_performs_no_writes(monkeypatch):
    from langgraph.types import Command

    fake = FakeDevice(ALL_DISABLED, AREA1_ENABLED)
    monkeypatch.setattr(module, "_device_request", fake)
    monkeypatch.setattr(module, "llm", make_fake_llm())

    state = {
        "messages": [
            module.SystemMessage(content=module.SYSTEM_PROMPT),
            module.HumanMessage(content="Enable motion detection in area 1."),
        ],
        "pending_plan": None,
        "approved_plan_id": None,
        "approval_status": None,
    }

    result = module.graph.invoke(state, config=config_for("e2e-reject"))
    payload = module.get_interrupt_payload(result)
    assert payload is not None

    result = module.graph.invoke(
        Command(resume={"decision": "reject", "plan_id": payload["plan_id"]}),
        config=config_for("e2e-reject"),
    )

    assert fake.writes == [], "rejection must not reach the camera"
    assert module.get_interrupt_payload(result) is None


def test_graph_rejects_resume_for_wrong_plan_even_when_approving(monkeypatch):
    from langgraph.types import Command

    fake = FakeDevice(ALL_DISABLED, AREA1_ENABLED)
    monkeypatch.setattr(module, "_device_request", fake)
    monkeypatch.setattr(module, "llm", make_fake_llm())

    state = {
        "messages": [
            module.SystemMessage(content=module.SYSTEM_PROMPT),
            module.HumanMessage(content="Enable motion detection in area 1."),
        ],
        "pending_plan": None,
        "approved_plan_id": None,
        "approval_status": None,
    }

    result = module.graph.invoke(state, config=config_for("e2e-wrongplan"))
    payload = module.get_interrupt_payload(result)
    assert payload is not None

    # Approve, but reference a plan ID that was never proposed.
    module.graph.invoke(
        Command(resume={"decision": "approve", "plan_id": "ffffffffffff"}),
        config=config_for("e2e-wrongplan"),
    )

    assert fake.writes == []


# --------------------------------------------------------------------
# Live camera read (read-only, never writes) - needs the local camera
# --------------------------------------------------------------------

def test_live_camera_read_returns_eight_areas():
    result = module.investigate_device.invoke({"command_id": "get_motion"})
    data = __import__("json").loads(result)

    assert data["ok"] is True, data.get("error")
    assert [a["area"] for a in data["areas"]] == [1, 2, 3, 4, 5, 6, 7, 8]
    for area in data["areas"]:
        assert area["enable"] in {"0", "1"}
        assert area["s"] != ""
        assert area["x"] != "" and area["y"] != ""
        assert area["polyX"].count(",") == 3
