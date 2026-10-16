"""The stream protocol (`stbench/stream_protocol.py`): the channels' reply to a question's `needs`, the
reply checks, the question views and the in-process `System` base. Expected values are hand tables from the plan
(§2.4)."""

from __future__ import annotations

from itertools import chain, combinations

import pytest

from stbench.answers import TYPES, Malformed, parse_answer
from stbench.stream_protocol import (
    CAPABILITIES,
    CHANNELS,
    DEFAULT_CHANNEL,
    GATE_CHANNEL,
    OPS,
    TIMEOUTS,
    System,
    ask_view,
    check_reply,
    failed,
    provides,
    read_answers,
    start_message,
    structured_view,
    timeout_key,
)

# The eight keys of a question row (plan §2.5), written out here so the views are checked against the plan.
QUESTION = {
    "qid": "0f3a9c1d7b2e4a65", "when": "during", "decision_k": 4, "text": "Is the kitchen clear right now?",
    "template": "zone_clear", "slots": {"zone": "kitchen", "tense": {"now": True, "hints": ["floor"]}},
    "answer_type": "yes_no", "needs": ["see", "pose"],
}


def test_the_picture_channel_asks_a_subset_of_the_gate_channel():
    """Contract (plan §2.4): while no family needs `rgb`, every `needs` set a question may have on the picture channel
    (`rgb_pose`, the default and the only agent channel) is also allowed on the gate channel `detections`, so the gate
    never passes on a channel that asks other questions. Hand sets, then every subset of the capabilities without
    `rgb`. Regression caught: a capability added to the picture channel that `detections` does not provide."""
    channel = "rgb_pose"
    assert provides("detections") == {"depth", "pose", "status", "detections", "see"}
    without_rgb = [c for c in CAPABILITIES if c != "rgb"]
    subsets = list(chain.from_iterable(combinations(without_rgb, n) for n in range(len(without_rgb) + 1)))
    assert len(subsets) == 32
    for needs in map(set, subsets):
        if needs <= provides(channel):
            assert needs <= provides("detections"), needs


def test_the_default_channel_is_the_camera_alone_and_a_question_that_needs_depth_is_not_asked_on_it():
    """Contract (2026-10-10: no depth image for the agent; VISION rule 9): the default channel is `rgb_pose`, which
    sends the camera image, the pose and the status and no depth, so what it provides leaves out `depth` and a question
    whose `needs` name depth is outside it; the gate channel stays `detections`, and no channel but it sends depth
    (0.6.0: `rgbd_pose` is retired). Hand sets. Regression caught: the default sliding back to a channel that sends
    depth (the agent would again be handed a helper), or `rgb_pose` sending a depth file a system could read."""
    assert DEFAULT_CHANNEL == "rgb_pose" and GATE_CHANNEL == "detections"
    assert CHANNELS["rgb_pose"] == {"rgb", "pose", "status"}
    assert set(CHANNELS) == {"rgb_pose", "detections"}
    assert [name for name, sends in CHANNELS.items() if "depth" in sends] == ["detections"]
    assert provides("rgb_pose") == {"rgb", "pose", "status", "see"} and "depth" not in provides("rgb_pose")
    assert {"depth", "pose"} <= provides("detections") and not {"depth", "pose"} <= provides("rgb_pose")
    assert {"see", "pose"} <= provides("rgb_pose")
    assert start_message("s", {}, "rgb_pose", "/w", "/s", {})["sends"] == ["pose", "rgb", "status"]


def test_detections_and_rgb_never_share_a_channel():
    """Contract (plan §2.4): no channel sends both pictures and perfect detections. Regression caught: a channel
    with both, which a system could use to see by detections and read by pictures at once."""
    assert len(CHANNELS) >= 2
    for channel, sends in CHANNELS.items():
        assert not {"rgb", "detections"} <= sends, channel
        assert not {"rgb", "detections"} <= provides(channel), channel


@pytest.mark.parametrize("stale", ["rgb", "oracle", "", "RGBD_POSE", "rgbd_pose"])
def test_a_stale_channel_name_is_refused(stale):
    """Contract (plan §2.4, map D-19; 0.6.0): the channels are renamed and `rgbd_pose` is retired, so an old `--channel
    rgb` or a run asking for the depth image is refused rather than mapped to the weakest channel. Regression caught:
    `provides` silently answering for a name that is no channel, or the depth image reaching an agent again."""
    with pytest.raises(ValueError, match="unknown channel"):
        provides(stale)


def test_views_carry_only_their_keys():
    """Contract (plan §2.4): `ask_view` is exactly `qid`, `text`, `answer_type`; `structured_view` adds exactly
    `template` and `slots`; both are deep copies. Regression caught: a gold-side key (`decision_k`, `needs`, `when`)
    reaching a system, or a dict shared with the question row that a system could edit."""
    ask = ask_view(QUESTION)
    assert ask == {"qid": QUESTION["qid"], "text": "Is the kitchen clear right now?", "answer_type": "yes_no"}
    structured = structured_view(QUESTION)
    assert set(structured) == {"qid", "text", "answer_type", "template", "slots"}
    assert structured["template"] == "zone_clear" and structured["slots"] == QUESTION["slots"]
    structured["slots"]["tense"]["hints"].append("rug")
    structured["slots"]["zone"] = "hall"
    assert QUESTION["slots"] == {"zone": "kitchen", "tense": {"now": True, "hints": ["floor"]}}


class Reverser(System):
    """A system that implements only `observe` and `answer` (the qid reversed)."""

    def observe(self, steps):
        pass

    def answer(self, question):
        return {"type": "text", "value": question["qid"][::-1]}


def test_the_default_ask_answers_each_question(tmp_path):
    """Contract (plan §2.4): `System.ask` maps `answer` over the questions, keyed by qid; `finish` returns `{}`;
    `start` keeps its three arguments; `task` does nothing. Regression caught: a baseline forced to reimplement `ask`,
    or a base whose `start` drops the session, robot or channel."""
    system = Reverser(tmp_path / "state", tmp_path / "work", seed=3)
    q1, q2 = {"qid": "abc", "text": "one?"}, {"qid": "xyz", "text": "two?"}
    got = system.ask("end", {"k": 3, "clock": "c"}, [q1, q2])
    assert got == {"abc": {"type": "text", "value": "cba"}, "xyz": {"type": "text", "value": "zyx"}}
    assert list(got) == ["abc", "xyz"]  # the questions' order
    assert system.ask("during", {"k": 1, "clock": "c"}, []) == {}
    assert system.finish() == {}
    assert system.task(0, "08:00", 0.0, "brief") is None
    system.start("0123456789abcdef", {"name": "vacuum"}, "rgb_pose")
    assert (system.session, system.robot, system.channel) == ("0123456789abcdef", {"name": "vacuum"}, "rgb_pose")
    assert system.opts == {"seed": 3} and system.public == tmp_path / "work" / "public"


def test_a_system_runs_on_every_channel_and_is_not_privileged_by_default():
    """Contract (plan §2.4): the base's flags are `channels` (all the channels), `privileged` and `structured` false;
    the harness reads them from the class. Regression caught: a default that excludes a channel (a stale name left
    over from the old `oracle` and `rgb`) or that hands `identity` to every system."""
    assert set(System.channels) == {"rgb_pose", "detections"}
    assert System.privileged is False and System.structured is False


GOOD = {"start": {"ok": True}, "task": {"ok": True}, "observe": {"ok": True}, "ask": {"answers": []},
        "finish": {"ok": True, "usage": {}}}
WRONG_OP = {"start": {"answers": []}, "task": {"answers": []}, "observe": {"answers": []}, "ask": {"ok": True},
            "finish": {"answers": []}}   # another op's good reply


@pytest.mark.parametrize("op", OPS)
def test_reply_checks_per_op(op):
    """Contract (plan §2.4, map §HA5): the good reply is `ok` true for start, task and observe, an `answers` list for
    ask, and `ok` true with a `usage` dict for finish; an `error` reply (even beside a good one), a non-object, another
    op's reply and anything else is refused with a reason. Regression caught: a malformed reply scored as an answer."""
    assert check_reply(op, GOOD[op]) is None
    refused = [{"ok": False}, {"ok": "yes"}, {"ok": 1}, {}, {"error": "boom"}, {**GOOD[op], "error": "boom"},
               None, [], "ok", {"answers": {}}, WRONG_OP[op]]
    for reply in refused:
        reason = check_reply(op, reply)
        assert isinstance(reason, str) and reason, (op, reply)
    assert "boom" in check_reply(op, {"error": "boom"})


def test_a_finish_reply_needs_a_usage_dict():
    """Contract (plan §2.4): `finish` needs `ok` true and a `usage` dict, so a reply without one is refused, while an
    empty dict (nothing reported) is good. Regression caught: usage silently null for a reply that never had any."""
    assert check_reply("finish", {"ok": True}) is not None
    assert check_reply("finish", {"ok": True, "usage": None}) is not None
    assert check_reply("finish", {"ok": True, "usage": []}) is not None
    assert check_reply("finish", {"ok": True, "usage": {"tokens_in": None}}) is None


def test_a_reply_to_an_unknown_op_is_a_bug_not_a_bad_reply():
    """Contract: `check_reply` raises ValueError for an op outside `OPS` (the harness never sends one), instead of
    judging its reply. Regression caught: a mistyped op name that every reply passes."""
    with pytest.raises(ValueError, match="unknown op"):
        check_reply("answer", {"answer": {}})


def test_failed_answers_are_one_shape():
    """Contract (plan §2.4): a question the harness could not get an answer for is `{"type": "failed", "reason": ...}`,
    which the scorer reads as a malformed answer (credit 0) under every answer type, never as an answer. Regression
    caught: two failure markers, one of which the scorer reads as an answer (a `failed` that parses as `unknown`, or as
    a value, would credit a question the harness never got an answer for)."""
    assert failed("timeout") == {"type": "failed", "reason": "timeout"}
    for answer_type in (None, *TYPES):
        assert isinstance(parse_answer(failed("timeout"), answer_type), Malformed), answer_type


def test_timeout_keys_name_each_message():
    """Contract (plan §2.4): a message's timeout key is its op, and `ask_during` or `ask_end` for an ask, and each is a
    key of `TIMEOUTS`; an ask without its `when`, another op with one, and a name that is no op are refused. Hand table.
    Regression caught: an end ask timed with the during ask's 600 s (the end ask carries the whole questionnaire and has
    3,600 s), or a mistyped op that selects no timeout."""
    assert [timeout_key(op) for op in ("start", "task", "observe", "finish")] == ["start", "task", "observe", "finish"]
    assert (timeout_key("ask", "during"), timeout_key("ask", "end")) == ("ask_during", "ask_end")
    assert TIMEOUTS["ask_during"] == 600.0 and TIMEOUTS["ask_end"] == 3600.0
    for op, when in [("ask", None), ("ask", "later"), ("observe", "during"), ("answer", None), ("ask_during", None)]:
        with pytest.raises(ValueError, match="no timeout"):
            timeout_key(op, when)


def test_start_message_by_hand():
    """Contract (plan §2.4): `start_message` is exactly the table's `start` message: the session, the robot, the
    channel, `sends` as the channel's sorted sensors, the two paths and the options, and no protocol name; a name that
    is no channel is refused. Regression caught: a start message without `sends` (an external system left to infer its
    channel from its steps), one that names another key (a protocol or version stamp among them), or one built for a
    stale channel name."""
    robot = {"name": "robot vacuum"}
    assert start_message("0123456789abcdef", robot, "detections", "/w", "/s", {"depth": 3}) == {
        "op": "start", "session": "0123456789abcdef", "robot": robot, "channel": "detections",
        "sends": ["depth", "detections", "pose", "status"], "workdir": "/w", "state_dir": "/s", "opts": {"depth": 3}}
    assert start_message("s", robot, "rgb_pose", "/w", "/s", {})["sends"] == ["pose", "rgb", "status"]
    for stale in ("rgb", "rgbd_pose"):
        with pytest.raises(ValueError, match="unknown channel"):
            start_message("s", robot, stale, "/w", "/s", {})


QUESTIONS = [{"qid": "a"}, {"qid": "b"}, {"qid": "c"}]
ONE = {"type": "unknown"}


def test_read_answers_by_hand():
    """Contract (plan §2.4): from a wire `answers` list, a qid with no answer is "missing", one answered twice is
    "answered twice" (neither copy stands), an answer that is not an object (or an entry without one) is "not an
    object", an answer for a qid that was not asked is ignored, and the rest stand; an in-process `{qid: answer}` dict
    reads the same way but cannot answer twice; any other reply is refused. Hand tables. Regression caught: a malformed
    reply scored as an answer, the second of two answers silently winning, or a late answer to another question
    taken for this one."""
    reply = [{"qid": "a", "answer": ONE}, {"qid": "b", "answer": ONE}, {"qid": "b", "answer": {"type": "count"}},
             {"qid": "zzz", "answer": ONE}, "not an entry", {"answer": ONE}, {"qid": 7, "answer": ONE}]
    assert read_answers(QUESTIONS, reply) == ({"a": ONE}, {"b": "answered twice", "c": "missing"})
    bad = [{"qid": "a", "answer": "unknown"}, {"qid": "b", "answer": None}, {"qid": "c"}]
    assert read_answers(QUESTIONS, bad) == ({}, {"a": "not an object", "b": "not an object", "c": "not an object"})
    assert read_answers(QUESTIONS, {"c": ONE, "a": [ONE], "x": ONE}) == ({"c": ONE}, {"a": "not an object",
                                                                                    "b": "missing"})
    assert read_answers([], [{"qid": "a", "answer": ONE}]) == ({}, {})
    assert read_answers(QUESTIONS, []) == ({}, {"a": "missing", "b": "missing", "c": "missing"})
    for wrong in (None, "answers", 3):
        with pytest.raises(ValueError, match="list of answers"):
            read_answers(QUESTIONS, wrong)
