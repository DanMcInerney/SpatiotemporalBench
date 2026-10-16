"""The answer types the reference agent's prompt shows (`brief.answer_types`, `llm_render.system_prompt`'s `kinds`):
those of the families the job asks and `unknown`, in the order of `answers.PROMPT_SHAPES`, and no others (2026-10-14:
the prompt listed every type, most of which a job never asks). Hand-written sets; no model, no network."""

from __future__ import annotations

import pytest

from codex_kit import Fake, Run, job_also_asks
from stbench import job_registry
from stbench.answers import PROMPT_SHAPES, TYPES
from stbench.brief import answer_types
from stbench.llm_render import ask_message, labelled, listed, system_prompt
from stream_kit import fake_job
from test_cmd_codex import run, small_case
from test_codex_system import COUNT, NUMBER, POSITION, coaching
from test_llm_render import public

# job id -> the types of the families it asks, by hand from the family table (`questions_*`'s `FamilySpec.answer_type`)
# in the order of `PROMPT_SHAPES`. vacuum/1 asks `object_where` (the default end family) and nothing during the job;
# vacuum/2 `object_where` after the job and `clear_path`, `tangle_ahead`, `ego_mem` and `open_heading` at look points;
# the helper `object_where` and `task_things` after and `clear_path`, `closest_approach`, `ego_mem`, `ego_now` and
# `open_heading` at look points; the AMR `task_things` and `closest_person` after and `closest_approach`, `ego_mem` and
# `ego_now`; the drone no end family and `clear_path` and `ego_mem` at look points.
JOB_TYPES = {
    "vacuum/1": ("position",),
    "vacuum/2": ("position", "offset", "number", "positions", "heading"),
    "helper/1": ("position", "offset", "number", "positions", "heading"),
    "amr/1": ("offset", "number", "positions"),
    "drone/1": ("offset", "number")}


def shapes_in(prompt: str) -> list[str]:
    """The answer types whose shape line the prompt shows, in order (a line `- <type>: <shape>`)."""
    names = {*TYPES, "unknown"}
    return [line[2:].split(":")[0] for line in prompt.splitlines() if line.startswith("- ") and
            line[2:].split(":")[0] in names]


@pytest.mark.parametrize("job_id", JOB_TYPES)
def test_the_prompt_of_each_registered_job_shows_the_types_of_the_families_it_asks_and_unknown(tmp_path, job_id):
    """Contract (2026-10-14): the answer shapes in a job's system prompt are exactly the shapes of the types of the
    families the job asks (its `looks`, its end families or the questionnaire's default, and its decision families),
    then `unknown`, in the order of `PROMPT_SHAPES`, each line the table's; the helper finds the same types from the
    job. Hand sets for the five registered jobs, so the drone is shown no position and the AMR none either (its
    positions are a list). Regression caught: every type listed to every job (the cells and route of a family no job
    asks), a type of a family the job asks left out so that its form is never taught, or a list in the wrong order."""
    job = job_registry.get(job_id)
    expected = JOB_TYPES[job_id]
    assert answer_types(job) == expected
    prompt = system_prompt(public(tmp_path), "rgb_pose", kinds=answer_types(job))
    assert shapes_in(prompt) == [*expected, "unknown"]
    for kind in (*expected, "unknown"):
        assert f"- {kind}: {PROMPT_SHAPES[kind]}" in prompt
    assert coaching(prompt) == []


def test_the_types_follow_from_the_decision_look_and_end_families_and_the_default_end():
    """Contract: `brief.answer_types` reads the `answer_type` of each family of a job's decision kinds and looks (during
    the job) and of its end families after it, where a job that names none (`ends` None) asks the questionnaire's
    default `object_where` and one with `ends=()` asks none; each type once, in the order of `PROMPT_SHAPES` whatever
    the order of the families, and `unknown` is never in it. Hand cases on fake jobs: `path_ahead` and `rear_clear`
    are numbers, `out_of_view` positions, `ego_mem` an offset, `when_seen` a time, `order` a yes or no. Regression
    caught: an end family left out (the AMR's `task_things`), the default end forgotten, a family counted twice, a list
    in the order the families were written, or a family the registry lacks passing silently."""
    assert answer_types(fake_job(looks=("ego_mem",))) == ("position", "offset")
    assert answer_types(fake_job(looks=("ego_mem",), ends=())) == ("offset",)
    assert answer_types(fake_job(decisions={"before_doorway": ("path_ahead", "out_of_view"),
                                            "room_done": ("rear_clear",)}, ends=())) == ("number", "positions")
    assert answer_types(fake_job(looks=("ego_mem",), ends=("when_seen", "order", "object_where"))) == (
        "position", "offset", "time", "yesno")
    assert answer_types(fake_job(ends=())) == ()
    with pytest.raises(ValueError, match="asks no_such_family, which the family registry lacks"):
        answer_types(fake_job(looks=("no_such_family",)))


def test_the_prompt_lists_the_kinds_given_and_unknown_in_the_order_of_the_shapes(tmp_path):
    """Contract: `system_prompt(kinds=...)` shows the shape of each type named, once, in the order of `PROMPT_SHAPES`
    (not the order named), then `unknown` and the unknown rule; no kinds shows only `unknown`; None, for a caller that
    knows no job, shows every type; a name that is no answer type (`unknown` too: it is always shown) is refused.
    Regression caught: a list in the caller's order, a type shown twice, `unknown` missing from a short list, the
    default dropping types a test or an external harness relies on, or a stray name passing silently."""
    named = system_prompt(public(tmp_path), "rgb_pose", kinds=("heading", "position", "heading"))
    assert shapes_in(named) == ["position", "heading", "unknown"]
    assert shapes_in(system_prompt(public(tmp_path / "none"), "rgb_pose", kinds=())) == ["unknown"]
    assert shapes_in(system_prompt(public(tmp_path / "all"), "rgb_pose")) == [*TYPES, "unknown"]
    assert listed(("count",)) == ("count", "unknown") and listed(None) == (*TYPES, "unknown")
    folder = public(tmp_path / "bad")
    for stray in (("blob",), ("position", "unknown")):
        with pytest.raises(ValueError, match="is no answer type"):
            system_prompt(folder, "rgb_pose", kinds=stray)


def test_a_job_that_asks_a_type_the_prompt_omits_is_refused_before_anything_is_sent(tmp_path):
    """Contract (2026-10-14): the agent is shown the form of every answer type its questions have. An ask holding a
    question of a type the system prompt did not show is refused, by `ask_message` (the types the prompt showed:
    `shown`) and by a `CodexSystem` given the job's `kinds`, which raises before an exec is started or a word is sent;
    with no types given (a caller that knows no job) the ask is as it was. Hand case: a prompt for a job that asks
    positions and counts, an ask of a position, of a count and of a number. Regression caught: a question the agent was
    never taught how to answer, which it then answers in a form the parser refuses."""
    shown = listed(("position", "count"))
    ok = labelled([POSITION, COUNT])
    assert "Answer type: count." in ask_message("during", {"k": 1, "clock": "c"}, ok, shown=shown)
    with pytest.raises(ValueError, match="shows no shape for the answer type number"):
        ask_message("during", {"k": 1, "clock": "c"}, labelled([POSITION, NUMBER]), shown=shown)
    assert ask_message("end", {"k": 1, "clock": "c"}, labelled([NUMBER])).count("Answer type: number.") == 1
    (tmp_path / "kinds").mkdir()
    with Run(tmp_path / "kinds", kinds=("position", "count")) as limited:
        with pytest.raises(ValueError, match="shows no shape for the answer type number"):
            limited.system.ask("during", {"k": 0, "clock": "2026-03-02T08:00:00"}, [NUMBER])
        assert limited.fake.execs() == [] and limited.system.execs == 0


def test_the_driver_refuses_a_case_with_a_question_the_job_does_not_ask(tmp_path, monkeypatch, capsys):
    """Contract (2026-10-14): `stbench run codex:` shows the prompt the types of the job's families only, so a case
    whose questions have another type (compiled from another version of the job: the drone stopped asking
    `drop_zone_people`, a list, and an older dataset still holds them) is refused with exit 2 before anything is
    created, the message naming the type, the job and the types it asks. The during questions count only on the arm
    that asks them (`--during on`): the off arm of such a case asks its end questions and runs. Hand case: the kit's
    six-frame case holds a `number` during question and a `position` end one, and `vacuum/1` asks positions only; the
    same case under a job that also asks a number runs on both arms. Regression caught: a run that reaches the first
    exec and fails at the first ask, one that quietly lists more types than the job asks to make the data fit, or an
    off arm refused for a question it never asks."""
    case = small_case(tmp_path / "data")
    fake = Fake(tmp_path)
    out = tmp_path / "out"
    assert run(monkeypatch, fake, "codex:fake-model", case.private, "--out", out) == 2
    error = capsys.readouterr().err
    assert "answer type number" in error and "job vacuum/1 does not ask (it asks position)" in error
    assert not out.exists() and fake.execs() == []
    assert run(monkeypatch, fake, "codex:fake-model", case.private, "--out", tmp_path / "off", "--during", "off") == 0
    job_also_asks(monkeypatch, "distance_between")
    assert run(monkeypatch, fake, "codex:fake-model", case.private, "--out", tmp_path / "on") == 0


def test_both_arms_of_the_leak_check_are_sent_the_same_prompt_with_the_jobs_shapes(tmp_path, monkeypatch):
    """Contract (the leak check's two arms differ in the asks alone, `brief`, `cmd_codex`): the types shown are the
    job's and not the run's `--during` arm's, so the first message of a run with the during questions on and the one
    with them off are the same bytes, and both show the shapes of the types the job asks and no other; the credit
    section of that prompt is the one for the job's answer types and families (the driver gives it both, `cmd_codex`):
    a place and a distance, and nothing of an offset, a corridor or a person. Hand case: the kit's case under `vacuum/1`
    made to also ask `distance_between`, whose families are a position and a number. Regression caught: shapes derived
    from the questions the arm asks, which would show the off arm fewer shapes than the on arm and make the check
    measure the prompt, or a driver that gives the section no kinds and families, so that every job is told every
    rule."""
    job_also_asks(monkeypatch, "distance_between")
    case = small_case(tmp_path / "data")
    prompts = {}
    for arm in ("on", "off"):
        (tmp_path / arm).mkdir()
        fake = Fake(tmp_path / arm)
        assert run(monkeypatch, fake, "codex:fake-model", case.private, "--out", tmp_path / f"out-{arm}",
                   "--during", arm) == 0
        (only,) = fake.execs()
        prompts[arm] = only["prompt"]
    assert prompts["on"] == prompts["off"]
    assert shapes_in(prompts["on"]) == ["position", "number", "unknown"]
    prompt = prompts["on"]
    for told in ("A place in the map, alone or in a list", "For a distance in metres: on the same side",
                 "The margin is 1 m for a place.\n- For a distance in metres, the margin is 10 cm or a tenth of the "
                 "right distance (the right number of metres), whichever is larger."):
        assert told in prompt
    for not_told in ("An offset goes to the point", "How far the body could move", "Going into a person",
                     "Going into something the robot had seen", "A list of places earns"):
        assert not_told not in prompt
