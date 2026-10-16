"""The brief: the words a system is told about its job, composed at run time (never a file).

The first text of a case is its paragraph: the robot, the place, the case's first assignment, the sensors the channel
delivers, the during-job questions' kinds when the job asks any, and the penalty. Each later task's text is that task's
assignment alone. The brief describes the job and the place, never the exam: it names no object, layout or answer. It
gives the agent no help with the task: no line on what to keep and no vocabulary of the kinds and
colours a question may use. Naming what it sees is part of perception, and the questions' own words are what the agent
gets.

What a job holds, and what this module holds. A job's assignment templates and swap pairs are its own
(`JobSpec.assignments`, `swaps`), so adding a job edits that job's spec and never this module. The place's words per
domain (`DOMAINS`), the sensor words, the during-job phrases and the rules of the three variants are here, all written
now so a later robot's lane edits none of them.

The during sentence (`during_sentence`) is derived from the job's `decisions` and `looks`: it names exactly the kinds of
the during families the job asks, at a decision point or a look point (`DURING_PHRASES`), so the first ask primes no
more than the brief did, and a job that asks none has no such sentence, so the brief promises no question that never
comes. It says nothing of where in the stream the questions come. It is the same bytes in every variant and on both
arms of the leak check (during questions on or off), which differ in the asks alone.

Variants. `correct`: each task's own assignment. `neutral`: no assignment, `NEUTRAL_ASSIGNMENT` for every task.
`swapped`: the assignment of the kind paired with the task's in the job's `swaps` (a path-compatible pair, so the same
stream fits both briefs); a job with no pair refuses it. The agent side never imports this module.
"""

from __future__ import annotations

from .answers import PROMPT_SHAPES
from .job_registry import JobSpec
from .stream_protocol import CHANNELS
from .timeline_ego import TaskPlan

DOMAINS = {"home": "one floor of a furnished home", "warehouse": "an aisle-and-rack warehouse",
           "delivery": "a few streets of a town"}
DURING_PHRASES = {"path_ahead": "what is in view around the robot", "out_of_view": "what is out of view",
                  "where_was_i": "where the robot has been", "uncovered": "what is left of the task",
                  "rear_clear": "whether a manoeuvre is clear", "zone_clear": "whether a manoeuvre is clear",
                  "ego_now": "where a thing you can see is from your camera",
                  "ego_mem": "where a thing you saw before is from your camera",
                  "way_to": "which way to set off to reach a thing you saw before",   # 0.11: names no thing
                  "clear_path": "how far the robot could move",
                  "open_heading": "which way the robot has the most room",
                  "person_mem": "where a thing you saw before is from your camera",   # `ego_mem`'s: names no people
                  # 0.10.0: the harm families; like `person_mem`, none names a person
                  "closest_approach": "how close something would come to the robot",
                  "approach_mem": "how close something would come to the robot",   # 0.11.0: the twin's phrase, deduped
                  "drop_zone_people": "what stands near the point below the robot",
                  "tangle_ahead": "what lies on the floor near the robot"}
DURING_SENTENCE = "At some points during the job you will be asked {kinds}."
NEUTRAL_ASSIGNMENT = "It is carrying on with its usual work."
VARIANTS = ("correct", "swapped", "neutral")
SENSOR_WORDS = {"rgb": "camera images", "depth": "depth", "detections": "detections of the things in view"}
OWN_WORDS = ("pose", "status")   # the robot's own readings, named together: "the robot's own pose and status"
PARAGRAPH = ("You are the onboard assistant of a {robot} working on {place}. {assignment} You will receive "
             "timestamped {sensors} as it works.\n\n{during}{after}Use only evidence available so far. Answer unknown "
             "when it is insufficient: a wrong answer costs more than unknown.")
AFTER_SENTENCE = "After the job you will answer structured questions about the environment and its history."


def _join(parts: list[str]) -> str:
    """"a", "a and b", "a, b, and c"."""
    if len(parts) < 3:
        return " and ".join(parts)
    return ", ".join(parts[:-1]) + ", and " + parts[-1]


def during_families(job: JobSpec) -> set[str]:
    """The during families `job` asks: those of its `decisions` and of its `looks` (asked at look points)."""
    return {family for families in job.decisions.values() for family in families} | set(job.looks)


def asks_during(job: JobSpec) -> bool:
    """Whether `job` asks any family during the job (its `decisions` list one, or it has `looks`): what
    `during_sentence` and the Codex prompt's mention of questions during the job are both derived from, so neither
    promises a question that never comes."""
    return bool(during_families(job))


def asks_end(job: JobSpec) -> bool:
    """Whether `job` asks any family after the job (its `ends`; None is the questionnaire's default, which is not
    empty): what the Codex prompt's mention of questions after the job is derived from, so it promises none that never
    comes (the drone asks only at look points)."""
    return job.ends is None or bool(job.ends)


LIST_FAMILIES = ("drop_zone_people", "tangle_ahead")   # 0.10.0: the during families whose answer is a list of places


def asks_lists(job: JobSpec) -> bool:
    """Whether `job` asks a list of places: `task_things` after the job or one of `LIST_FAMILIES` during it. What the
    list bullets of the prompt's credit section (`llm_credit`) are derived from, so it speaks of lists only for a job
    that asks one. The vacuum asks both since 0.11.0 (its `task_things` is the list of what its brush could catch, and
    `tangle_ahead` comes during the job), so either alone gives it the list paragraph."""
    return "task_things" in (job.ends or ()) or bool(set(LIST_FAMILIES) & during_families(job))


def list_hazards(job: JobSpec) -> tuple[str, ...]:
    """The `LIST_FAMILIES` that `job` asks during the job, in their order: what the credit section's list catastrophe
    (`llm_credit.LIST_CRITICAL`, the omission of a critical person or cord) is derived from, so it speaks of one only
    for a job that asks the family. Empty for the helper, the AMR (their list is `task_things`, after the job) and the
    drone since it stopped asking `drop_zone_people`; the vacuum's is `tangle_ahead` alone (its `task_things` is asked
    after the job and carries no catastrophe)."""
    asked = during_families(job)
    return tuple(family for family in LIST_FAMILIES if family in asked)


def asks_headings(job: JobSpec) -> bool:
    """Whether `job` asks `open_heading` during the job: what the prompt's heading definition and regret bullets
    (`llm_credit`) are derived from, so it speaks of headings only for a job that asks one."""
    return "open_heading" in during_families(job)


def asks_turns(job: JobSpec) -> bool:
    """Whether `job`'s `clear_path` may ask a turned move (left or right, `JobSpec.clear_directions`): what the
    prompt's turned-move bullet (`llm_credit.CORRIDOR_TURN`) is told for. A flier asks ahead and back only."""
    return "clear_path" in asked_families(job) and bool({"left", "right"} & set(job.clear_directions))


def asked_families(job: JobSpec) -> tuple[str, ...]:
    """The families `job` asks, sorted: each family of its `decisions` and `looks` during the job and each end family
    after it (`questionnaire.asked_ends`, the default for a job that names none). What `answer_types` and the credit
    section's family-specific bullets (`llm_credit.credit_sentence`'s `families`: the corridor, a person's distance, a
    person's offset) are derived from. ValueError for a family the registry lacks."""
    # imported here: the registry loads `questions_adjacent`, which imports `compose` from this module
    from .questionnaire import asked_ends
    from .questions_text import FAMILIES
    asked = during_families(job) | set(asked_ends(job))
    if missing := sorted(asked - FAMILIES.keys()):
        raise ValueError(f"job {job.id} asks {', '.join(missing)}, which the family registry lacks")
    return tuple(sorted(asked))


def answer_types(job: JobSpec) -> tuple[str, ...]:
    """The answer types `job` asks, in `answers.PROMPT_SHAPES`' order and without `unknown` (every prompt lists that
    one): the `answer_type` of each family it asks (`asked_families`). What the system prompt's list of answer shapes
    and the credit section's type-specific bullets are derived from (`llm_render.system_prompt`'s `kinds`), so they show
    a job the shapes and rules of the questions it asks and no others. ValueError for a family the registry lacks."""
    from .questions_text import FAMILIES
    kinds = {FAMILIES[family].answer_type for family in asked_families(job)}
    return tuple(kind for kind in PROMPT_SHAPES if kind in kinds)


def during_sentence(job: JobSpec) -> str:
    """The sentence naming the kinds of during question `job` asks (the families its `decisions` and `looks` list), each
    phrase once and in the order of `DURING_PHRASES`; "" for a job that asks none. ValueError for a family with no
    phrase."""
    asked = during_families(job)
    if missing := sorted(asked - DURING_PHRASES.keys()):
        raise ValueError(f"job {job.id} asks {', '.join(missing)} during the job; DURING_PHRASES has no words for it")
    phrases = list(dict.fromkeys(phrase for family, phrase in DURING_PHRASES.items() if family in asked))
    return DURING_SENTENCE.format(kinds=_join(phrases)) if phrases else ""


def _sensors(channel: str) -> str:
    """The channel's sensors in words, in a fixed order: pictures, depth, detections, then the robot's own readings."""
    if channel not in CHANNELS:
        raise ValueError(f"unknown channel {channel!r}; the channels are {', '.join(CHANNELS)}")
    sends = CHANNELS[channel]
    parts = [SENSOR_WORDS[name] for name in SENSOR_WORDS if name in sends]
    own = [name for name in OWN_WORDS if name in sends]
    if own:
        parts.append("the robot's own " + " and ".join(own))
    return _join(parts)


def _partner(job: JobSpec, kind: str) -> str:
    for a, b in job.swaps:
        if kind in (a, b):
            return b if kind == a else a
    raise ValueError(f"job {job.id} pairs no task kind with {kind!r}, so it has no swapped brief")


def _assignment(plan: TaskPlan, job: JobSpec, variant: str) -> str:
    if variant == "neutral":
        return NEUTRAL_ASSIGNMENT
    kind = _partner(job, plan.kind) if variant == "swapped" else plan.kind
    if kind not in job.assignments:
        raise ValueError(f"job {job.id} has no assignment text for the task kind {kind!r}")
    try:
        return job.assignments[kind].format(**plan.params)
    except KeyError as missing:
        raise ValueError(f"the {kind!r} assignment of job {job.id} names {missing}, which the {plan.kind!r} plan "
                         "does not give") from None


def compose(plans: list[TaskPlan], *, job: JobSpec, robot: dict, domain: str, channel: str,
            variant: str = "correct") -> list[str]:
    """One text per task of a case's window, in order: the paragraph with assignment 0, then each later task's
    assignment alone. `robot` is `robot.json`'s dict (its `name`), `domain` one of `DOMAINS`, `channel` one of
    `stream_protocol.CHANNELS` and `variant` one of `VARIANTS`. ValueError for any other, for no tasks, for a task kind
    the job has no text for, and for `swapped` on a kind the job pairs with none."""
    if variant not in VARIANTS:
        raise ValueError(f"the brief variants are {', '.join(VARIANTS)}, not {variant!r}")
    if domain not in DOMAINS:
        raise ValueError(f"the domains are {', '.join(DOMAINS)}, not {domain!r}")
    if not plans:
        raise ValueError("a brief needs at least one task")
    assignments = [_assignment(plan, job, variant) for plan in plans]
    during = during_sentence(job)
    first = PARAGRAPH.format(robot=robot["name"], place=DOMAINS[domain], assignment=assignments[0],
                             sensors=_sensors(channel), during=during + " " if during else "",
                             after=AFTER_SENTENCE + " " if asks_end(job) else "")
    return [first, *assignments[1:]]
