# Vision

## The goal

Robots are getting LLM and agent brains. A brain that runs a robot has to perceive and remember the space it works in,
using only what the robot itself has.

SpatiotemporalBench measures how well an agent does that. Given only the robot's camera and what its controller knows
about itself (its pose, the clock, its task), can the agent:
- estimate where things are relative to the robot;
- understand the space it is in;
- remember that space, and how it changed over time;
- anticipate how it will change;
- pick out the things that matter for its task;
- and know when it does not know?

## Success

- **The score ranks agents by that ability and nothing else.** Not by prompt tricks, guessing, or facts that can be
  learned without the sensors.
- **The score predicts control.** Agents that score higher run the robot better when they choose its actions. This is
  checked on a closed-loop set before anyone claims it.
- **It is cheap and fast to run.** Running an agent is the only cost, and a run takes minutes, not hours.
- **It is open and trusted.** Code, data and results are public, and others can build on them.

## Principles

- **Only what the robot would have.** The agent gets the robot's own sensors and self-knowledge: no extra sensors,
  labels, maps or helper tools.
- **The truth decides.** Every answer is checked by code against what really happened and what the robot could have
  seen. No person or model writes or judges an answer.
- **Scores need the sensors.** Nothing can be answered from the question's wording or the robot's own log, and
  nothing but a place's fixed structure from other cases or a copy of the place. The agent is never given a map, so
  the fixed structure around the robot (its walls, racks, buildings and furniture, and how far the robot could move
  before meeting them) is perceived and remembered like anything else, and is asked. A system that holds the
  place's layout can answer those questions without the sensors, and only a held-out place makes them a claim
  about every system. The map reader, which does so, is an internal design diagnostic in `stbench check`, not a
  published result.
- **Hard because real jobs are hard.** The accuracy asked for, the things the robot meets and how long it works come
  from real robots and real jobs. Never trick questions, and never a tolerance tuned to make a model fail.
- **Perception and memory count equally.**
- **Agents are tested as they come.** An agent may bring any memory or tools of its own. Only its answers count, and
  what it spent is reported beside them.
- **Results are reported as they are,** with their uncertainty, baselines, cost and failures.
- **When choices conflict:** valid, then simple, then realistic, then fast.

## How questions are made and scored

**Every question is a decision the robot's brain faces in its job.** It asks about an action and its consequence, never
about a thing for its own sake. Together the questions cover four kinds of knowledge:
- **Harm:** what an action now would hit, tangle on or endanger.
- **Help:** which action now best moves the task forward, such as the best heading, route or approach.
- **Situation:** what is around the robot now, in view and out of view, and how it has changed.
- **Reuse:** what is worth remembering to do the job here again: where things are kept, which ways are clear, what
  stays put and what moves.

The strongest questions need all three of what is in view now, what was seen before, and whether it has changed since.
Some ask about actions toward space the robot cannot see now, which only memory answers. A question asks about the
world and actions, never about a sensor's raw output, so the same question fits any set of sensors. Answers are open:
a distance, a heading, a route, a place or a set, never a multiple choice.

**Credit follows the consequence for the task.**
- **Credit is layered.** The right direction earns some credit: the side, the order, blocked or clear. The right thing
  or action earns more, and the right amount within what the job needs earns most. Full credit is kept for precision at
  the limit of what the sensors can resolve.
- **Credit falls off smoothly,** with no flat top, so perfect answers are rare while roughly right ones still score.
- **An error that would cause harm costs more than a cautious one.** Overstating clearance risks a crash;
  understating it only wastes time. The worst such errors, one that would drive into something the robot saw, onto a
  person or into a tangle, are charged a flat penalty in the reported score and in no validity check
  ([question-design.md](question-design.md), "How answers are graded").
- **A decision scores by how close its outcome comes to the best one.**
- **A set scores per item.** It is perfect only when it holds every right item and nothing else.
- **`unknown` is right when the sensors never showed the answer.** A confident wrong answer costs more than `unknown`.
- **Full credit is hard but always reachable.** An ideal reader of the same sensors, with perfect memory, scores 1.

## Speed

A benchmark that takes hours to run gets run rarely. Cases run as fast as the agent allows: in parallel, with the
short and medium tiers runnable in minutes, and checks of a change on a small subset first. Harder questions replace
easy ones. A case may ask more questions: the frames it streams are its fixed cost, and a question costs far more
than a frame, so a case asks as many fair questions as its frames support within its time
([question-design.md](question-design.md), "When questions are asked", has the measured costs and density). Free
checks and validations are kept fast too.

*Amended 2026-10-14, for more questions per case.* The sentence it replaces read "Harder questions
replace easy ones; they never add questions or time to a case". The rest of this section stands.

*Amended 2026-10-16.* The principle "Scores need the sensors" read "a map reader's score is
published beside them". It is not: the agent is never given a map, and how a system that holds one scores is not what
is tested, which is the agent's spatiotemporal perception and memory. The map reader stays in `stbench check` as a
design diagnostic, and no gate changes.

## What it never measures

Control skill, free text, or how a memory system works inside.

[methodology.md](methodology.md) holds the rules that turn these principles into questions, scores and checks.
