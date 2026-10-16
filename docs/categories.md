# Question categories

How a question is authored. A category is one decision a robot's brain faces, with a gold template over the timeline and
the visibility log, an answer scored on the one graded curve, and a list of the ways it can be answered without the
sensors. A new job, or a new place for an old one, is written by saying which categories it faces and filling in each
one's hooks; the questions follow from the template. A family is written only for a motion, objective or predicate the
existing gold cannot express. Code: `stbench.categories` (`CATEGORIES`, `FAMILY_CATEGORY`); `tests/test_categories.py`
holds this page, that map and the family registry to each other. The reasons are in
[question-design.md](question-design.md), the record and gold mechanics in [questions.md](questions.md) and the rules in
[methodology.md](methodology.md). Each family has a page in [families/](families/), whose first row names its category.

**Three splits of the same families.** The *ability* and the scope make the headline's cells (methodology rule 4). The
*kind* (harm, help, situation, reuse) is a split of the reports (`abilities.FAMILY_KIND`). The *category* is the
authoring unit. They do not line up: C1 and C2 fill clearance; C3 fills distance (`now`) and memory; C4 fills clearance
(the room) and memory (the goal); C5 fills relevance and clearance (`tangle_ahead`); C6 would fill change.

## The two rules

1. **The robot is on one side of every relation** (the one-side rule). A category relates a thing, a person, space or
   a set to the robot's body, its motion or its task point (the drop, the way it drove). A relation between two things
   for its own sake (VSI-Bench's relative direction and absolute distance, MindCube's views) is never asked: no job
   acts on it, and the only hardness it adds is error compounding.
2. **Complexity is composition, never a riddle** (the composition rule). A question composes some of five steps, each
   a thing a real controller does:
   - (a) **perceive** range and bearing from one monocular frame;
   - (b) **remember** something now out of view;
   - (c) **carry the robot's own motion** since then (the pose is given, the picture is not);
   - (d) **carry a mover's motion** (two sightings make a velocity, extrapolated over the job's horizon);
   - (e) **compare across a gap** (the world changed between tasks).

   A category's difficulty is the number of steps it composes. A tighter tolerance, a longer gap or a count is not a
   step ([methodology.md](methodology.md) section 4; [question-design.md](question-design.md) section 3). On 0.10.0's
   answers neither the age of the evidence nor the amount of own motion made a row harder (one model's reported score by
   age band: +0.65 now, +0.64 recent, +0.49 task, +0.62 earlier); the kind of composition did: a heading, a list, a
   remembered mover.

## The six categories

| # | Category | Decision it serves (kind) | Steps | Answer | Families asked today |
| --- | --- | --- | --- | --- | --- |
| C1 | Room to move | how far can my body go if I make motion m (harm) | a; b and c by motion | a number, graded on C | `clear_path` |
| C2 | Separation | how close would (did) a person come if we both keep moving (harm) | a, c, d; b in the memory variant | a number, graded on C | `closest_approach`, `approach_mem`, `closest_person` |
| C3 | Locate | where is the thing: from my camera now, as last seen, or on the map (situation, reuse) | a (`ego_now`); a, b, c (`ego_mem`); a, b (`object_where`) | an offset graded on C, or a place graded on R | `ego_now`, `ego_mem`, `object_where` |
| C4 | Which way | which heading to take, for room or to reach a remembered thing (help) | b, c (`open_heading`); a, b, c (`way_to`) | a heading, graded by its regret | `open_heading`, `way_to` |
| C5 | What matters here | which things meet my job's predicate (harm during, reuse at the end) | a, b, a selection | a set of places, each graded on M | `task_things`, `tangle_ahead` |
| C6 | Change | where is the thing I remember now that it moved between tasks (reuse) | a, b, e | a place graded on R | none: gated on data |

**Retired** means no job asks the family: it keeps its spec, its code and its page, and the registry knows it.

**One curve.** Every metric answer is credited the same way ([question-design.md](question-design.md) section 6,
[questions.md](questions.md) "Credit"): four quarters (direction, thing, amount, precision) on the need `N' = max(N, 0.1
g, 2 rho)`, where `N` is the job's own margin (the clearance C, the re-find radius R or a list's match radius M), `g` the
distance judged and `rho` what one pixel resolves. The first two terms are halved on the harmful side. An answer that
would end in harm is a catastrophe, charged in the reported score only. A category's entry below names its `N`, its
harmful side and its catastrophe, and nothing else about scoring. `unknown` is right where the sensors never showed the
answer, and a wrong answer costs more than `unknown`.

**A gate catches each risk.** The gates are `blind`, `form`, `questionnaire_blind`, `path_log`, `other_case`,
`memory_needed` (on `memory` rows, `current_view` and `current_view_prior`), `separation`, the balance rule and
`all_unknown`, each with an in-pool twin where it learns from other scripts ([balance.md](balance.md),
`stbench.gates`). A family that fails a gate leaves that job, by rule; no constant is re-tuned.

## C1 Room to move

- **Decision.** How far can my body go before it touches anything if I make motion m (reverse, turn in place and drive,
  fly ahead)? Harm.
- **Steps.** (a); (b) for the space behind or beside; (c) when the motion turns away from the view.
- **Gold.** Over the free-space grid, each cell's latest state in the visibility log up to the ask k (`grid.latest`),
  the robot's swept cells marked free. Sweep the body's footprint (a flier's height band above `fly_floor_m`) along the
  motion and stop at the first contact (`grid.corridor_clear`, `clear_profile`). A run that reaches never-seen space
  before any contact is an `unknown` gold: empty space must be seen empty (methodology rule 2). The cap is the job's.
- **Answer, credit.** Metres from the body's front or back edge, up to the cap. `N` = C. The harmful side is an answer
  longer than the gold. Catastrophe `over`, weight 2: the true corridor is under the body's length plus C and the answer
  is not.
- **Dead bands.** A phase in which the robot is not driving (`clear_phases`; the status answers), and a flier below
  `fly_floor_m`.
- **Risks, gates.**
  - The corridor is fixed structure, so a system that holds the map answers it. The map reader is a reference, not a
    gate: the one exception to methodology rule 5, so the drone's score here is a claim only about a system with no
    copy of the map. Report it; never tighten the tolerance.
  - A decision frame puts the question at the same plan pose in every script (`memory_needed` read +0.211 on the
    helper's decision-frame rows until `before_place` left the draw). Jitter approach poses per script (R7).
  - A constant per direction (`blind`, balance; the vacuum's right turn scored +0.28) and the robot's own swept path
    (`path_log`: the corridor behind it).
- **`JobSpec` hooks.** `clear_directions`, `clear_phases`, `fly_floor_m`, `clearance`, slot 1 of `look_shares`,
  `look_unknown_keep`, `look_decisions`.
- **Families.** `clear_path`. Retired: `path_ahead`, `rear_clear`, `zone_clear`, `clearance_then`.

| Robot | Example |
| --- | --- |
| Floor (vacuum) | "If you reversed straight back now, how far could your body go before it touched anything?" |
| Manipulator (helper) | "If you turned in place to face your left and then drove straight, how far could your body go before it touched anything?" Data round: an arm sweep (needs the arm in the timeline and the render) |
| AMR | Not asked: 62% of the ahead golds are the 5 m cap and a held-out constant scores +0.33. Data round: pallets drawn per run to jut into the aisles |
| Drone | "If you flew straight ahead now at this height, how far could your body go before it touched anything you have seen?" Data round: the descent, with content under the drop drawn per flight |

## C2 Separation

- **Decision.** How close would a person come to me in the next 5 s if I keep my velocity and they keep theirs, or how
  close did one come during a task? Harm.
- **Steps.** (a); (c) the robot's own velocity; (d) the person's; (b) in the memory variant.
- **Gold.** `approach_rule.closest_m`: the least floor distance over the horizon H = 5 s between the point below the
  camera, moving at the velocity the robot delivers, and the person's hull, moving at the velocity two evidence
  sightings within `VEL_S` give (zero for a stander); capped at 10 m. The memory variant takes the latest evidence
  sighting before the ask (`approach_rule.last_seen`). The audit variant takes the least sighted distance over a task.
- **Answer, credit.** Metres, capped. `N` = C. The harmful side is an answer longer than the gold. Catastrophe `over`,
  weight 3 (a person).
- **Dead bands.** Two people at a sighting (the sensors give no track identity); a person the frame shows (256 px or
  more) and does not place as evidence (a sighting needs 1,024 px, light and a box clear of the bottom edge); a walker
  last seen walking, in the memory variant (no reader can say where they went); a person who moved unseen (a truth
  screen, so a catastrophe is charged only for a person who was there).
- **Risks, gates.**
  - The "no one" cap answers many golds: the draw keeps a share of them (balance).
  - A small walk area against the need makes any sensorless answer about where people are good (`blind`, and the
    pose-keyed pools of other scripts' golds: `path_log` and `current_view_prior`, with their in-pool twins).
  - People mostly stand (the home's resident about 97% of the time), so step (d) is rare: a supply fact, not a gate. A
    prediction must beat "nothing moves" (methodology rule 5).
- **`JobSpec` hooks.** The people file of the place (`people_<catalog>.py`, [people.md](people.md)), `looks` and
  `look_shares`, `ends` with `closest_person`, `clearance`. The share of "no one"
  golds kept is the family's own constant.
- **Families.** `closest_approach`, `approach_mem`, `closest_person`.

| Robot | Example |
| --- | --- |
| Floor (vacuum) | None: no person is placeable from an 8 cm camera |
| Manipulator (helper) | "If you kept moving as you are now, how close would the person you can see nearest you come in the next 5 seconds?" (`closest_approach`), and for the resident last seen behind it as it reverses (`approach_mem`) |
| AMR | The picker seen 20 s ago at a blind aisle end, now behind the rack (`approach_mem`); "how close did a person come during your first task?" (`closest_person`) |
| Drone | Not asked: at altitude a person is not the contact hazard. Data round: "before you lower, how close is the person nearest the point below you, as you last saw them?" |

## C3 Locate

- **Decision.** Where is the thing: from my camera now (`ego_now`), from my camera now as I last saw it (`ego_mem`), or
  on the map, to find it again (`object_where`)? Situation, and reuse for the map.
- **Steps.** (a) in view; (a), (b), (c) for `ego_mem`; (a), (b) and the pose at the sighting for a place.
- **Gold.** The named thing's last recognisable sighting in the window (enough pixels, a box clear of the image edge,
  light: `recognise`), held only while every later glimpse lies within `MOVED_NO_M` of it (`World.steady`). An offset is
  that sighting's nearest visible point, put in camera k's floor frame through the delivered poses; it is the
  displacement [Ego4D's VQ3D](https://arxiv.org/abs/2110.07058) asks, with the robot's own poses given. A place is the
  thing's middle in map coordinates. `ego_mem` and `object_where` draw 20% of their questions from the `unknown` pool (a
  real thing never shown in the window, or a description matching nothing here). `World.told` excludes the task's own
  target.
- **Answer, credit.** An offset in metres ahead and to the left, `N` = C, the harmful side a thing placed farther than it
  is; or a place, `N` = R with no harmful side, where the room the thing is in earns nothing (methodology rule 7).
- **Dead bands.** A thing seen only under the recognition floor; a thing a look-alike hides; for the memory variants, a thing in
  the recent view (the last 8 frames).
- **Risks, gates.**
  - A kind's typical range answers `ego_now`: `kind_range` is its declared reference, reported above, not a gate.
  - A kind that recurs at one spot is a layout prior for `object_where` (`blind_in_pool`, `path_log_in_pool`).
  - A name that leaks a thing never shown (the colour and look-alike screens), and an `unknown` that the question's form
    predicts (`form`).
- **`JobSpec` hooks.** `looks` and `look_shares` (slot 2 for `ego_mem`), `ends` (`object_where`), `clearance`,
  `reacquire` (R), `screen_fallback`.
- **Families.** `ego_now`, `ego_mem`, `object_where`. Retired: `person_mem`, `person_where`, `out_of_view`, `when_seen`,
  `order`, `distance_between`, `where_was_i`.

| Robot | Example |
| --- | --- |
| Floor (vacuum) | "As you last saw it, where is the point of the floor fan you saw that is nearest to your camera now?" (`ego_mem`); "Where did you last see the red bag?" (`object_where`) |
| Manipulator (helper) | "Where is the point of the grey remote you can see now that is nearest to your camera?" (`ego_now`), the same as last seen (`ego_mem`), and "Where did you last see the grey remote?" |
| AMR | The blue tote in view (`ego_now`) and as last seen (`ego_mem`) |
| Drone | "As you last saw it, where is the point of the white van you saw that is nearest to your camera now?" (`ego_mem`) |

## C4 Which way

- **Decision.** Which heading do I take: for the most room (`open_heading`), or toward a thing I saw before (`way_to`)?
  Help.
- **Steps.** (b), (c) on structure for the room; (a), (b), (c) on a thing and the layout between for the goal.
- **Gold.** An outcome for each of 360 headings: the run of the corridor turned to that heading (`clear_profile`), or how
  much nearer one step D = R of driving along it brings the robot to the goal, by the shortest way its body fits through
  floor it has seen or swept (`route_field`). A heading with a run under T = L + C, or into never-seen floor, has no
  outcome. `way_to` draws 20% of its questions from C3's `unknown` pool; `open_heading` has no `unknown` gold.
- **Answer, credit.** A heading in degrees to the left, -180 to 180. Credit is the regret (the best outcome less the
  answered heading's) through the kernel, `N` = C, `g` the best run (`open_heading`) or R (`way_to`). A heading with no
  room to move is on the harmful side. A wrong doorway costs the whole regret.
- **Dead bands.** No route through seen floor; the robot already within R of the goal; the told target; the own-log
  screen (`FamilySpec.own_log_screen`, held in reserve: it drops a point where the robot's own log already scores 0 or
  more).
- **Risks, gates.** "Turn back the way you came" (`path_log`); a constant 180 (balance); a longest run that is a layout
  fact (the map reader, a reference). The robot's own delivered poses can point the right way, which is why the own-log
  screen is held in reserve.
- **`JobSpec` hooks.** `looks` and `look_shares`, `look_phases` (the driving phases), `reacquire`, `clearance`.
- **Families.** `open_heading`, `way_to`. Retired: `adjacent`.

| Robot | Example |
| --- | --- |
| Floor (vacuum) | "If you had to turn in place and drive straight away from here, which way, of those you have seen, would let your body go farthest?" (`open_heading`) |
| Manipulator (helper) | "If you had to go now to the router you saw, which way would you set off to get there by the shortest way your body fits through?" (`way_to`); `open_heading` |
| AMR | None: a fleet manager with a map routes it, and every best run is the cap |
| Drone | None on this map: the open high street is the longest run from every point. Data round: the way to a clear alternate drop spot when the planned one is blocked, if open spots vary per flight |

## C5 What matters here

- **Decision.** Which things meet my job's predicate: on the way I drove, catchable by my brush, people at my drop?
  Harm while the job runs, reuse at its end.
- **Steps.** (a), (b) and a selection of what to keep.
- **Gold.** The entities of the predicate's kinds whose footprint meets its region (near the way driven, within a reach
  of the body's front, within a radius of the task point) in its phases, each with a recognisable sighting in the span. A
  kind that recurs at one spot across scripts (`hazard_rule.SPOT_KINDS`) and a thing under `min_height_m` are don't-care
  circles. A critical subset (within `FRONT_M` of the brush's front edge; within C of the drop point) carries the
  catastrophe.
- **Answer, credit.** Positions. A set scores per item: each item is a place, earning its direction quarter within M of
  it; a missed item costs one and a false one half; an empty gold is credited on a curve in the false places. Catastrophe
  `miss`, weight 3, for the hazard families during the job (`tangle_ahead`, `drop_zone_people`); the end list has none.
- **Dead bands.** A point outside the family's `look_phases` (the hazard families are asked while the robot cleans or
  is in transit, or in the drone's drop phases), and for `drop_zone_people` a disc not wholly in the frame. An empty
  list is an answer, not a dead band.
- **Risks, gates.**
  - Items in one dense patch the way always crosses: a blind list of two dense places earned until a list item became a
    place (`blind_in_pool`, `form_in_pool`).
  - An empty share the form predicts (balance; the empties are thinned to the job's `empty_keep_percent`).
- **`JobSpec` hooks.** `task_things` (`TaskThings`: `phrase`, `anchor`, `phases`, `until`, `reach_m`, `band_m`, `floor`,
  `min_height_m`, `dont_care_kinds`, `item_kinds`, `empty_keep_percent`), `ends`, `looks`, `look_phases`.
- **Families.** `task_things`, `tangle_ahead`. Retired: `drop_zone_people` (written for the drone's drop, asked by no
  job until a data round supplies it), `uncovered`, `slots_empty`.

| Robot | Example |
| --- | --- |
| Floor (vacuum) | "List anything lying on the floor within 1 m of you that your brush could catch" (`tangle_ahead`); at the end, the catchable things in the rooms cleaned (`task_things`) |
| Manipulator (helper) | "Where were the things on the floor within 1.2 m of the way you went to fetch and put down the thing?" |
| AMR | "Where were the things within 2 m of the way you drove in your first task?" |
| Drone | "Before you lower the package: where are the people within 4 m of the point below you?" (`drop_zone_people`: written, asked by no job) |

## C6 Change

**Gated on data: no job asks it today.** The three families that asked it are retired.

- **Decision.** Where is the thing I remember now, after it moved between tasks? Reuse.
- **Steps.** (a), (b), (e).
- **Gold.** A thing recognised at two places at least `STALE_M` apart (`World.arrived`); the gold is the newer sighting.
- **Answer, credit.** A place, `N` = R, as C3's.
- **Design.** No new wording. Change is the *moved stratum* of C3: `object_where` (and `ego_mem`) admit a thing that
  arrived, and the reports split `moved` from `steady` rows. A scripted baseline `first_sighting` (a perfect memory that
  never updates) must sit at the floor on the moved stratum while `memory_table` sits at the ceiling. One stratum
  replaces three families: `object_was` presupposed the move in its wording, `same_place` was a yes/no that the latest
  sighting answered, and 77% of the vacuum's `changed` golds were the empty set.
- **Why it waits.** Supply. In 16 medium cases the world moves 90 to 165 things a job, but the robot sees both sides of
  11 to 28 and can name 1 to 7. Choosing which things move cannot fix it: the largest loss is re-seeing (R4). And a kind
  predicts motion (vans moved 10 of 32 times, skips 0 of 39), which `blind` would learn (R5).
- **`JobSpec` hooks.** `ends` with the moved stratum; case windows of two or more tasks; a revisit plan.
- **Families.** None asked. Retired: `object_was`, `same_place`, `changed`.

| Robot | Example |
| --- | --- |
| Floor (vacuum) | "Where did you last see the extension cord?" after the resident moved it between cleans and the vacuum saw it again |
| Manipulator (helper) | "Where did you last see the laptop?" after the resident carried it from the desk to the sofa between two fetches |
| AMR | "Where did you last see the blue tote?" after a picker restaged it and the AMR drove that aisle again |
| Drone | "As you last saw it, where is the white van now?" on the way back, after it moved during the delivery |

## What the categories leave out

Each is left out on purpose, with the evidence. The numbers attributed to other benchmarks are their published results
on their own data. [VSI-Bench](https://arxiv.org/abs/2412.14171) is the reference point: where it asks a decision
(its route plan), the categories ask it as a graded, gated number or heading, and where it asks counts and sizes they
do not.

| Left out | Evidence | Where the decision lives |
| --- | --- | --- |
| Counting (VSI-Bench count, VSI-Super running count, OST-Bench quantity) | Not a decision. A frequency guess scores 62.1 on VSI-Bench's count, above Gemini-1.5 Pro's 56.2 ([TsT](https://arxiv.org/abs/2511.04655) debiasing cuts 54% of the benchmark) | A set scored per item (C5) contains the count and says where |
| Object size, room size (VSI-Bench, [STI-Bench](https://arxiv.org/abs/2503.23765) dimension) | Category priors answer it: a random forest on the category's mean size scores 61.4 MRA on VSI-Bench's object size, with no image (TsT) | Size matters only as whether the body fits, which C1 asks |
| Appearance order, when it was last seen, durations (VSI-Bench, Ego4D NLQ, [Beyond3D](https://arxiv.org/abs/2609.34630)) | `when_seen` and `order` failed on lags a model can learn (`blind_in_pool` +0.15 on the helper, +0.30 on the drone; `order` +0.25 on `other_case`) | Time is a dial, not a category: every gold row carries `age_s` and `age_band`, and the reports score by them |
| Imagined viewpoints (VSI-Bench relative direction, [MindCube](https://arxiv.org/abs/2506.21458), [ViewSpatial-Bench](https://arxiv.org/abs/2505.21500), [SAT](https://arxiv.org/abs/2412.07755) perspective) | The one-side rule. On relative direction a frequency guess scores 47.9, above Gemini-1.5 Pro's 46.3 | The bearing from the robot is C3's offset |
| Own pose and ego-motion as a question ([OST-Bench](https://arxiv.org/abs/2507.07984) agent state, [VSTI-Bench](https://arxiv.org/abs/2505.20279) camera displacement) | The pose is delivered with every frame. `where_was_i` was a header lookup that both reference models answered at 100% | Ego-motion enters only composed, as step (c) |
| Pairwise relations: the distance or offset between two things, which of four is closest (`distance_between`) | The one-side rule. The offset between two remembered things is two `object_where` answers subtracted, so its only added hardness is error compounding, which is stingier scoring in disguise | The robot to a thing: C3 |
| A route between fixed named places (VSI-Bench route plan, [HourVideo](https://arxiv.org/abs/2411.04998) room-to-room, `adjacent`) | The same in every case of a place, so `blind` learns it; asked only of a held-out place (methodology rule 5) | The way to a remembered thing: C4 |
| Free text, multiple choice, LLM judges ([OpenEQA](https://open-eqa.github.io/)); attribute and function questions | Methodology rule 7. Blind GPT-4 scores 51.0 on OpenEQA's state questions against GPT-4V's 57.7 | None: not spatial |

What the field has and the categories lack is change and mover motion, which the newest benchmarks
([Beyond3D](https://arxiv.org/abs/2609.34630), [OVMAN](https://arxiv.org/abs/2609.06424),
[SpatialMind-2K](https://arxiv.org/abs/2610.04139)) show are hard and unsolved: C6 and C2's step (d) wait on the data of
the section below. What they have as decisions, the categories ask better: SAT's action consequence and
[RoboVQA](https://arxiv.org/abs/2311.00899)'s affordance as C1 and C2, VSI-Bench's route plan and HourVideo's object
retrieval as C4, Ego4D's VQ3D as C3.

## The data a new environment must supply

A category is only as hard as the data under it. Each rule is checkable on the timelines before any render, with a
frustum proxy for "seen" (a thing is seen when it lies in the camera's frustum at a keyframe). The repository ships
no script for the pre-render checks yet; the gates of `stbench check` give the same verdicts after the render.

- **R1 Wide golds.** For each family a job asks, the gold's spread across scripts at every sensorless key (place, phase,
  plan-pose cell, kind) must be wide against the need: the best held-out constant, scored through the kernel, must not
  beat `unknown`. A spread under about 4 N' fails (a constant at the median then earns the direction and the 2N' quarter
  on half the rows). This is the balance gate, run before the render.
- **R2 People reach the work.** Every work point (drop, station, doorway) has people near it in some scripts and not
  in others, by a per-run coin, at a distance drawn over the whole cap: no keep-off that fixes the distance, and walk
  areas connected to the work. A walk area of a few need-circles recurs by construction: for n' placements of other
  scripts spread uniformly over an area A, the share within r of another script's is about `1 - exp(-n' pi r^2 / A)`.
  Questions about people are relative to the robot's motion or its task point (C2, C5 at the drop), never a list of
  where people stand, unless R1 passes.
- **R3 Walkers at the decision.** A stated share of the person sightings at look points are walking, and walks cross the
  robot's way at occluded ends (aisle ends, doorways) at a sourced rate. Without them step (d) is a rare row.
- **R4 Revisits.** A multi-task case re-drives regions an earlier task saw, after a gap. Re-seeing, not moving, is the
  change funnel's largest loss.
- **R5 Kind does not predict motion.** Each movable kind's chance to move in a gap is drawn per script, or equal across
  kinds, and destinations are wide.
- **R6 Change off camera, within tasks too.** People carry things while the robot works elsewhere, never on camera
  ([world.md](world.md)), so a one-task case can hold a change.
- **R7 No fixed plan pose at a decision frame** for a structure family: jitter approach poses per script.
- **R8 Movables per run, not per slot.** Vehicles at continuous kerb positions per flight (slot centres scored +0.25 to
  +0.40 on the drone), totes and pallets per task.
- **R9 Untold coins.** A coin the robot responds to is told (`World.told`). A decision question about a blocker is
  asked before the response and with the coin balanced.
- **R10 Placeable at the decision.** A person or thing a category asks about must clear the recognition floor (1,024
  px) at the ask. The drone's pedestrians cover 250 to 340 px: lower the inspect hover, or ask nearer.

## Adding a new environment or job

1. **Write the job's needs table** in [question-design.md](question-design.md) section 2's form: how it can do harm, what
   moves its task forward, what it should keep for next time. Each row names a decision, not a thing.
2. **Instantiate categories, not families.** For each of C1 to C5, say whether the job faces the decision; if so, fill
   that category's `JobSpec` hooks with sourced numbers (C, R, the body, the motion primitives and phases, the people
   file, the predicate, the horizon and the cap). Each value traces to a source of its kind and never moves after a score
   is read (methodology rule 7). Reuse the category's family. Write a new family only for a motion primitive, objective
   or predicate the existing gold cannot express ([question-design.md](question-design.md) section 9); then add it to
   `FAMILY_CATEGORY`, give its page a `Category` row and name it in its category's `Families` bullet here, or
   `tests/test_categories.py` fails.
3. **Draw the slot table before data:** harm (C1, C2, C5 during the job) in slot 1, what was seen before (C3 memory, C4
   goal) in slot 2, the rest in slot 3, and the end families in `ends`. No score enters a share.
4. **Supply the data** above, then run the pre-render checks on the timelines: R1's held-out constant per family, supply
   through the frustum proxy, the kind-motion table (R5), the revisit count (R4). Fix the generator, not the question.
5. **Render, compile, read the supply:** the questions per (job, ability, scope) cell (`ability_counts`; a cell under 20
   weighs `min(1, n / 20)`, and the eval report lists the thin ones). Then `stbench check --select job=<job>`, scoped. A
   family that fails a gate leaves that job by rule, with the rule stated on the family's page; never re-tune a
   constant.
6. **Document** the job's page, this page's example row for the new robot type, and a family page only for a new family.
