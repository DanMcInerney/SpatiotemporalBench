# Methodology

The rules that turn [the vision](vision.md) into questions, scores and checks. When a design choice is unclear, these
rules decide it, in the order valid, simple, realistic, fast. How each question family applies them is in
[questions.md](questions.md) and [families/](families/); how a run is scored and reported is in [eval.md](eval.md).

Code and docs cite these rules by number ("rule 5", "truth rule 2"). Older text calls them VISION rules; the numbers
are unchanged.

## 1. What it measures

**It measures task-conditioned spatiotemporal ability, perception and memory both:**
- perceiving the space through the robot's own sensors;
- keeping what matters;
- reasoning over both in space and time, in the service of a job.

| Ability | A controller asks itself |
| --- | --- |
| Distance | "Where is the nearest part of that drum I can see, from me, now?" |
| Memory | "Where, as I last saw it, is the red bin from where I am?" "Where did I last see the laundry basket?" "Which way do I set off to fetch the mug I saw in the other room?" |
| Change | "Was the cart where it was on my first pass?" (not measured yet: too few observable changes at today's case lengths) |
| Task relevance | "What stood within 2 m of the way I drove?" |
| Clearance | "If I drove straight ahead, backed up or turned and drove, how far could my body go?" "How close did a person come to me in that task?" "How close would the person nearest me come in the next 5 s, if we both kept moving?" "How close would the person I saw last, now out of view, come?" (`approach_mem`, 0.11.0) "Where is the person I saw, from here?" "What is on the floor near me that my brush could catch?" "Who stands within 4 m of the point below me?" "Which way gives my body the most room?" (`open_heading`, counted since 0.11.0) |
| Prediction | "Where will that picker be in two seconds?" (not measured yet; `closest_approach` extrapolates both velocities at the ask, as the ideal reader of the frames would, and is a clearance question: its gold is not the future, which no sensor shows) |
| Knowing what it does not know | `unknown` when the sensors never showed the answer, or declining to place a thing it cannot place within the job's accuracy |

Own path and progress are not asked: the pose and status are given, so they are log lookups. The corridor question, the
heading question and the route question are asked only in the phases in which the robot drives, which keeps them from
being a lookup of the status. **Fixed structure
is asked as clearance:** how far the robot could move from where it is before its body meets anything, fixed structure
included, as it has seen it, and which way gives it the most room. The agent is given no map, so it must perceive and
remember the structure. The layout's topology alone (which rooms join which, the route between named places) still
waits for a held-out place. **The way to a remembered thing is asked as help** (`way_to`, 0.11.0): which way the helper
would set off to fetch a thing it saw before, by the shortest way its body fits through the floor it has seen. It needs
the thing's place, which differs by script, and the layout between, so neither the place's layout nor the robot's own
log gives it. Safety is
asked as clearance and distance: how far the robot can move, and how near the people and things around it are.

**The four kinds.** Every question is also one of the vision's four kinds ([vision.md](vision.md), "How questions are
made and scored"), and the reports split by them (`abilities.FAMILY_KIND`):
- **harm:** what an action now would hit or endanger: `clear_path`, `closest_person`, `closest_approach`,
  `approach_mem` (0.11.0), `drop_zone_people` (asked by no job since the final review), `tangle_ahead`, and `person_mem`
  (asked by no job since 0.10.0);
- **help:** which action now moves the task forward: `open_heading` (the ability clearance, counted since 0.11.0) and
  `way_to` (the ability memory);
- **situation:** what is around the robot, in view and out of view: `ego_now`, `ego_mem`;
- **reuse:** what is worth keeping to do the job here again: `object_where`, `task_things`.

The kind is a split of the reports, not a cell of the headline (rule 4).

**Perception and memory are both scored.**
- The agent is fed the robot's camera and what its controller knows of itself (pose, clock, task status), and nothing
  else: it estimates distances from the pictures. Depth is recorded for the scorer only.
- Questions about what is in view now test perception and spatial reasoning. Questions about anything else test
  memory as well.
- Both count equally in the headline, and they are reported apart.
- A perfect-perception channel separates perception losses from memory losses, as a diagnostic.

**Judging what matters is part of the test.** The brief names the job and the place, never which objects or facts
will be asked about. The agent decides what to keep, and that judgement is measured: the same stream under a neutral
brief should score worse on the job's questions.

**The agent is tested as it comes.** A system under test may bring any memory or tools of its own; only its answers
count, and what it spent is reported beside its score. The reference agent brings none: its context is its memory and
`submit_answer` its only tool.

**It never measures:**
- **Control skill or physics.** Paths, jobs and world changes are scripted; the robot's motion is never scored.
  Whether this ability predicts control is checked separately (§1).
- **Free text or a label list.** Answers are positions, times, sets, sequences, yes/no and `unknown`.
- **A memory system's internals.** No mechanism is required, rewarded or forbidden.
- **Facts learnable without the sensors:** what the robot's own path log or the questions' wording gives away, and what
  is the same in every case of a place, except the fixed structure around the robot (above), which is asked and whose
  score is a claim only about a system that holds no copy of the map (rule 5).

## 2. The three layers

| Layer | What it holds | Open? |
| --- | --- | --- |
| **Simulator** | places, robots and their jobs, world changes → one timeline → rendered sensor recordings and the truth | all of it |
| **Dataset** | cases: a robot's sensor stream through part of a working day, its brief, and the truth | public, except a held-out set's gold |
| **Benchmark** | the questions, the protocol that streams a case to an agent, the scorer, the baselines | all but the held-out gold |

- **The world timeline is the single source of truth.** It records every thing's pose, state and existence over
  time, and the robot's path and actions. Recordings, visibility, questions and gold answers are all compiled from it.
- **What a robot would have is stored apart from what only the scorer sees.** The agent only ever receives the
  robot's side.

## 3. The rules

Every design question is settled by four rules, in this order:
1. **Valid:** it measures spatiotemporal ability and nothing else. A fast, realistic benchmark that rewards shortcuts
   is worthless.
2. **Simple:** the fewest places, robots, jobs, sensors, question families and moving parts that measure the claim.
3. **Realistic:** places, robots, jobs, paths, objects, people and change rates that real robots meet.
4. **Fast:** render once; then running a benchmark is only the agent's own time.

A design decision names the rule that decided it.

### Truth

1. **Every gold answer is computed** from the world timeline and what the sensors recorded. No person or LLM ever writes
   or judges one.
2. **Gold is what the robot could know.**
   - It is computed from what was visible, and when. If something changed while nobody was watching, the gold is what
     was last seen, or `unknown`.
   - Every gold names its evidence. A claim that space is empty needs space seen to be empty.
   - **Test:** nothing unobserved changes a kept gold.
3. **The truth never reaches the agent.**
   - It sees only the robot's side: its brief, its sensors, and the questions.
   - Nothing it sees carries hidden state, entity IDs, seeds or revealing names.
   - A held-out set of places stays private, so training on public data cannot learn the answers.

### Spatiotemporal ability, not something else

4. **Every question needs the sensors, and says what it tests.**
   - A `now` question is answered from what is in view at the moment it is asked: perception and spatial reasoning.
     What is in view is the latest frame: a system gets one frame at a time (0.10.0), so a gold is `now` only when the
     latest frame alone gives it, and a corridor or a heading that needs an earlier frame is `memory`.
   - A `memory` question needs something the robot is not seeing then: what is behind it, what it saw before, what
     changed since.
   - The mix is declared before data. Perception and memory count equally: the headline is the mean of the `now` and
     `memory` columns, whatever their question counts, and each is reported with its count. Within each column its
     abilities count by their weights (equally when every cell holds 20 questions or more, the next bullet), each pooled
     over its own questions; an ability whose questions fall in both columns (clearance) is one cell in each, so the unit
     is the (ability, column) cell. The mean of the cells that count, by their weights, and the pooled mean are
     published beside the headline.
   - **A cell weighs `w = min(1, n / 20)`** (0.11.0; `ability_counts.weights`, `score_stats`), `n` the questions of its
     (job, ability, column) on the release's dev root, all compiled cases and lengths. It replaces the rule of 0.6.0 to
     0.10.0 that left a cell of fewer than 20 questions out of the headline, which flipped cells in and out between
     compiles (the helper's clearance `now` cell held 26, 14 and 51 questions across the compiles of 0.10.0). A cell's
     mean pools its rows by their weights, the cell weighs in its column the largest weight of its rows, and a column
     weighs in the headline the largest weight of its cells, so a column of thin cells alone does not count as half the
     headline. A dataset whose cells all hold 20 questions or more is unweighted, so 0.10.0's headlines move only by the
     new credit. One cell set serves every score of a release: the dev, the fast tier and the held-out set are scored on
     the dev root's cells (`--cells-from`, [eval.md](eval.md)), so their headlines have the same cells.
   - The reports also split the questions by the four kinds (`by.kind`, beside `by.ability`), each kind pooled within
     itself. It is a split and not a set of cells, because a kind spans abilities and each ability keeps its own cell.
     Since 0.11.0 `open_heading` is a clearance question and counts in the headline, and `way_to` a memory question;
     only the retired `prediction` ability is a diagnostic ([families/open_heading.md](families/open_heading.md): the
     heading's credit was a cliff through 0.10.0, which the one need rule of rule 7 replaces).
   - Each gold row also records the age of its evidence (0.10.0): `age_s`, the clock seconds from the gold's oldest
     evidence keyframe to the ask, and `age_band`, `now` exactly when the scope is `now`, else `recent` (up to 10 s),
     `task` (the evidence is in the task the question falls in) or `earlier` (an earlier task); both are null on an
     `unknown` row. The reports give the score by band, overall and for each job, a memory curve of credit against how
     long ago the robot saw the answer. Like the kinds, the bands are a split of the reports: the headline is still the
     mean of the `now` and `memory` columns.
   - **Test:** an agent that sees only the current view scores at the floor on the `memory` questions.
5. **Shortcuts are gates.** A question family ships only when:
   - perfect knowledge of what was seen scores at the ceiling;
   - each of these sits at the floor: the best answer learned from other cases (blind), the robot's own path and
     action log alone, another case's stream, the other questions' wording, and all-`unknown`;
   - for a `memory` family, the current view alone sits at the floor too.

   **Exemptions:**
   - a `now` family is answered from the current view by definition;
   - a family about the robot's own path is answered from its own log by definition.

   For these, that baseline is the reference, not a shortcut, and they are reported in their own columns. A fact that is
   the same in every case of a place is asked only of places the agent has never seen, with one exception: the fixed
   structure around the robot (§1). A structure family asks how far the robot's body could move or which way gives it
   the most room; only `clear_path` and `open_heading` are, and making another needs this rule amended, never a gate's
   failure (`way_to` is not one: its gold moves with the thing, which differs by script, so `other_case` is a gate on it
   and it has no map-reader reference). Its references are the `map_reader` (the public map's raycast from the delivered
   pose, and on a held-out place, which has no map, the pose-keyed pool of other scripts' golds) and `other_case`; they
   are not gates. `stbench check` reports them as design diagnostics (the agent is never given a map, and the map
   reader's score is not a published result). Everything else stays a gate on a structure family: `blind`,
   `form`, `questionnaire_blind`, `all_unknown`, the balance rule, `path_log` answering from the robot's own log alone
   (its swept path, detours and straight runs, no map, no pool of answers; each rule stands only where it beats
   abstaining on other scripts) and, on its `memory` rows, `current_view` and `current_view_prior` without a pose key.
   `separation` reads the gated set, and the check prints the family above the map reader, as distance is printed above
   `kind_range`. On a public place such a score is a claim only about a system that holds no copy of the place's map or
   data; a held-out place makes it a claim about every system. Thresholds are fixed before data. A family that fails is
   fixed or dropped, never forced.

   **A heading family's own-log screen** (0.11.0; `heading_screen`, `FamilySpec.own_log_screen`, off by default and off
   in the specs as shipped). `open_heading` and `way_to` are answered by a heading, and the robot's own delivered poses
   can point the right way: `path_log` answers the first with the longest straight run it drove and the second with the
   way back to where it was eight steps ago. The screen drops a point as a dead band when that unbarred rule already
   scores 0 or more on the point's own row under the row's own credit, that is, when the log alone points within about
   twice the need of the best way. It mirrors `World.told`, which never asks what the robot's log tells. It is the
   fallback declared before the free check, for a job where `path_log` or its in-pool twin then reads over the shortcut
   line; it takes nothing from the points it keeps and no threshold moves to pass a gate. A family that still fails
   after it leaves that job, by rule.

   No constant answer may score above `all_unknown` on a family's declared selection: the compile draws answer classes
   so that none can, and the gates confirm it. A family may declare a scripted reference that is not a shortcut
   (`ego_now`'s typical range per kind) and is reported against it. A prediction family must beat the prediction that
   nothing moves.

   A question's target is what the frames show. Its gold is what a perfect reading of the delivered frames gives, with
   the robot's camera model and pose and without depth, and its full credit is reachable at the camera's resolution.
   An offset names the nearest visible point of a thing, not its hidden middle; a place in the map (`object_where`, an
   item of `task_things`) names the middle of the thing where it stood, and where its extent is hidden the credit allows
   it (the resolution grows by the half-length). The prompt says which point each answer names.
6. **Questions do not teach, and time does not leak.**
   - The stream reaches the agent in time order. Nothing from the future is readable.
   - A question asked during the job is about the robot's own situation, at a moment a controller would ask it, and
     its words reveal nothing about the world. Everything else is asked after the job.
   - No subject is asked twice in a case. No thing, place or time that one end question names appears in another.
   - A thing a during question names is a subject for the rest of the case. A task named by its ordinal is not a
     subject.
   - How much asking during the job changes later answers is measured and published, not assumed.
   - Which family a look point asks is drawn from the case id and the point alone, from a table each job declares
     before data; the draw reads no world, gold or answer, so which question is asked reveals nothing of what is in
     view. Which frames are asked is drawn the same way: any frame of the case can be a look point, and the frames of the
     decisions a job's task turns on come first in the draw. A question's presence can show that a frame is one of the
     robot's decision moments, which its status and subgoal already show and the plan alone decides.
   - A frame and its questions reach the reference agent together, one question for each ask, so a later question cannot
     inform an earlier answer.
   - Nothing about a question's form predicts its answer.
   - The agent runs in its own process with no access to the dataset's truth or the benchmark's code.
7. **Answers are typed and scored by code.**
   - No LLM judge and no open multiple choice.
   - A wrong answer costs more than `unknown`.
   - **A metric answer is credited by the consequence for the job** (four quarters):
     - a quarter for the right *direction* (the side, the order, blocked or clear);
     - a quarter for the right *thing or action*, full within twice the need N' (below) and falling to nothing at four
       times;
     - a quarter for the right *amount*, full within the need and falling to nothing at twice the need;
     - the last quarter only for *precision at the camera's resolution*: full within one pixel at the geometry the gold
       rests on (`tolerances.resolution_m`), falling log-linearly to nothing at the need.

     The credit is continuous and strictly falling between the resolution and four times the need, so no answer coarser
     than a pixel earns the whole, and its one step is the direction quarter. The penalised score (`2s - 1`, `unknown`
     0) is the **gate score**: every gate, `knowable` and the evidence verdict read it, as the row's `score_gate`. The
     **reported score** is the penalised score except for a catastrophe (below), which scores minus its weight; it is the
     row's `score`, and the headline, the reports and the reference rows read it (0.10.0; until then the two were one
     score and this sentence read "unchanged"). A weight inside the gate score would lower every shortcut that guesses,
     so a family could pass a gate because of the weight and not because the sensors are needed.
   - **The need is the job's** and is declared as before: where a thing is from the robot, how far its body could move
     and how close a person came are credited against the job's clearance C, the margin the robot's own operating spec
     keeps beside its path to the things it meets, since what the robot does with any of them is stop C short; a place
     in the map against the radius R within which the robot finds the thing again; a list against its matching radius
     M. Each value:
     - traces to a source of that kind, never a detection size, a placement or docking precision, or a
       facility-layout rule;
     - is declared before data and stated in the prompt;
     - never moves after a score is read.
   - **One need rule for every distance** (0.11.0; `credit_kernel.KAPPA`). The job's need N is the
     whole need while the distance judged is within ten of it, the near field where harm happens. Beyond that the margin
     grows with the distance, the same for every robot and every camera:

         N' = max(N, 0.1 g, 2 rho)

     where `g` is the distance the answer is judged over: a distance's gold value (`clear_path`, `closest_person`,
     `closest_approach`), an offset's ground range from the camera (`ego_now`, `ego_mem`), a heading's best run
     (`open_heading`) or the re-find radius R (`way_to`). A place keeps R and a list item its match radius M, with no
     distance term. The first two terms are halved on the harmful side; `2 rho`, two pixels of the camera, is not.
     - Before 0.11.0 the far-field margin was whatever two pixels of the robot's camera resolved at that range, so the
       same 20% error cost -0.42 on one ground robot and scored +0.60 on another. A door 5 m away, answered 4 m,
       now scores 0.00 on the vacuum, the helper and the AMR alike, as good as `unknown`; 4.5 m scores +0.50 and 6.0 m
       scores -0.42. The drone keeps its 2.4 m clearance, which holds to 24 m
       ([question-design.md](question-design.md), section 6, has the table).
     - A tenth is the largest relative slack that keeps every constant answer at the floor and the models apart: a fifth
       lifts the AMR's constants. The form, a floor and a relative term, is `numpy.isclose`'s `max(atol, rtol |b|)`.
       Full credit is still only at the camera's pixel.
   - **Harm costs more than caution.** On the harmful side the need is halved, in its first two terms: a clearance
     longer than the gold, a closest approach reported wider, a thing or a person placed farther from the camera than it
     is, a heading without room to move (below). A place to re-find a thing has no harmful side. Where one pixel is half
     the need or more, the need is already widened to two pixels on both sides, so the halving does nothing there (the
     kernel's design; on the vacuum's corridor `rho / C` reaches 0.50 at its 90th percentile). The halved side is why an
     over-estimate costs more than an under-estimate of the same size.
   - **A catastrophe costs a flat weight in the reported score only.** An answer whose action would end in harm,
     with a harmful error larger than `max(C, 2 rho)` (the clearance the job keeps, or twice what a pixel resolves), scores
     minus a weight, whatever credit it earned: 2 (`credit_kernel.CATASTROPHE_THING`) for contact with a thing the robot
     saw, which damages or stops the job and a person clears (a corridor said longer than it is, where the true corridor
     is under the body's length plus C and the answer is not: a long corridor overstated is a wrong answer and no
     collision); 3 (`CATASTROPHE_PERSON`) for contact with a person or a hazard
     of the robot's own kind, a tangle, which endangers someone or ends the job until a person intervenes. The rule is on
     the gold row (`tolerances.catastrophe`); the weights are stated values, declared once with that reason and never tuned
     to a score. `unknown` is never a catastrophe. The reports give the catastrophe rate, the share of the harm rows
     answered that are catastrophes ([question-design.md](question-design.md), "How answers are graded").
   - **A place's direction is its radius.** A place earns its direction quarter within R of the thing and nowhere
     else: the room a thing is in is a fact of the map and the kind, which a system that holds the layout knows without
     the camera, so crediting it would pay that system for the layout (the 0.9.0 review found it was the whole of a
     sensorless layout prior's lead on the helper's `object_where`).
   - **A set scores per item.** Each gold item earns its own credit, a missed item costs one and a false one half; the
     list is perfect only when it holds every item, each to the camera's resolution, and nothing else.
     - **Each item is a place** (0.11.0; `score_set.graded_set`). Its direction quarter is earned within the match
       radius M of the item and nowhere else, the rule above with M for R. Through 0.10.0 a list pair earned that
       quarter at any distance up to four radii, the one place answer outside the rule: a sensorless prior that put
       places over the densest part of a room was paid it. The pairing reach (four radii), the other three quarters and
       the don't-care circles are unchanged.
     - **The empty gold's one thing is the empty list:** `s = 1 / (1 + FP_WEIGHT |false|)`. One false place earns 2/3 and
       three 0.4 (`unknown` earns 0); through 0.10.0 any false place earned 0, as ten did.
   - **A decision scores by its regret,** what it gives up against the best action, on the need rule above.
     - **A heading's direction is room to move** (`open_heading`; 0.11.0). The right direction is a run of the room the
       body needs, T (the body's length plus C), or longer, or the best run less a pixel where no heading has that much.
       The harmful side is the complement, a run under T. The regret is the best run less the answered heading's run, with
       `g` the best run. (Through 0.10.0 the direction was a run of half the best, and the credit was a cliff: a heading
       2 to 10 degrees off a doorway scored -0.4 to -1.) The variant that paid every heading with room linearly in that
       room let the robot's own path log alone score +0.38 over abstaining on the helper, so it is not used; this one
       leaves the log at +0.04 on the helper and -0.14 on the vacuum (the design's measurement on 0.10.0's rows).
     - **The route credit** (`way_to`; 0.11.0). The answered heading is judged by the progress it makes toward a thing the
       robot saw before: the outcome `o` of a heading is how much nearer one R of driving along it brings the robot to the
       thing, through floor it has seen and by the shortest way its body fits through, and the credit is the kernel on the
       regret `max(o) - o`, with N = C and rho the field's 0.1 m cell, so on the helper N' = max(0.15, 0.1, 0.2) = 0.2 m. A heading with no room to move (a run under T), or
       into floor never seen, has no outcome: its error is the best outcome plus R, it has no direction and it is on the
       harmful side, which scores -1. A wrong doorway costs its whole regret.
   - **Full credit is reachable.** An ideal reader of the same frames places every point to a pixel and says `unknown`
     exactly where nothing was seen, so it scores 1: the oracle and `memory_table` reproduce each gold and stay at the
     ceiling, and every shortcut stays at the floor.

### General

8. **No special cases.**
   - A robot is a profile: its sensors and where they are mounted, its speed, its footprint, and how its jobs move it.
   - No contact sensors and no physics.
   - Nothing special-cases one robot, one agent, one memory system or one model.
9. **Every sensor is recorded once; the agent is fed the camera.** The agent is fed the camera and the robot's own
   pose, clock and status; other recordings serve the scorer only. Each question declares what it needs.

### Realism

10. **Realism is measured, not asserted.**
    - **The camera images are hyper-realistic.** Every frame should pass, at a glance, for a photo from the real
      robot's camera:
      - rendered in Blender from photoreal free-licensed assets;
      - as cluttered and worn as real places;
      - with real light and exposure;
      - through a model of that robot's camera.
    - Each place's look is checked against real reference photos and measured (objects per frame, light levels).
      The look is approved before the place is rendered in full.
    - Jobs and paths follow real robots. Change rates, people and traffic follow a published figure where one exists
      and are a modelling choice otherwise, each noted beside its number in its module's `SOURCES`
      ([people.md](people.md), [world.md](world.md)).
    - How often the world changes follows the sourced rates; which things change is chosen among those where the
      robot works, so that a change can be observed.
    - Every object a question depends on must be recognisable by a person in a single frame. A pixel floor below
      today's is set per place and kind by a blind review of crops at that floor, against a bar declared before the
      crops are seen; a kind that fails keeps the higher floor.
    - Difficulty comes from realistic jobs: the accuracy they need, the things they meet and their real lengths,
      never manufactured.
      - A strong model near the ceiling is a reason to check whether the questions ask less than the job does, not
        evidence that they do.
      - The remedy is the job's own requirement or what it really meets, checked with more than one model, never a
        tolerance or a build decision chosen to reach a score.
      - If memory turns out not to be the bottleneck at realistic lengths, that is reported.
11. **Open end to end.**
    - Assets are CC0, CC-BY or BSD only, with a licence manifest.
    - No proprietary runtime or non-commercial asset anywhere in the pipeline.

### Speed and honesty

12. **Render once.**
    - Running the benchmark only streams files to the agent.
    - Free checks finish in under 3 minutes.
    - A quick run answers a question in minutes.
13. **One axis at a time.**
    - Fixed, where variation would confound: the question grammar, the scorer, the protocol.
    - Axes, where variation is the measurement: place, robot, job, case length, sensors, brief.
    - **Test:** changing one axis changes no other component's output.
14. **Report results as they are.**
    - Per domain, case length, job and question family, with uncertainty, baselines and resources (tokens, cost,
      time), failures included.
    - The same code over the same inputs gives the same bytes. Determinism is checked by computing twice, never
      against a recorded digest, and nothing is stamped with a version.

## 4. Do not

These are the ways a benchmark like this goes off the rails:
- Grade control, free text, or a memory system's internals.
- Let a model write, fix or judge a gold answer.
- Ask a question the sensors could not have answered, or count as memory a question the current view answers.
- Ask, during the job, a question whose words tell the agent about the world or about what will be asked later.
- Name, in the brief or a question, objects or facts for no reason the job gives.
- Impose a memory mechanism, a notes format or a context budget on the agent's headline score.
- Add a place, robot, job, sensor or question family that no measurement needs.
- Build a special case for one robot, agent, memory system or model.
- Make cases longer or stranger than real jobs to force memory to matter.
- Render during evaluation, or depend on non-commercial assets or proprietary runtimes.
- Use generative world models or AI restyling for anything a question depends on.
- Check a change with a long or paid run when a free check or a quick run answers it.
- Tune jobs or questions to what one system happens to fail.
- Credit an answer more loosely than the job needs, or more strictly than the job needs or the camera can resolve.
  Full credit sits at the gold's own one-pixel resolution (`tolerances.resolution_m`, rule 7), which an ideal reader of
  the frames reaches, and never below it.
- Claim the score predicts control before the closed-loop check shows it.
