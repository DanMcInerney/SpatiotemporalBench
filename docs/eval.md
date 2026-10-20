# `stbench eval`: a system over a dataset's cases

`stbench run` runs a system over one case. `stbench eval` runs it over a selection of the dataset's compiled cases,
resumable, in parallel, under an optional spend cap, and writes one result a layout: `results.json`
and `results.md`. `stbench eval compare` lays results side by side and refuses results that differ in more than one
thing; `stbench eval distance` breaks a run's distance questions down against the job's clearance and by range band. Both go through `stream.run_case`, so a case is streamed, asked, scored and written exactly as `run` does it
(`cmd_run.write_run`).

```sh
uv run stbench eval SYSTEM --data D [--out RUNDIR] [--cases ID,...] [--select KEY=VALUE ...] [--channel C] [--brief V]
                   [--during on|off] [--arms LEVER=V1,V2[,...]] [--workers N] [--local] [--timeout S] [--reasoning E]
                   [--text-only] [--memory SPEC] [--compact-at TOKENS] [--cells-from DEVROOT]
                   [--max-usd TOTAL --case-usd X]
uv run stbench eval compare RESULTS_A RESULTS_B [...] [--out FILE.md] [--cells-from DEVROOT]
uv run stbench eval distance RUN [REFERENCE_RUN ...] --data D [--out FILE.md] [--cells-from DEVROOT]
```

`--text-only` is for a Codex or Claude agent: it is sent no image (a zero-shot check of what the pictures add). It is
refused for every other system (`stbench run` too). A text-only run cannot be taken for a camera one: its default folder
ends in `-text-only` (`runs/<system>-<channel>-text-only`; `stbench run`'s `runs/<system>-<case>-text-only`),
`results.json` and `results.md` and the viewer's header say it, `summary.json` holds `text_only`, each row of
`rows.jsonl` carries `"text_only": true` (a camera run's rows gain no key), a resume must find it unchanged, and `eval
compare` takes it as a lever of its own, one at a time.

`--memory SPEC` is for a Codex or Claude agent too: a memory store beside its context ([codex.md](codex.md), "A memory
beside the reference agent"; [claude.md](claude.md), "A memory beside the agent"). It is refused (exit 2) for every other system, on the detections channel and with `--text-only`. A run
with a memory cannot be taken for a reference one: its default folder ends in `-memory-<SPEC's stem>`, `results.json`
holds `"memory": "<name>"` (a reference row has no such key), `results.md` and the viewer's header name the system
`<system> with memory <name>`, a resume must find the same name, and `eval compare` takes the memory as part of the
agent lever, with the system and the reasoning effort: a memory row and the reference row of its model and effort differ
in one lever.

`--compact-at TOKENS` is for a Codex or Claude agent too: Codex compacts the thread once its context passes that many
tokens, earlier than its own limit ([codex.md](codex.md), "Compaction is Codex's own"), and Claude Code does the same
through its auto-compaction variables, from 1 to 66,400 tokens in steps of 800 ([claude.md](claude.md), "Compaction"),
for experiments on memory beyond the context. TOKENS is a whole number, at least 1 (anything else is a usage error, exit
2); it is refused (exit 2) for every other system, and for Claude beyond its range. A run with a limit cannot be taken for a reference one: `results.json` holds `"compact_at": TOKENS` (a
reference row has no such key), `results.md` and the viewer's header name the system `<system>, compacts at 50k tokens`,
`codex.summary.json` (`claude.summary.json`) and the journal's `run` event hold `compact_at` (null without a limit), a resume must find the same
limit, the export names the row `...-compact-TOKENS-...`, and `eval compare` takes the limit as part of the agent lever,
with the system, the reasoning effort and the memory. The default folder does not change, as it does not for
`--reasoning`: give a limited run an `--out` of its own. Both arms of a comparison must be run with the same limit.

Exit 0: every selected case finished (in every arm). Exit 1: some did not (ended, invalid, refused, the spend cap); the
results so far are written and marked partial. Exit 2: refused before anything ran, or an agent that cannot run at all.

## SYSTEM

As for `stbench run` (`docs/agent.md`), through the same registry:

- a **baseline** (`all_unknown`, `memory_table`, `current_view_prior`, ...). It runs the whole selection in one
  `suite.run_suite` (`--workers` processes), because its priors and the system it keys on (`prior_nm`: `current_view`
  runs first) need every case; only the folders of the cases that are not finished are written. `--max-usd` is refused
  for it, and a channel the system's class does not list is refused;
- `FILE.py:Class`, an **external system**: in the Docker sandbox, the verified mode, or with `--local` as a plain
  subprocess stamped `sandbox: none` and not verified. One case a thread, `--workers` at a time;
- `<agent>:<model>`, a **reference agent** through `cmd_run.AGENTS` (`codex:<model>` or `claude:<model>`; `--reasoning`
  goes to it). One case a thread, `--workers` at a time: a `codex:` or `claude:` case is a CLI process, so keep it at
  about 20 or fewer. `--channel detections` runs it on the detections arm (what the pixels cost), reading each frame's
  detections as text ([codex.md](codex.md), "The detections arm"); compare it with an `rgb_pose` run (the default, the
  camera alone). An agent that raises `Refused` (no `codex` or `claude`, an unlisted model) cannot run at all: the pool
  stops, nothing is left behind and the exit is 2; one that raises `RunInvalid` (its leak scan hit) leaves the case
  `invalid`, its files kept to inspect.

`--local` with anything but `FILE.py:Class` and `--reasoning` with anything but an agent are refused.

## The selection and what is refused first

The cases are the dataset's compiled cases that ask a question (`check.compiled_cases`), narrowed by `--cases` (ids)
and `--select KEY=VALUE` on `layout`, `robot`, `job` and `length`: a key given twice is either value, different keys
must all hold (`case_select` is the one owner of the rule; `stbench check` takes the same two options, below). The
command refuses, with exit 2 and nothing written: an unknown SYSTEM or an option it cannot take, a
selection with no case, a case with no gold, a RUNDIR under the
dataset's `scripts/` or `private/`, a bad `--arms`, `--select` or `--max-usd` (it needs `--case-usd`), and the `swapped` brief,
as an option or as an arm: it is retired (0.5.0-control), since a job's subgoal says the true task in every frame and a
swapped brief would contradict it. A case that cannot run for its own reason (no question the channel can answer) is
`refused: <message>` in the results and the run goes on.

## RUNDIR

By default `D/runs/<system>-<channel>` (characters of the system outside `[A-Za-z0-9._-]` become `_`), with `-<brief>`
and `-during-off` added when those are not the defaults. Each case writes `cases/<cid>/`: `answers.jsonl`,
`rows.jsonl`, `summary.json` (the keys of `docs/format.md`'s run folder, with `remote_memory`, `cost_usd` and `cost_note`
from the stream's usage report and the price table) and whatever an agent writes itself (`codex.calls.jsonl`,
`codex.summary.json`). `results.json` and `results.md` are written from those folders, never from memory.

## Resume

A case is **finished** when its `summary.json` says `complete` and its rows were written. Running the same command
again skips the finished cases and runs every other one from its start: its folder is removed first (a Codex call
cannot be replayed, and a stale journal would refuse the rerun). `results.json` is written, with no case finished,
before the first case runs, so a run killed at any point still says what it was. A rerun into the folder with another
system, channel, brief, during flag, sandbox, reasoning effort, `text_only`, memory or compaction limit (`RESUME_KEYS`),
or onto another compile (a finished case's qids not in the gold), is refused, never relabelled. After a change to the scorer or the baselines,
run into a fresh `--out`. The results list the cases of the current selection.

## The spend cap

`--max-usd TOTAL --case-usd X`: a case starts only while what the run folders' summaries spent, plus what the cases in
flight reserve (X each), plus X, stays within TOTAL. A case's spend is the price table's (`prices.cost_usd` of its
reported tokens, never a figure of the agent's own), and X when it cannot be priced (its `cost_note` says why), so the
cap holds. When a case may not start none starts until one in flight ends; with none in flight and none allowed the run
stops, partial, and the results say so. Under a resume what the folders already spent counts.

## Arms: `--arms LEVER=V1,V2[,...]`

One lever's arms run on every case: LEVER is `during` (`on`, `off`) or `brief` (`correct`, `neutral`). Each
value runs into `RUNDIR/<lever>-<value>`, a run folder of its own whose `results.json` is stamped with that value, the
arms of one case one after another in the order `sorted(values, key=stable_hash(case id, lever, value))`, so no arm
always runs first (the order matters to the leak check; a baseline is deterministic, so it simply runs each arm over
the whole selection in turn). The lever given as an option of its own is refused;
everything else is equal by construction (the brief's during sentence, for a job that has one, is in every arm).
The leak check is `--arms during=on,off` on the reference agent and `eval compare` of the two folders; since
0.7.0 every job asks during the job (at its look points), so the two arms differ in every job.
The conditioning comparison is `--arms brief=correct,neutral`.

## `results.json`

```text
{"system", "channel", "brief", "during" (bool), "sandbox" ("in-process" | "docker" | "none"),
 "reasoning" (agents; else null), "text_only" (bool), "memory" (an agent's memory name; absent otherwise),
 "compact_at" (an agent's compaction limit in tokens; absent otherwise),
 "verified" (sandbox docker or in-process, and no case with remote_memory), "partial", "max_usd",
 "cost_usd" (the sum over the cases that ran; null when any of theirs is),
 "cases": [{"id", "script", "layout", "robot", "job", "length", "status" ("complete", "ended: <why>",
            "refused: <message>", "invalid", "pending"), "failures" (by reason), "wall_s" (seconds per message),
            "state_bytes", "usage" (CaseRun.usage as is), "usage_problems", "tokens" (usage_totals),
            "remote_memory", "cost_usd", "cost_note", "error"}],
 "rows": [score.row rows of the finished cases],
 "layouts": {layout: score_stats.aggregate(that layout's rows) or null},
 "notes": [...]}
```

The resource fields come from `stream_protocol.usage_problems`, `usage_totals`, `remote_memory` and `prices.cost_usd`,
`cost_note`. Each layout is aggregated on its own and nothing pools layouts (methodology rule 13). The notes say why a result
is not a benchmark score: partial, unsandboxed, a remote store (unverified), a usage report with problems, the
budget's stop, a refusing agent. `eval_results.problems` checks a document against the schema and `load` refuses one
that fails.

### The splits

A layout's aggregate (`score_stats.aggregate`) reports the headline, the mean of the `now` and `memory` columns (methodology
rule 4), and beside it:

- The headline's columns are `columns` (`{scope: {mean, ci, n}}`): each is the **mean of the cells in it by their
  weights**, a cell being an (ability, column) pair (`abilities.ability_of`: distance, memory, change, relevance,
  clearance, prediction; a family not in the table is an ability of its own, so an older run stays scorable), each cell
  the reported score pooled over its own rows (a ratio of sums). A cell weighs the same whatever its number of
  questions once it holds 20 or more, and the two columns count equally whatever theirs. Clearance has a cell in each
  column, because its rows take the column of their scope (`clear_path`, `closest_approach`, `tangle_ahead` and
  `open_heading` by evidence, `closest_person` in `memory`); `way_to` is a `memory` question.
  No other ability has rows in both. When the rows hold one scope (a during-off arm, a family, a job with no `now` question) the headline is that
  column. Its 95% interval is a bootstrap over scripts of the mean itself: the abilities, both columns and their mean
  are computed on every resample, from one draw of scripts, so the interval is not built from the columns'
  intervals. A resample that lacks an ability drops it from its column; one that lacks a column takes the other
  alone. `n` is the questions counted in the column.
- **Weights (0.11.0).** A row weighs what its (job, ability, column) cell weighs in the **whole dataset's** golds
  (`ability_counts`, read from `--data`, or from `--cells-from` another root's, so a subset run, a pilot or a fast tier
  keeps the release's cells): `min(1, n / 20)` for its `n` questions (`abilities.MIN_ABILITY_N`). A cell's mean pools its
  rows by their weights; the cell weighs in its column the largest weight of its rows, and the column in the headline
  the largest weight of its cells. So a cell of 14 questions counts at 0.70 of a full cell, one of 6 at 0.30, and a
  column whose cells are all thin is weighed down as a whole, where 0.10.0 left every cell of fewer than 20 questions
  out of the mean and a count that crossed 20 between two compiles flipped it in or out (the helper's clearance `now`
  cell went 26, 14, 51 during 0.10.0's development). **No cell is a diagnostic for holding few questions**; `thin`
  lists the cells below 1, each with its weight, and `results.md` names them. A dataset whose cells all hold 20 or
  more is unweighted, so 0.10.0's headlines move only by the rule changes. On the 0.11.0 dev root every cell holds at
  least 20 (the least hold 22: the AMR's clearance `now` and the vacuum's relevance). **Only `prediction`, an ability no
  job asks (`abilities.DIAGNOSTIC`), is a diagnostic:** it is shown and counted in no mean. `open_heading` was one
  through 0.10.0, while its credit was a cliff ([families/open_heading.md](families/open_heading.md)); since 0.11.0 it is
  a clearance question and counts. The rule is relaxed for the dataset as a whole, never from a run's rows, when every
  non-diagnostic cell of the dataset is thin (a development one, `ability_counts.thin_rule`): none is applied,
  `thin_ignored` says so, and `results.json` carries the note that the headline is not the release statistic. Rows that
  are all diagnostic give a null headline and no columns (`stbench run` and `results.md` say so). A results document of
  an earlier build lists a thin cell alone, which reads as weight 0, the rule it was scored on.
- `abilities_mean` and `pooled`, `{mean, ci}`, from the same resample: the mean of every counted cell by its
  weight, and the counted questions pooled. They are published beside the headline because reweighting alone can reorder
  systems.
- `above_floor`, `{mean, ci}`: the headline less the unknown share, weighted the same way. Answering `unknown`
  throughout scores exactly the unknown share (1 on an unknown gold, 0 on an answerable one), so this is what a system
  scores beyond abstaining, resampled by script like the headline.
- `by[dimension][value]`, `{mean, ci, n}`, for `scope`, `own_path`, `family`, `ability`, `kind`, `when`, `length`, `layout`,
  `robot`, `job`, `critical`, `slice`, `relevance`, `answer_class` (`none` for a gold that declares none) and `age_band`
  (0.10.0). An
  ability's `mean` pools all its questions; its entry also holds `cells`, one for each scope it has rows in (`n`,
  `counted`, `diagnostic`, the means and intervals, and its `weight` in the headline: its column's share of the columns'
  weights, times its share of its own column's cell weights, which is one over the columns that have a counted cell,
  over the counted cells of its own column, when no cell is thin), and says its `scope`, how many of its questions are `counted`
  in the means, whether it is a `diagnostic` (none counted) and the `counted_mean` and `counted_ci` over the counted
  ones alone, which is what the report, the viewer and `eval compare` show in the lines of an ability. Every `by` group is pooled within itself (questions weigh equally, a
  diagnostic ability's rows among them), so `by.length` or `by.job` need not equal the headline of those rows. **`kind`** (0.9.0)
  is the vision's four kinds of question (`abilities.kind_of`, [vision.md](vision.md), "How questions are made
  and scored"): **harm** (`clear_path`, `closest_person`, `closest_approach`, `tangle_ahead`, and `drop_zone_people` and `person_mem`, which no job asks), **help** (`open_heading` and, since 0.11.0, `way_to`), **situation**
  (`ego_now`, `ego_mem`) and **reuse** (`object_where`, `task_things`). It is a split beside `ability`, pooled within
  itself over every row of the kind, and not a set of headline cells. Help's two families are counted since 0.11.0
  (`open_heading` is clearance, `way_to` memory), so help reports its mean unflagged; through 0.10.0 `open_heading` was
  the diagnostic ability `heading`, every help cell would have held under 20 questions, and help was flagged
  `diagnostic`. A `kind` entry still holds `counted` (its rows in the means) and `diagnostic` (none counted): a kind
  none of whose rows the means count reports its mean flagged `diagnostic`, as an ability does, and a kind with any
  counted row is not flagged. The headline's cells do not depend on `kind`. **`slice`** is the
  row's own, `answerable` or `unknown`. **`relevance`** is `unknown` for an unknown gold whatever its tag, else
  `critical` or `situational` by the row's `critical` tag (`score_stats.relevance`), so the answerable rows split into
  those that matter to the job and the rest, and the unknown ones are their own class. Both are derived from the rows'
  existing fields.

**The reported score, the age of the evidence and catastrophes (0.10.0).** Every mean above is of the *reported* score
(`score.reported`, the row's `score`): the penalised score `2s - 1`, except an answer that would have ended in harm, a
catastrophe, which scores minus its weight, 2 or 3 ([question-design.md](question-design.md), "How answers are graded").
The gates, `knowable` and the evidence verdict read the penalised score alone (the row's `score_gate`), so no harm weight
moves a verdict; `stbench check` is the same with the term in. Two reports stand beside the headline, from the same
resample:
- **`by.age_band`** and **`job_age_band`**: the reported score and the count by the age of the gold's evidence (`now`,
  `recent` up to 10 s, `task`, `earlier`; `none` for a row with no band, an `unknown` gold or a gold of an earlier build),
  overall and for each job: the memory curve, credit against how long ago the robot saw the answer. They are a split like
  `by.kind`: the headline's cells do not depend on them;
- **`catastrophe_rate`**: `{rate, ci, n, of, jobs}`, the rows whose answer is a catastrophe (`n`) over the harm rows answered
  (`of`: the family's kind is harm and the answer is not `unknown`), overall and for each job that has a harm row.

An aggregate or a result of an earlier build holds neither, and its report lacks both sections.

**References.** A family may declare a scripted reference that is not a shortcut, and `stbench check` prints the family
above it: `kind_range` for `ego_now`, and `map_reader` for the structure families `clear_path` and `open_heading`. The
map reader is an internal design diagnostic (the agent is never given a map), reported by `stbench check` on each family
and in no results table. `map_reader` raycasts the public
map from the delivered pose and answers it as it is; on a held-out layout it holds no map and answers from the
pose-keyed pool of other scripts' golds where the held-out rule allows. On a public place a structure family's score is a claim only about a system that holds no copy of the map; a
held-out place makes it a claim about every system. `map_reader` and `other_case` are left out of such a family's
shortcut and separation gates on every layout; `path_log`, `blind`, `form`, `questionnaire_blind` and, on its `memory`
rows, `current_view` and `current_view_prior` (without a pose key) stay in them.

**`path_log`'s own-log rules.** `path_log` answers `clear_path` (the detour its body made round something) and
`open_heading` (the heading of the longest straight run over the cells its body swept) from the robot's own poses
alone: no camera, no map, no pool of answers. A fixed rule like these is no proof of anything, and each could answer at
a loss (the detour scored -0.20 on the helper, below `all_unknown`, which `stbench check` reports as BROKEN). So each
stands only where it beats abstaining on other scripts, by one bar (`baselines_path_own.kept`): the records of the
family that the rule answers come from at least three scripts and five rows, and the lower end of the 95% interval of
its gain over `unknown`, a bootstrap over those scripts of `score.penalised` (the family's own credit: the ladder, the
graded credits, the regret of a heading), is above 0; else it answers `unknown`. Each prior record carries the rule's
answer as `own`, built by the function the system answers with, from the poses the robot was sent up to the ask. The
`_in_pool` twin of `path_log` keeps the rule unasked, as it keeps every pool's own best response.

The `scope` split is where a family that asks about the present shows. An `ego_now` question (where a thing in the
latest frame is from the camera) is `now`: no memory answers it, so its score sits in the `now` line and never in
`memory`'s, and a run's memory curve is read from the `memory` line and the lengths alone. The gate table of `stbench check`
(`docs/format.md`) says the same in its `scope` column (`now`, `memory` or `mixed`, from the oracle's rows), and its
`memory_needed` gate is `n/a` on a family with no `memory` row. Whether a shortcut beats `all_unknown` is read there by
the evidence verdict (`gates_evidence`): a gain over the gate's margin with a 95% interval above 0, in the whole group,
in a length tier or in a slice (answerable or unknown), the pairing and the interval those of `eval compare`.

`results.md` opens with the notes, then the system, options, sandbox and cost, the cases that did not finish
and, per layout:

- the headline (the mean of the `now` and `memory` columns below it, over the counted questions) with its 95%
  interval, **above the floor** (the headline less the unknown share, written in the row's name), the two columns,
  the cells' mean by their weights, the counted questions pooled and the own-path splits, each with its `n`;
- **per kind** (`by.kind`, 0.9.0): a table of harm, help, situation and reuse, each pooled over its own questions with
  its `n`, mean and 95% interval, under the per-ability table; a kind none of whose questions the means count is flagged
  `(diagnostic: counted in no mean)` (none is in 0.11.0: help is `open_heading` and `way_to`, both counted);
- **per ability**, one line for each (ability, column) cell, so clearance prints as two lines, one with the column `now`
  and one with `memory`: the ability and its column, its questions, those counted in the means, its mean over all of them with the interval, its mean
  over the counted ones, a diagnostic cell flagged, its weight in the headline, and the (job, ability, column) cells
  under 20 questions named with their weights (`n / 20`), and a line naming the cells the means count. The place's mix differs (the apartment's `now` column is distance and clearance, its
  `memory` column memory, relevance and clearance; the warehouse's `now` column is distance alone; downtown's is
  clearance alone), so `results.md` names the cells the place has, with their counts and weights;
- **per age of the evidence** (`### Per age of the evidence`, 0.10.0): a line for each band over every job of the layout,
  then for each job and band, with the questions, the mean reported score and its interval;
- **catastrophes** (`### Catastrophes`, before the outcome counts): the catastrophe rate overall and for each job, with the
  counts of catastrophes and of harm rows answered;
- **per family**, and **per length** in the order short, medium, long (`case.LENGTHS`, so the memory curve reads left to
  right), each with `n`, the mean and the 95% interval;
- **per relevance**: the answerable questions, the critical and the situational ones among them, and the unknown ones;
- the outcome counts, the confidently-wrong and abstention rates, the **credit bands by ability** (below), and the
  tokens, cost and wall time of the finished cases.

Beside the penalised headline stand two diagnostics (0.7.0), never in it. Both are read from the rows of `results.json`
alone and computed as the headline is (the same thin pairs, the same bootstrap over scripts), so each is the headline's
own statistic over other values:

- the **unpenalised credit** (`credit` per ability): the mean of the credit `s` of each answer, not `s - (1 - s)k`. On an
  unknown question the credit is 1 for a right `unknown` and 0 for any other answer. It shows what the penalty costs a
  system beside what its answers earned;
- the **abstention bound**: the score the same answers would get had the system answered `unknown` exactly where an
  answerable question's answer scored below 0 (`eval_report.bound_score`), that score becoming 0. A score on an unknown
  question is kept, so a wrong answer there stays negative: the bound never credits the right answer itself, and it
  is the most that knowing one's own precision could add on the questions that have an answer.

The **credit bands** (`eval_report.credit_band`) split each ability's questions by the credit `s` of the answer, one band for each tier of the credit ([questions.md](questions.md), "Credit"): `full` (s = 1,
within one pixel), `within need` (s at least 3/4), `right thing` (s at least 1/2, within twice the need), `direction` (an
answer with s at least 1/4: the right direction, or the right thing coarsely), `miss` (less), and `unknown` (every answer
of `unknown`, right or wrong). An answer is in the highest band it reaches. Each is a share of the ability's questions.
A band is read from the row's `s` alone, so an older run's `ladder` row falls in the highest band its `s` reaches.

## `eval compare`

Two to 26 results (a `results.json` or the run folder holding one); the first is the base. Per layout each result's
headline, its `now` and `memory` columns, a line for each ability (with its number of questions in the base, a
diagnostic flagged) and its family means, and each later result's difference from the base over the questions both
answered, paired on (case, qid), with a 95% CI from a bootstrap over scripts and the number of pairs
(`score_stats.paired_difference`). The headline's difference is the difference of the two headlines over the questions
both answered, and a column's the difference of that column (`headline=True`, resampled jointly, with the diagnostic
and thin abilities left out as in the results); every ability and family line's difference pools its own questions. A line under the notes states the headline's rule.

A second table gives the same for the **relevance classes**: the critical answerable rows, the situational answerable
ones and the unknown ones apart, so a difference that lies on the critical rows is not lost in the headline among the
others. Under it, one line for each later result says whether its **paired critical difference** is shown: its
difference from the base on the critical answerable rows has a 95% interval whose lower bound is above 0. That is the
rule declared before data for the conditioning comparison, a correct brief against a neutral one. The line reads `- B -
A on the critical answerable rows: shown (lower bound +0.0xx, over N pairs)`, or `not shown` with the bound, or `not
shown (no pairs)` or `not shown (one script, so no interval)`.

When the results differ in `during` the off arm holds no during rows, so every mean and every difference is over the end
questions alone and the table says so. Refused (exit 2, nothing printed): a result sharing no case with the first;
results scored on different compiles (a shared case with no question in common); results that differ in more than one
**lever group**: `agent` (the system, the reasoning effort, the memory and the compaction limit), `channel`, `brief`,
`during` or `text_only` (methodology rule 13: one axis at a time).

## `eval distance`

`stbench eval distance RUN [REFERENCE_RUN ...] --data D [--out FILE.md]` reports the offset questions (`ego_now`,
`ego_mem`) of a finished run folder, from `results.json` (each row's score and `range_m`), the case folders'
`answers.jsonl` and the dataset's golds. The credit hides how a system errs; this keeps it. It has two parts, and a
third follows (the credit sensitivity, below).

**The clearance sections** come first, one a layout (the credit is `graded` on the job's clearance C and the
gold's resolution). A line for each job and family, and one for all of the job's offset questions when it has more than
one family, with:

- `n`, and `answered` (those answered with an offset; an `unknown` or a malformed answer has no error);
- `C (m)`: the row's `need_m` (`tolerances.need_m`), the number the scorer uses, so the report cannot drift from it (the
  job's one clearance; the cell gives the lowest and the highest where questions differ);
- the mean score of the `n` questions;
- the share of the answered ones whose 2-D error (the distance in metres between the answered and the true offset, what
  the scorer tests) is within `rho` (the row's `resolution_m`: one pixel, full credit), within **N** and within **2N**,
  each question against its own and met up to `score.EPS` as the scorer's anchors are;
- the **harmful share**: the share of the answered errors that point away from the camera along the gold's bearing (the
  thing placed farther than it is), the side whose need is halved;
- the **median error** in metres over the answered ones.

A row counts when it is `graded` with `need_m` and `resolution_m`; a row of an older compile whose ladder is exactly
`[C/2, C, 2C]`, to 1e-6 m (`eval_distance.clearance_ladder_of`), is read as before, by its rungs, with no `rho` or
harmful share. Every other question is left out of these sections: a four-rung or relative ladder (a compile before 0.7.0
has only those, so its report has no clearance section), a row the resolvability fallback widened. None needs a
`range_m`.

**The band sections** keep the relative reading as a diagnostic. Per layout and family, a line for the bands `near`
(under 6 place units), `mid` and `far` (from 16 units), at the unit of each question's job, and one for all, with:

- `n`, and `answered` (those answered with an offset);
- the mean score;
- the **range error**: the median of (answered ground range - true ground range) over `range_m`, signed, so a negative
  number is a systematic under-estimate;
- the **error / range**: the median 2-D distance between the answer and the true offset over `range_m`, the scale
  the 0.6.0 credit rungs were fractions of;
- the **bearing error**: the median angle in degrees between the answered and the true direction;
- with reference runs (the scripted `kind_range`, the typical range of the kind, and `all_unknown`), the pooled score
  difference from the better reference over the questions both answered, with a 95% CI from a bootstrap over scripts.
  The better reference is chosen per layout and job: the run whose mean score is larger over the questions the run and
  every reference scored there, and the report names it and each mean. A
  system that abstains scores what `all_unknown` does, so it is never above the reference, however `kind_range` scores
  (`kind_range` scored under `all_unknown` on the AMR and the vacuum in 0.6.0).

The golds are checked: each scored row's qid must be in `data`'s gold of its case, or the command exits 2. The range is the 3-D slant, so a drone 15-20 m up (unit 1.0 m) is never in `near`: its split leans toward `mid` and
`far`, and is reported as it falls. A row without `range_m` (a run before 0.6.0) is left out of the band sections.
Layouts are not pooled.

### Credit sensitivity

`stbench eval distance` ends with the run's credit sensitivity (`eval_sensitivity`; `--out` writes it with the rest):
the headline re-scored with the need of the gold rows multiplied by 0.5, 1 and 2. A job's clearance C and re-find
radius R only set those needs, so scaling them is scaling C or R, and the same answers are scored again with no
recompile. It reads the run's `results.json` rows, the case folders' `answers.jsonl` and the golds, scores each row
through `score.row` on the scaled gold (the scorer's own credit, never a copy of it) and aggregates each layout as the
run did (`ability_counts.thin_rule`). There are two lines, each scaling one thing:

- **C**: every offset, number and heading gold (`ego_now`, `ego_mem`, `person_mem`, `clear_path`, `closest_person`,
  `closest_approach`, `open_heading`) has its `need_m` multiplied, and a number's or a heading's `threshold_m` (the body's
  length plus C) moves with it, and so does the margin of a catastrophe rule of an `over` or `radial` row (`max(need, 2
  rho)`, made again from the new need); the resolution `resolution_m` stays, because the camera's pixel does not move with the job's margin. A
  ladder offset gold of an older compile has its rungs multiplied;
- **R**: every place gold (`object_where`) has its `need_m` multiplied, its widening kept in `resolution_m`. A ladder
  place of the form `[R/2 + w, R + w, 2R + w]` has its R part multiplied and its widening kept; a gold of another form,
  such as a compile before 0.7.0's relative ladder, is left as it is.

No other credit is scaled: a list's match radius, `distance_between`, the `route` credit of `way_to` (its need is the
field's cell and the job's C together, not C alone), the change and self families, a time ladder (seconds). The report gives, per layout and line, the headline and its 95% interval at each
factor, how many questions were scaled, and a line saying whether the factor 1 headline is the published one. It is not,
when the answers or golds are not the ones the run was scored on.

## Looking at a run: `stbench view`

`stbench view` turns a run into one HTML page to troubleshoot a system with: the cases it ran, and for each case what it
was asked, what it answered, the gold and how that scored, the stream it was sent frame by frame, and the transport log
of an agent. It reads the run folders and the dataset as they stand and writes the one file (and the depth pictures
beside it, below), so it works on a run that is still going, was killed or has no `results.json`.

```sh
stbench view runs/luna --data data                # writes runs/luna/view.html (and view.html.frames/), prints its path
stbench view data/runs/luna-arms --out luna.html   # RUN_DIR under DATASET/runs/: --data is found by itself
```

RUN_DIR is a run folder of `eval`, a folder of run folders (the arms of `--arms`, each listed by its folder's name, with
a selector on the page) or the folder one `stbench run` wrote. The dataset is the one the run ran on, given with
`--data` or found as the nearest folder above RUN_DIR that holds `private/` and `scripts/`; without either the command
exits 2. It also exits 2, writing nothing, when RUN_DIR holds no run and when `--out` is under the dataset's `scripts/`
or `private/`: the page holds the gold, which is public in this release and which a system must never be handed.

The page is one self-contained file: no server, no network, no external script, style or font, and it opens from disk
(the pictures it links are other files, below). Its header gives the system, the channel, the brief, whether the during
questions were asked, the reasoning effort and model where the run states them, the headline per layout as
`results.json` records it, with its `now` and `memory` columns and their counts and each ability's mean and count
(a diagnostic flagged), the cases finished of all, and the
tokens, cost and wall time of the cases that ran. The headline is the file's and the viewer computes no statistic of its
own: eval writes `results.json` when a run begins (every case pending) and when it ends, so a run in progress says that
its headline is not there yet. The cases finished are counted from the case folders, and a banner says that an
unfinished run is not a benchmark score and whether `results.json` is absent or is only that stamp from the start. Below
it a table lists the cases (id, layout, job, length, status, the questions and their outcomes, mean score, wall time,
tokens, cost), sortable by any column and filterable by text (case, job, status, and the questions' and answers' words),
status, layout, job, length and "has wrong answers". A status is `complete`, `ended: <why>`, `invalid`, `refused:
<why>`, `unfinished (no summary.json)` or `not started`; a case folder that `results.json` does not list is shown and
marked, and left out of the totals.

A click opens a case in place (the address becomes `#case-<id>`, which opens it again; `#case-<id>@<arm>` in a folder of
arms): its facts (the script, the window of the script's tasks, the failures, the timing, the tokens and cost) and its
questions in the order asked, each with the solver's answer in words (a position as `(x, y)`, yes or no, a clock, `unknown`,
a list of places) and how far it is off, the gold with its credit and tolerance, the outcome, the score, whether the
question is task-critical, the slice, and the pictures of the keyframes the gold is read from (linked from the dataset's
`scripts/<sid>/sensors/rgb/` by `file:` address; a missing picture is said so, and one the browser cannot load, as when the
page is served over http, shows its link as text with a note that the pictures load when the page is opened from disk). A
question the run did not ask (`--during off`) is listed as not asked. A case then holds two collapsed sections, the stream
it was sent (next paragraphs) and the transport log of an agent (`codex.calls.jsonl` and `codex.summary.json`, or any
`<name>.calls.jsonl` and `<name>.summary.json`): each record is a line that opens into its prompt,
its tool calls with their arguments and results, its reply, its token counts, and any error or warning, and a box shows
only those. The raw `events` and the `argv` of the journal are left out of the page and a text over 20,000 characters is cut
(the page says so); a system with no journal has no log. The page never scrolls sideways: the tables scroll in boxes of
their own and long text wraps.

**The stream, frame by frame** (collapsed; its frames are built when it is opened) shows what the harness delivered to
the system, one step at a time, to see what the robot's brain was given and when. It has previous and next buttons, buttons
for the previous and next look window, a slider over the case's frames, the keys (left and right arrows, Home and End, Page Up
and Page Down while it has the focus) and a counter, `frame 37 of 412` (the step `k` plus one). For the frame it shows the
step `k` in the case (counted from 0, as the system numbers it), the case's clock, the base's position and heading, the
camera's position and pitch, the status fields, the look window it falls in (the question compiler's cut, up to 8 steps, after
each decision point and each task's last step; it is never sent: the system gets one frame at a time) and the task it
belongs to, and the pictures of that step only: the camera
image alone on `rgb_pose` (the default: the agent is sent no depth, so none is shown), or on the `detections` channel (which sends no
image) the depth image and the frame's detections, one line each (kind, attributes and state, map and camera centres, size,
box; their keys, fresh for each frame, and the poses' quaternions are not shown). The steps are the stream's own:
`stream.delivered` is the function `run_case` builds a case's window with, over the same files and for the run's channel
(the case's `summary.json`, else the run's; a run that records none is shown on the default channel, `rgb_pose`, and the case says so), so they
are the steps the system received and not a reading of the transport log. A strip under the slider marks where each look window
ends (a thin line), where each task begins (`T1`, `T2`, ...) and where the questions were asked: a during question at the
step it was asked after, which ends a look window, and the end questions after the last frame. A mark goes to its frame, and
the question labels in a frame's facts (`Q3 ego_now`, numbered as the table above numbers them) take the reader to
that question's row and light it up; a question the run did not ask is dimmed.

The pictures are linked from the dataset by `file:` address, with the fallback of the evidence frames (their link as text and
a note when the browser cannot load them). A depth image of the dataset is a 16-bit PNG that a browser shows nearly black, so
for each depth image the page shows (only a run whose channel sent depth shows any) `stbench view` draws an 8-bit grey picture from the sensor's values (`view_depth`): 0,
no reading, is black, `near_m` of `sensors/camera.json` is white (255), `far_m` black (`view_depth.depth_view`; no agent is sent this picture), linear in metres
between, in the camera's unit; the page states the range under the picture. They are written into a folder next to the page
named by its whole name plus `.frames` (`luna.html.frames/<script id>/<keyframe>.png`), one for each depth image of the
cases shown however many arms ran them, and linked by absolute `file:` address like the rest (move the page and the folder
together and the links break, as the evidence frames' do when the dataset moves). A picture that is not older than its
depth image is not drawn again, so a second view is quick; delete the folder to draw them anew. Nothing is written under
the dataset. Each (case, channel) stream is in the page once, however many arms ran it, as numbers, and only the current frame's
pictures are loaded.

Everything the run and the dataset hold is untrusted text (a model's answers, a tool's results, a status or an attribute): it
is embedded as JSON in a script block with `<`, `>` and `&` escaped, and the page puts it on the screen only as text. The
page is a function of the folders, with no clock in it, so two views of one run are the same bytes. The reader is `view.py`
(`view_questions.py`, `view_log.py`, `view_frames.py`, `view_depth.py`, `view_files.py`); the page is `view_page.py` with
`src/stbench/viewer/`.

## Fast loops: scoped and reused checks, the fast tier (0.11)

A change is tested at the cheapest tier that can show it. A full row of the reference agents costs most of
a weekly quota, and a full `stbench check` takes about 19 minutes, so nothing below runs either to read a small change.

| Tier | What | Cost | Use |
| --- | --- | --- | --- |
| 0. Rescore | the stored answers of a run through the current scorer and aggregate | free, seconds a run folder | every scorer change: the answers stay, the rows and headlines are made again |
| 1. Free check | `stbench check`, scoped to what changed (`--select`, `--cases`), with the stored answers of the systems that do not read the scorer (`--answers`, `--reuse`) | free, minutes | every compile and scorer change |
| 2. Fast tier | the 24 cases below, sol (medium) and luna (xhigh) | about 45 minutes, a few quota points | every change to the questions, the prompt or the frames |
| 3. Publish rows | the whole dev root | the weekly quota | only to publish |

Tier 0 is a local script, not shipped with the package, that runs each finished case's `answers.jsonl` through
`cmd_run.write_run` into a scratch copy and `eval_results.build` over it, prints each layout's headline beside the
stored one and writes nothing into the run folder unless asked; it lives outside the package (it is a function of the
stored files and the scorer, no more).

### A scoped check: `--select`, `--cases`

```sh
uv run stbench check --data D --verdict evidence --workers 8 --select job=helper/1 --out check-helper.json
uv run stbench check --data D --cases ID,ID,... --out check-cases.json
```

`--select KEY=VALUE` (`layout`, `robot`, `job`, `length`) and `--cases ID,...` are `stbench eval`'s, through the same
code (`case_select`): a key given twice is either value, different keys must all hold, and the two options together are
the intersection. A check refuses, exit 2 and before any system runs, an unknown key, an id the dataset lacks as a case
that asks, and a selection that leaves no case.

- **Whole groups give the full check's numbers.** The gates are per (layout, job), and a system's options look only at the
  cases of its own layout and job, so `--select layout=...`, `robot=...` or `job=...` checks each group it keeps exactly as
  the full check checks it: the same gate rows, scores, balance entries and verdict for the group (a test compares them,
  run twice).
- **A selection by length or by case id narrows a group itself.** Its priors and `other_case`'s partners are then built from
  the cases left, so that reading is its own (it is refused when a group has no other script's case left to feed
  `other_case`), and it is not the full check's reading of those cases.
- **It says what it was scoped to.** The printout begins and ends with a `SCOPED CHECK (...)` line (the selection, the groups
  and cases read of all of them), and the report holds a `scope`. A scoped check never writes `DATASET/check_report.json`,
  which is the whole dataset's and what `stbench export` reads: it writes a report only to `--out`.

### Stored answers: `--answers DIR`, `--reuse DIR`

```sh
uv run stbench check --data D --verdict evidence --workers 8 --answers A > check.txt   # the first check
uv run stbench check --data D --verdict evidence --workers 8 --reuse A                 # after a scorer change
```

`--answers DIR` writes each scripted system's answers to `DIR/<system>/<case>.json` (`check_answers`). `--reuse DIR`
reloads the answers of the systems whose answers do not pass through the scorer (`Baseline.reads_scorer` false: `oracle`,
`memory_table`, `current_view`, `other_case`, `all_unknown`, `kind_range`, `latest_table`, `first_table`) and runs the rest
(`blind`, `path_log`, `questionnaire_blind`, `current_view_prior`, `map_reader`, the in-pool twins and, in every check,
the form adversary), whose best responses are chosen by the scorer. The printout and the report name the systems reloaded.

- A system is reloaded only when every case of the check has a whole file holding an answer to exactly the case's questions
  by qid, so the files of an earlier compile (qids are redrawn by a compile) are run again and never scored as answers.
- A system that reads other cases (the priors, `other_case`) is reloaded only when its case's group holds the same cases as
  when the file was stored, so a check narrowed by length or case id runs those.
- `--reuse` is for a change to the scorer and to nothing else. It cannot see a change to a system or to the data under the
  same qids: after one, `--answers` into a fresh folder.

### The fast tier

24 dev cases, chosen by a fixed rule and never by a model's scores: per job, the scripts that host both a short and a
medium case that ask a question are ordered by `stable_hash("fast-tier-0.11", script id)`, and the first 3 are taken, each
with its short and its medium case (4 jobs x 3 scripts x 2 cases). The 24 ids are written once, on one line, to the fast
tier's file and do not change within 0.11. Sol and luna run them with `stbench eval --cases`, and a change of about 0.10 in
one model's pooled score (some 220 questions) is what they can show; run-to-run noise is large, so nothing smaller is read.

```sh
uv run stbench eval codex:gpt-6.1-sol --data D --cases $(cat fast-tier.txt) --channel rgb_pose --reasoning medium --workers 9 --out OUT/sol
```

## The pilot report: `stbench pilot report`

`stbench pilot report` turns the pilot's `results.json` documents into one ASCII markdown report, a function of those
documents alone (the bootstrap seeds are fixed). The arithmetic is `power.py` (numpy and `score_stats` only); the
section per option is `cmd_pilot.py`.

```sh
uv run stbench pilot report [--runs R ...] [--leak ON OFF]... [--brief BASE OTHER]... [--release PLAN.json]
                            [--out FILE.md]
```

Each argument is a `results.json` or the run folder holding one. Give at least one option; the report has a section for
each.

| Option | Reports |
| --- | --- |
| `--runs R ...` | the pilot rows: each document is during on with the correct brief. Per agent (system, memory, compaction limit and reasoning effort) and job: the scripts, the cases and questions a case by length, sigma (the SD of the score over questions, also per family), the design effect and rho from 10 scripts, the headline with its 95% interval and the `now` and `memory` columns (the thin pairs the documents record left out of their means), how the cases ended, and per case-run by length the mean tokens (in, of them cached, out), wall time and notional USD |
| `--leak ON OFF` | D2: the pair differs in `during` alone. d = the mean of (on - off) of the penalised score (`score_gate`, never the reported score) over the end questions, paired by (case, question), with the script bootstrap's 95% interval, the pairs and the scripts. The verdict reads the pairs whose gold is answerable, per job and pooled; the pairs whose gold is `unknown` are an abstention shift with no verdict, and each (job, family) has its rows (`pilot_leak`; "D2's verdict" below). Repeat for each job; the pairs must be of one agent |
| `--brief BASE OTHER` | the brief swap: BASE is the correct brief, OTHER the neutral one (or, in a 0.4.x result, the swapped one), differing in `brief` alone. BASE - OTHER per job on the task-critical rows and on the rest, with the cases by length (the report says when most are short). Descriptive: no gate |
| `--release PLAN.json` | the count rule below, per job: sigma, m, K target, K, the predicted half-width, frames, GPU-hours and points, then the totals and f |

Results do not record the quota, so the report has no points: the plan's `points_per_script` comes from the driver's
log (a script's tokens by length, from the per-length means, times the measured rate).

### The design effect and rho

R = T / N is a pooled mean of the penalised score over N rows (each column of the headline is one), and the rows of
one script (its windows share a world and a path) are not independent. With S scripts of n_s rows summing to T_s:

```text
cluster variance  V_c = S / (S - 1) * sum_s (T_s - R n_s)^2 / N^2     (None for S < 2)
naive variance    V_n = s^2 / N                                      (s^2 the sample variance, n - 1)
design effect     DE  = V_c / V_n
Kish ICC          rho = (DE - 1) / (m_bar - 1), clipped to [0, 1]     (m_bar = N / S rows a script)
```

DE and rho are reported per (agent, job) with at least 10 scripts and scores that vary. Clustering by question or case
instead of by script makes the interval too narrow; the interval of every headline is the script bootstrap
(`score_stats`; the headline's is the bootstrap of the mean of its two columns), and rho enters the count only through
the formula below.

### The count rule

For a pooled mean over K scripts of m questions each, with sigma the SD of the score over questions:

```text
variance       sigma^2 (1 + (m - 1) rho) / (m K)
half-width     h(K) = 1.959964 * sigma * sqrt((1 + (m - 1) rho) / (m K))
scripts needed K_target = ceil( (1.959964 * sigma / h*)^2 * (1 + (m - 1) rho) / m )
```

With sigma 0.8, m 20, rho 0.05 and h* = 0.10 that is K = 24 (h(24) = 0.0999, h(23) = 0.1021). The release's inputs:
- **sigma** of a job: the larger of the agents' SDs over the job's pilot rows. A job without pilot rows takes the plan's
  `fallback.sigma`.
- **m** of a job: q_short + q_medium + q_long, the mean questions of a pilot case of each length. A length with fewer
  than 3 pilot cases takes q_short (the mean of every case if there is no short one). A job without pilot rows takes
  `questions_per_script` from the plan.
- **rho**: the largest Kish ICC over the (agent, job) pairs with at least 10 scripts, at least 0.05; without such a pair,
  `fallback.rho`.

The budgets cut the counts: K_j = max(k_min, floor(f * K_target(j))) for the largest f <= 1 whose scripts fit both
`render_hours` (the sum of K_j * frames_per_script * seconds_per_frame) and `points` (the sum of K_j *
points_per_script). K only changes where f * K_target(j) crosses a whole number, so f is the largest such breakpoint that
fits (exact in fractions); the report prints it, and the predicted half-width at each K_j. When even k_min scripts a job
overrun a budget, K is k_min everywhere, f is 0 and the report says OVER BUDGET. The rule reads no release outcome.

### D2's verdict

d = on - off over the end questions, with `power.leak_verdict` against the margin 0.05: **pass** when the interval's upper
bound is at or below the margin (a bound on the margin passes), **fail** when d is above it, otherwise **inconclusive**
(the check cannot exclude a leak of the upper bound). The off arm asks no during question, so during rows are never
paired. The report reads the pairs in two ways (0.11; `pilot_leak`):

- **The leak** is the pairs whose gold is answerable, a thing the agent saw: did asking during the job change what it
  recalls of it? The pooled verdict and the per-job rows read these alone, and on a fail the jobs whose own answerable d is
  above the margin lose their during families.
- **The abstention shift** is the pairs whose gold is `unknown`, a thing never seen, where the right answer is to say so.
  Asking during the job changed how willing the agent was to say it, which is calibration and not recall, so the report
  gives its d and interval per job and pooled with no verdict, and beside them the d over all pairs, the reading the 0.9.0
  report published. The prompt's sentence that a named thing may never have been seen is meant to shrink it.
- **By family**, each (job, family) has a row for each gold it has pairs for, so a family whose own d rises is seen where
  it is (the vacuum's cords, which its during `tangle_ahead` questions also cover, are the one to read).

On the 0.9.0 leak run (sol medium, 193 pairs) the answerable pairs give d +0.001 [-0.063, +0.063] over 171 pairs
(inconclusive: the check excludes a leak above about +0.06), the `unknown` pairs +0.364 [+0.087, +0.737] over 22, and all
pairs +0.043 [-0.030, +0.118]: the vacuum's failing +0.178 of that report was two `unknown`-gold pairs. The report takes
a few seconds.

### The plan

`PLAN.json`:

```text
{"target": 0.10, "k_min": 6, "render_hours": 30, "points": 60,
 "fallback": {"sigma": 0.95, "rho": 0.10},
 "jobs": {"<job>": {"frames_per_script", "seconds_per_frame", "points_per_script",
                    "questions_per_script" (only for a job with no pilot rows)}, ...}}
```

### Refused

Exit 2, saying why, nothing printed or written: no option; a document `eval_results.load` refuses; a `--runs` document
with during off or another brief; `--runs` documents scored on different compiles, or two documents of one agent holding
the same case; a pair that differs in more than its one lever (or that `eval compare` would refuse), a `--leak` pair
given off first or of two agents, a `--brief` pair whose BASE is not the correct brief; leak pairs with no end question
in common; and a plan that lacks a number, or a job with no pilot rows and no `questions_per_script`.
