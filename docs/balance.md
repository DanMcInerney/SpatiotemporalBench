# The balance rule

No constant answer may score above `all_unknown` on a family (methodology rule 5). A system that has never
seen the world must not gain by always saying the common thing: the yes of a coin that is not fair, the position most
golds share, the empty set. The compile draws the answer classes to prevent it, and `stbench check` reads whether the
draw worked, from the golds alone, never from a system's answers (`suite_balance`).

## The classes and the draw

Each gold row has an `answer_class` (`records.ANSWER_CLASSES`), the class its draw balanced; null where the family
declares none (`ego_now`). `records.BALANCE_POOLS` names the pools of each classed family, and `check_gold` refuses a
class that is in none of its family's pools.

| Family | Pools (classes) | The draw |
| --- | --- | --- |
| `object_where`, `ego_mem`, `way_to` | `answerable`, `unknown` | the unknown slice 0.20 of the moments (`questionnaire.UNKNOWN_PERCENT`; for the two look families, of the look points) |
| `clear_path` | `answerable`, `unknown` | no slice: a corridor is seen or not. The job keeps a share of the `unknown` golds (`look_unknown_keep`): the helper 35%, the drone 75%, the vacuum all (the AMR asks none) |
| `task_things` | `empty`, `items` | the empty golds thinned to the job's keep share (`TaskThings.empty_keep_percent`: vacuum 20, helper 26, AMR 43) |
| `closest_person`, `closest_approach`, `approach_mem` | `person`, `no_one` | the world's: a person is in range or not (`approach_mem` has no `no_one` row in the release) |
| `tangle_ahead` | `empty`, `items` | the world's |
| `object_was` | `earlier`, `later` | a fair coin, `later` 0.50 (`COIN_PERCENT`); asked by no job |
| `same_place` | `yes`, `no` (= `moved` and `vacated`) | a fair coin, `no` 0.50; asked by no job |
| `ego_now` | none | none: the best constant offset scores far under `all_unknown` |
| `open_heading` | `answerable` only | the world's: a point where no heading was seen asks nothing, so there is no `unknown` gold. The family has one pool, and the check reads its best constant heading against `all_unknown`, which scores 0 on every row of a family with no `unknown` gold |

## The check

For each (layout, job, family), `suite_balance.read`:

1. Takes the **best constant**: the one answer, of the types and candidates `answer_candidates` offers (the better of
   yes and no, the best grid point or offset, the empty set), with the best mean penalised score over the golds of the
   family's questions of the other scripts. It is chosen leaving each script out in turn and scored on the script left
   out, so a constant that is only the best of many on the golds it was chosen from gains nothing. This is the
   zero-feature rule of the form adversary (`form_check`), without its support floor and its inner held-out round: the
   outer round is already the held-out score, and a rule that abstains on a thin pool would make the check pass for
   nothing. A time gold is learned as a lag, and a layout-constant family (reference `other_case`) from public layouts
   only, as everywhere else.
2. Takes its **gain**: the paired difference of the constant's rows and `all_unknown`'s on the penalised score (pairs on
   (case, question)), with a 95% interval from a bootstrap over scripts (2000 resamples, seed 0). The unit that
   resamples is the script, because a script's cases are windows of one world.
3. **Fails** the family on that job when the gain's mean is above 0 and the interval lies above 0. A group **too thin
   to read** cannot fail: when some script's held-out pool (the other scripts' golds of the family) is under the support
   floor of the priors (`blind.supported`: 5 golds of 3 scripts, which is fewer than four scripts or a handful of
   questions), the best constant is the best of many on a few correlated golds. It is reported as `thin` with its
   numbers and the reason. A group of one script has no interval at all.

A family above 0 does not ship on that job. The 0.11.0 and 0.11.1 releases read 23 groups each: none is thin and none
fails. The verdict is part of `stbench check --verdict evidence` (its exit code and `passed_evidence`); the printout has
one `BALANCE ok|thin|FAIL layout job family: ...` line for every family under any verdict, before the gate table's
`EVIDENCE VERDICT PASSES|FAILS` line, which counts the balance rule and the broken instruments too, and the report's
`balance` holds the entries (`constant`, `mean`, `ci`, `n`, `scripts`, `thin`, `fails`).

## The shares

The asked class shares of each family are reported against the draw (`class_shares` in the group's numbers and a `class
shares` line in the printout) and are **not gated**. At about 40 questions a job a fair coin's yes share has a standard
deviation of 0.08, so "0.5 ± 0.05" would fail a good draw about half the time; likewise 0.20 ± 0.03 at 80. The binding
test is the best constant on whatever shares came out.

## No class cells

The evidence reading of the shortcut-type gates (`gates_evidence`) has no cell for a declared class, only the group, each
length and each slice. A class cell holds just the rows whose gold has that answer, so a system that always gives that
answer scores +1 there against `all_unknown`'s 0: the time-blind notebooks of `time_needed` (always the earlier state, or
the later; always "yes" on `same_place`) would fail both change families on every job. A gain on one class that does not
net positive over the group is not a shortcut, and the pooled, length and slice cells still catch one that does.
