"""A decision point's `response` (`timeline_ego.DecisionPoint`): what the robot did after the ask, an optional key of
the ego that an ego without it does not carry."""

from __future__ import annotations

from dataclasses import replace

from builders import minimal_timeline
from stbench import timeline as T
from stbench.timeline_ego import DecisionPoint, ego_dict, ego_from_dict


def with_decisions(*decisions: DecisionPoint) -> T.Timeline:
    tl = minimal_timeline()
    tl.ego = replace(tl.ego, decisions=list(decisions))
    return tl


def test_an_ego_without_responses_is_written_as_it_was():
    """Contract: a decision point with no response is written as `{k, kind}` and nothing more, so the file of a
    timeline whose jobs give no responses (every script made before the coins) has the same bytes as it had. Regression
    caught: a `response: null` key (or any other default) written for every decision, which would change the bytes of
    every such script's `timeline.json`."""
    tl = with_decisions(DecisionPoint(1, "before_doorway"), DecisionPoint(4, "room_done"))
    assert all(set(d) == {"k", "kind"} for d in ego_dict(tl.ego)["decisions"])
    assert '"response"' not in T.dumps(tl)


def test_a_response_round_trips_and_belongs_to_the_ego_hash():
    """Contract: a decision point's response survives `ego_dict` and `ego_from_dict`, and a save and a load of the
    timeline (equal timelines, byte-identical files); only the decisions that have one carry the key; a response changes
    the ego hash and leaves the world's. A file written without the key loads with `response` None. Regression caught:
    a response dropped on load (the subgoal and the checks would read nothing), written in a different place, or left
    out of the hash that tells two egos apart in a run."""
    plain = with_decisions(DecisionPoint(1, "before_room"), DecisionPoint(4, "room_done"))
    tl = with_decisions(DecisionPoint(1, "before_room", "skip"), DecisionPoint(4, "room_done"))
    assert [d.get("response") for d in ego_dict(tl.ego)["decisions"]] == ["skip", None]
    assert ego_from_dict(ego_dict(tl.ego)) == tl.ego
    assert T.loads(T.dumps(tl)) == tl and T.dumps(T.loads(T.dumps(tl))) == T.dumps(tl)
    assert T.dumps(tl).count('"response"') == 1
    assert tl.ego_hash != plain.ego_hash and tl.world_hash == plain.world_hash
    assert [d.response for d in T.loads(T.dumps(plain)).ego.decisions] == [None, None]
