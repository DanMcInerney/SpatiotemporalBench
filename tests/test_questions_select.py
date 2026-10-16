"""A candidate question's identity (`questions_select.Cand.key`)."""

from __future__ import annotations

from stbench.questions_select import Cand


def test_a_during_candidate_is_keyed_by_its_decision_point():
    """Contract (map Q-H11): two during candidates that differ only in their decision point have different keys, so
    ordering and the subject check never merge them, while an end candidate's key is still the family and slots alone.
    Hand values: a `path_ahead` candidate with slots {"cap": 3.0} at keyframes 21 and 49 has the keys
    '["path_ahead",{"cap":3.0},21]' and '["path_ahead",{"cap":3.0},49]'; an end candidate has
    '["object_where",{"event":"x"}]'. Regression caught: every during question of a case sharing one key (its slots
    hold only the cap), so one decision point stands in for all of them."""
    at = [Cand("path_ahead", {"cap": 3.0}, {"type": "unknown"}, decision_k=k) for k in (21, 49)]
    assert [c.key for c in at] == ['["path_ahead",{"cap":3.0},21]', '["path_ahead",{"cap":3.0},49]']
    assert Cand("object_where", {"event": "x"}, {"type": "unknown"}).key == '["object_where",{"event":"x"}]'
    assert len({c.key for c in at}) == 2
