"""`closest_approach`'s scope, evidence and pairing of a person's two sightings (the validity fixes after lane J's
check and the final review's S3): the robot's velocity is delivered with every step, so a row whose person stands is
`now` (frame k and its velocity give the gold) and only a row whose person moves is `memory` (it needs their earlier
sighting), and a person's velocity is estimated only when the sensors, which give no identity, pair the sightings
unambiguously.
Hand-worked on `approach_kit`'s scene: keyframes are 1.5 s apart (12 is delivered as 18.0 s, 13 as 19.5 s, 11 as
16.5 s), the robot drives north at 0.3 m/s (its camera floor point (3.0, 1.13) at the look point 13), a person is a
0.34 x 0.56 m footprint or a floor point. Evidence is 1,500 px; a glimpse is the 200 to 300 px a test sets."""

from __future__ import annotations

import pytest

import approach_kit as ak
import person_kit as pk
from ego_kit import K
from stbench import questions, questions_approach
from stbench.questions_approach import candidate, closest_approach

STILL_GOLD = 0.5831   # a floor point 0.3 m east and 2.0 m ahead, the camera 1.5 m closer after 5 s: hypot(0.3, 0.5)
WALKING_GOLD = 0.3007   # from (3.3, 4.43) at 16.5 s to (3.3, 3.13) at 19.5 s: 0.4333 m/s south against 0.3 m/s north


def cand(s, k=K):
    return candidate(pk.cut(s, k), k)


def row(s):
    """The gold row of the look point's candidate, with its evidence age."""
    w = pk.cut(s, K)
    (_,), (g,) = questions.records([candidate(w, K)], w)
    return g


def test_a_still_persons_row_is_now_with_the_robots_delivered_velocity_and_a_walkers_is_memory():
    """Contract (final review S3; DECISIONS 2026-10-14, "The final review's repair pass"): every delivered step carries
    the robot's own velocity (keyframe 13's is 0.45 m north in 1.5 s, 0.3 m/s), so a person who stands, or has no
    earlier sighting within 4 s to move against, is answered by frame 13 alone: scope `now`, the ask the only evidence,
    age 0.0 and band `now`. A person who walks needs their earlier sighting (keyframe 12, 18.0 s, the ask 19.5 s):
    scope `memory`, evidence 12 and 13, age 1.5 and band `recent`. Regression caught: a still person's row still
    `memory` (the robot's velocity treated as needing the frame before), a walker's tagged `now` (its gold needs the
    earlier sighting), or an age counted from a frame the evidence does not hold."""
    still = ak.scene()
    pk.put_person(still, "A", (3.0, 3.63), [12, K])
    walking = ak.scene()
    ak.walker(walking, "A", (3.3, 3.93), (3.3, 3.13))
    alone = ak.scene()
    pk.put_person(alone, "A", (3.0, 3.63), [K])
    for s in (still, alone):
        g = row(s)
        assert g["scope"] == "now" and g["evidence_k"] == [K] and g["age_s"] == 0.0 and g["age_band"] == "now"
    g = row(walking)
    assert g["scope"] == "memory" and g["evidence_k"] == [12, K] and g["age_s"] == 1.5 and g["age_band"] == "recent"


def test_no_one_stays_now_because_frame_k_alone_answers_it(monkeypatch):
    """Contract: a "no one" (no evidence person in the ask's frame, no person row of 256 px or more within 10 m) has the
    cap, 10, for its gold whatever the robot's velocity, which frame k alone shows (nobody within the cap), so it is
    `now` with the ask as its evidence, age 0 and band `now`; a 200 px glimpse (under the screen) changes nothing.
    Regression caught: a "no one" made `memory` for a velocity it does not use, or an evidence set that reaches back."""
    monkeypatch.setattr(questions_approach, "NO_ONE_KEEP_PERCENT", 100)
    empty = ak.scene()
    glimpse = ak.scene()
    pk.put_person(glimpse, "B", (3.0, 6.0), [K], px=200)
    for s in (empty, glimpse):
        g = row(s)
        assert g["gold"] == {"type": "number", "value": 10.0} and g["answer_class"] == "no_one"
        assert g["scope"] == "now" and g["evidence_k"] == [K] and g["age_s"] == 0.0 and g["age_band"] == "now"


def test_the_evidence_holds_the_earlier_sighting_only_when_the_person_moves():
    """Contract: a person's earlier sighting is evidence only when the gold needs it, which is when their speed is over
    `V_STILL`; the robot's velocity is delivered with the ask's step, so the frame before the ask is never evidence.
    By hand: A is seen at keyframes 11 (16.5 s) and 13 (19.5 s), 3.0 s apart, and not at 12. Moving 1.3 m south in 3.0 s
    (0.4333 m/s) against the robot's 0.3 m/s north, the gap of 2.0 m closes at 0.7333 m/s: 0.02 m at 2.7 s, hypot(0.3,
    0.02) = 0.3007; the evidence is 11 and 13 and the age 3.0 s. Moving 0.6 m in 3.0 s (0.2 m/s) is standing: 0.5831,
    the evidence 13 alone, the age 0.0 s, scope `now`. Regression caught: a still person's row aged by a sighting it
    does not use, a walker's evidence that lists the frame before the ask, or a frame that holds nothing the gold
    reads."""
    walks = ak.scene()
    ak.walker(walks, "A", (3.3, 4.43), (3.3, 3.13), seen=(11, K))
    g = row(walks)
    assert g["gold"] == {"type": "number", "value": WALKING_GOLD} and g["scope"] == "memory"
    assert g["evidence_k"] == [11, K] and g["age_s"] == 3.0 and g["age_band"] == "recent"
    stands = ak.scene()
    ak.walker(stands, "A", (3.3, 3.73), (3.3, 3.13), seen=(11, K))
    g = row(stands)
    assert g["gold"] == {"type": "number", "value": STILL_GOLD} and g["scope"] == "now"
    assert g["evidence_k"] == [K] and g["age_s"] == 0.0


@pytest.mark.parametrize("in_k, in_before", [(("A", "B"), ("A",)), (("A",), ("A", "B")), (("A", "B"), ("A", "B"))])
def test_two_evidence_people_in_a_frame_are_a_dead_band(in_k, in_before):
    """Contract (the sensors give no identity): the person's velocity pairs a sighting in the ask's frame with one in
    the earlier frame used, and with two evidence people in either frame a reader cannot tell a walker from two people
    standing, so the point is a dead band and `closest_approach` gives no moment. B is 3.2 m away, farther than A, and
    still makes no difference. Regression caught: the nearest person paired with whoever the truth says they are, which
    the sensors do not give (lane J's `knowable` failure, +0.119)."""
    s = ak.scene()
    pk.put_person(s, "A", (3.0, 3.63), [12, K])
    pk.put_person(s, "B", (5.0, 3.63), sorted(k for k, who in ((K, in_k), (12, in_before)) if "B" in who))
    w = pk.cut(s, K)
    assert candidate(w, K) == questions_approach.TWO_PEOPLE and closest_approach(w, K) is None


def test_one_person_in_each_frame_is_asked_with_their_velocity_whoever_else_is_about():
    """Contract: with exactly one evidence person in the ask's frame and in the earlier frame used, the pairing is
    unambiguous and the person's velocity is used, even if another person is in the world, in frames the pairing does
    not read. By hand: A walks 0.8 m south between 12 and 13 (0.5333 m/s) against the robot's 0.3 m/s north: the 2.0 m
    gap is gone at 2.4 s, 0.3 m to the side is left: 0.3. B is an evidence person only at keyframe 5. Regression caught:
    a dead band for any second person in the case, or the walker held still because someone else exists."""
    s = ak.scene()
    ak.walker(s, "A", (3.3, 3.93), (3.3, 3.13))
    pk.put_person(s, "B", (6.0, 3.0), [5])
    w = pk.cut(s, K)
    assert questions_approach.person_hull(w, "B", 5) is not None   # B is evidence there, in a frame the pairing skips
    c = candidate(w, K)
    assert c.gold == {"type": "number", "value": 0.3} and c.scope == "memory" and c.evidence_k == (12, K)


def test_a_person_the_frame_shows_and_does_not_place_is_a_dead_band_in_either_frame():
    """Contract: a person row of 256 px or more that is no evidence sighting (shown, not placed) in the ask's frame or
    in the earlier frame used leaves the pairing open, so the point is a dead band; 255 px, or the same row in a frame
    the pairing does not read, does not. By hand: A is the evidence person at keyframes 12 and 13; B, a glimpse at
    (6.0, 5.0), 4.9 m away (farther than A's 2.22 m, so the nearer-person screen is not what drops it). Regression
    caught: a second, half-seen person ignored in the frames the velocity reads, or a screen that drops any glimpse."""
    def asked(*where, px=300):
        s = ak.scene()
        pk.put_person(s, "A", (3.0, 3.63), [12, K])
        pk.put_person(s, "B", (6.0, 5.0), list(where), px=px)
        return cand(s)
    assert asked(K) == questions_approach.UNPLACED and asked(12) == questions_approach.UNPLACED
    assert asked(11).scope == "now" and asked(K, px=255).scope == "now"


def test_the_earlier_frame_used_is_chosen_without_identity():
    """Contract: the earlier frame is the latest within 4 s that holds an evidence sighting of anyone, as a reader
    without identity picks it, so when that frame holds only someone else the pairing would be wrong and the point is a
    dead band, while a frame with nobody in it is skipped. By hand: A is at keyframes 11 and 13; B alone at keyframe 12
    makes keyframe 12 the frame used (it holds B, not A); with B at keyframe 10 (before A's 11) or absent, A's 11 is the
    frame used. Regression caught: the gold pairing A's two sightings across B's frame, which the
    sensors cannot."""
    def asked(b_at):
        s = ak.scene()
        ak.walker(s, "A", (3.3, 4.43), (3.3, 3.13), seen=(11, K))
        if b_at is not None:
            pk.put_person(s, "B", (6.0, 5.0), [b_at])
        return cand(s)
    assert asked(12) == questions_approach.OTHER_EARLIER
    for b_at in (10, None):
        c = asked(b_at)
        assert c.gold == {"type": "number", "value": WALKING_GOLD} and c.evidence_k == (11, K)
