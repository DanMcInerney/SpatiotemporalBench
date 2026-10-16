"""What the `kind_range` tests share (lane A6): `ego_now` questions, delivered steps that show one thing at a known
offset from their camera, and prior records written by hand. `kind_range` is driven through the real `table_ego`
answerer (`table_answers.ANSWERERS["ego_now"]`), which measures from the camera position of the ask's step.

`seen_at` builds the step: the vacuum's base at (1, 0) heading +y (90 degrees), so that ahead is +y and left is -x,
and a detection of the thing `ahead` metres ahead of that step's own camera and `left` to its left, by hand: the point
(camera x - left, camera y + ahead). `prior_of` writes one record a gold: the golds are offsets of known floor
distance, spread over the scripts `s0, s1, ...` in turn.
"""

from __future__ import annotations

from baseline_kit import offset, question, record, step

EGO = "ego_now"


def ego_question(kind: str = "bin", qid: str = "q1", color: str = "red") -> dict:
    """An `ego_now` question about the `kind` of `color` as the structured view gives it."""
    return question(EGO, {"kind": kind, "attributes": {"color": color}}, qid, "offset")


def seen_at(k: int, ahead: float, left: float, kind: str = "bin") -> dict:
    """Step `k` of the vacuum (base (1, 0), heading +y) with a red `kind` detected `ahead` m ahead of its camera and
    `left` m to its left, along the floor (the detection's height is 0.1 m); its floor hull is the one point, so that is
    the thing's nearest point, which the answerer measures from the camera of the step."""
    cx, cy = step(k, 1.0, 0.0, 90.0)["camera"]["pos"][:2]
    d = {"d": "a", "kind": kind, "attributes": {"color": "red"}, "bbox_px": [0, 0, 40, 40],
         "center_cam": [0.0, 0.0, 1.0], "center_map": [cx - left, cy + ahead, 0.1], "extent_m": [0.2, 0.2, 0.2],
         "floor_hull": [[round(cx - left, 3), round(cy + ahead, 3)]]}
    return step(k, 1.0, 0.0, 90.0, detections=[d])


def prior_of(kind: str, golds: list[tuple[float, float]], *, scripts: int = 3, start: int = 0) -> list[dict]:
    """One `ego_now` record of `kind` for each gold `(ahead, left)`, from the scripts `s<start>` ... in turn over
    `scripts` of them."""
    return [record(ego_question(kind, qid=f"p{start}{i}"), offset(a, left), script=f"s{start + i % scripts}",
                   when="during") for i, (a, left) in enumerate(golds)]
