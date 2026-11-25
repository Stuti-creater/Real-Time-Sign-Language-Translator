"""
synthetic_hands.py
==================
Builds artificial 21-point hands (in "palm length" units) so the classifier rules
can be tested without a camera. Each finger is a chain of 3 bones that can be
bent by chosen angles and splayed sideways.

This verifies the LOGIC of the rules. It does not replace testing with a real
webcam and real hands (see README / PROJECT_REPORT).
"""

import math

import numpy as np

# knuckle (MCP) positions, bone lengths and sideways splay direction per finger
MCP = {"index": (-0.30, 0.95), "middle": (0.0, 1.0), "ring": (0.28, 0.95), "pinky": (0.52, 0.85)}
BONES = {
    "index": (0.38, 0.22, 0.17),
    "middle": (0.42, 0.25, 0.18),
    "ring": (0.39, 0.23, 0.18),
    "pinky": (0.30, 0.18, 0.15),
}
FINGER_IDS = {"index": 5, "middle": 9, "ring": 13, "pinky": 17}

STRAIGHT = (0, 0, 0)
CURLED = (90, 100, 70)
BENT_C = (45, 45, 35)
BENT_F = (50, 60, 30)


def _finger(name, bends, splay_deg):
    """Return the 4 points (MCP, PIP, DIP, TIP) of one finger."""
    x, y = MCP[name]
    pos = np.array([x, y, 0.0])
    points = [pos.copy()]
    total = 0.0
    s = math.radians(splay_deg)
    for bone, bend in zip(BONES[name], bends):
        total += math.radians(bend)
        # direction: up (y), rotated sideways by splay, then bent towards the palm (-z)
        d = np.array([math.sin(s) * math.cos(total), math.cos(s) * math.cos(total), -math.sin(total)])
        pos = pos + bone * d
        points.append(pos.copy())
    return points


def make_hand(fingers, thumb, splay=None, thumb_target=None):
    """
    fingers: dict name -> bend tuple, e.g. {"index": STRAIGHT, ...}
    thumb:   'out', 'side_up', 'folded', 'tuck_mid', 'touch_index', 'touch_mid', 'c_shape'
    """
    splay = splay or {}
    pts = np.zeros((21, 3))
    pts[0] = (0, 0, 0)
    tips = {}
    for name, first in FINGER_IDS.items():
        chain = _finger(name, fingers[name], splay.get(name, 0))
        for k, pt in enumerate(chain):
            pts[first + k] = pt
        tips[name] = chain[-1]

    targets = {
        "out": (-1.10, 0.62, 0.0),
        "side_up": (-0.45, 1.15, 0.0),
        "folded": (0.0, 0.45, 0.0),
        "tuck_mid": tuple(np.array(tips["middle"]) + (-0.05, 0.0, 0.0)),
        "touch_index": tuple(tips["index"]),
        "touch_mid": tuple(tips["middle"]),
        "c_shape": (-0.55, 0.55, -0.6),
    }
    tip = np.array(thumb_target if thumb_target else targets[thumb])
    cmc = np.array([-0.22, 0.22, 0.0])
    mcp = np.array([-0.42, 0.45, 0.0])
    pts[1], pts[2] = cmc, mcp
    pts[3] = mcp + (tip - mcp) * 0.5
    pts[4] = tip
    return pts


def rotate(pts, degrees):
    """Rotate the whole hand in the image plane (to prove rotation independence)."""
    a = math.radians(degrees)
    r = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    return pts @ r.T


UP, DOWN, CURL = STRAIGHT, CURLED, CURLED


def hand_for(label):
    """Return a synthetic hand that should be classified as `label`."""
    s = {"index": STRAIGHT, "middle": STRAIGHT, "ring": STRAIGHT, "pinky": STRAIGHT}
    c = {"index": CURLED, "middle": CURLED, "ring": CURLED, "pinky": CURLED}
    if label == "A":
        return make_hand(c, "side_up")
    if label == "B":
        return make_hand(s, "folded")
    if label == "C":
        return make_hand({k: BENT_C for k in c}, "c_shape")
    if label == "D":
        return make_hand({**c, "index": STRAIGHT}, "touch_mid")
    if label == "E":
        return make_hand(c, "tuck_mid")
    if label == "F":
        return make_hand({**s, "index": BENT_F}, "touch_index")
    if label == "I":
        return make_hand({**c, "pinky": STRAIGHT}, "folded")
    if label == "L":
        return make_hand({**c, "index": STRAIGHT}, "out")
    if label == "V":
        return make_hand({**c, "index": STRAIGHT, "middle": STRAIGHT}, "folded",
                         splay={"index": -15, "middle": 15})
    if label == "W":
        return make_hand({**s, "pinky": CURLED}, "folded",
                         splay={"index": -15, "middle": 0, "ring": 15})
    if label == "Y":
        return make_hand({**c, "pinky": STRAIGHT}, "out")
    if label == "HELLO":
        return make_hand(s, "out", splay={"index": -12, "middle": -3, "ring": 5, "pinky": 15})
    if label == "I LOVE YOU":
        return make_hand({**c, "index": STRAIGHT, "pinky": STRAIGHT}, "out")
    raise ValueError(label)
