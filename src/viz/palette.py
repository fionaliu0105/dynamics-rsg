"""Shared figure colors, so the figures in one deck use a consistent scheme.

All figure code imports from here instead of hard-coding hex values or picking
``viridis`` / ``tab10`` per function. There are two color families:

* ``RDM_CMAP``: a blue-to-red ramp for every heatmap (condition RDMs, the RDM
  gallery, and the arm-by-arm within/between matrices). Low distance is blue,
  high distance is red.
* ``ARM_COLORS``: one fixed color per learning-rule arm, for every point, bar, or
  line plot keyed by arm. The two PC arms get adjacent warm tones (same rule, two
  inference budgets), and the untrained control is neutral gray.

Both families come from the same five anchor colors, so a heatmap and a bar chart
on neighboring slides match.
"""

from __future__ import annotations

from matplotlib.colors import LinearSegmentedColormap

# Five anchor hues (the provided scheme): blue -> light blue -> yellow -> orange -> red.
_ANCHORS = ["#5271AE", "#70ACDE", "#F5CC7D", "#FFA660", "#D85B59"]

# Continuous ramp for condition-RDM heatmaps and galleries. Sequential reading:
# low value = blue, high value = red.
RDM_CMAP = LinearSegmentedColormap.from_list("rsg_blue_red", _ANCHORS, N=256)

# Monochrome blue ramp for the arm-by-arm within/between distance matrices. Light blue
# is a small distance (near), dark blue is a large distance (far).
_BLUES = ["#C6EAF3", "#8FD3ED", "#5AA9CD", "#357AA7", "#0E4775"]
MATRIX_CMAP = LinearSegmentedColormap.from_list("rsg_blues", _BLUES, N=256)

# Categorical arm colors. Keys are the canonical arm/rule labels used across the
# pipeline; PC's two step-counts sit next to each other in the warm range on purpose.
ARM_COLORS = {
    "untrained": "#9AA3AF",  # neutral gray, no hue for the control
    "bptt": "#5271AE",  # blue (nonlocal in time and space)
    "pc_steps20": "#F5CC7D",  # yellow (PC, 20-step inference)
    "pc_steps100": "#FFA660",  # orange (PC, 100-step inference)
    "rflo": "#D85B59",  # red (local in time and space)
}

# Prior colors reuse the ramp endpoints, so short/long stays consistent with the
# heatmap's low/high reading.
PRIOR_COLORS = {"short": "#5271AE", "long": "#D85B59"}

# Signed-difference coloring for the paired contrast: warm = worse (above 0),
# cool = better/closer to DMFC (below 0). Same endpoints as the ramp.
DELTA_POS = "#D85B59"  # arm is farther from DMFC than the reference
DELTA_NEG = "#5271AE"  # arm is closer to DMFC than the reference

# Neutral fallback for any key not in ARM_COLORS (keeps a stray label from crashing).
_FALLBACK = "#9AA3AF"

# A few aliases so model-to-model comparison labels ("pc") and casual keys still resolve.
_ALIASES = {
    "pc": "pc_steps100",
    "pc20": "pc_steps20",
    "pc100": "pc_steps100",
}


def arm_color(arm: str) -> str:
    """Color for an arm/rule label, tolerant of aliases and unknown keys."""
    key = str(arm)
    if key in ARM_COLORS:
        return ARM_COLORS[key]
    if key in _ALIASES:
        return ARM_COLORS[_ALIASES[key]]
    return _FALLBACK
