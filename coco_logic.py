# coco_logic_rules.py
"""
LogicDiag knowledge base for COCO-2017
-------------------------------------
* Root (“Object”)        – virtual root node
* 12 coarse super-classes – COCO official *supercategory* field
* 80 leaf classes        – standard COCO ‘thing’ categories
The module builds three kinds of logic rules exactly as in §3.3
of the paper:
  • Composition     (child  ⇒ parent)
  • Decomposition   (parent ⇒ OR children)
  • Exclusion       (siblings are mutually exclusive)
Import and add the resulting list `K` to your training script:
    from coco_logic_rules import K
    ...
    probs, O_fixed = resolve_conflicts(o_logits, K)
"""
from logicdiag import CompositionRule, DecompositionRule, ExclusionRule

# ------------------------------------------------------------------
# 1)  Class lists (index order must match your CLASSES array)
# ------------------------------------------------------------------
ROOT               = "object"     # the single root node used in the paper
SUPER_CATEGORIES   = [
    "person", "vehicle", "outdoor", "animal", "accessory", "sports",
    "kitchen", "food", "furniture", "electronic", "appliance", "indoor"
]

# fine classes grouped by their COCO *supercategory*
FINE_CLASSES = {
    "person":    ["person"],

    "vehicle":   ["bicycle", "car", "motorcycle", "airplane", "bus",
                  "train", "truck", "boat"],

    "outdoor":   ["traffic light", "fire hydrant", "stop sign",
                  "parking meter", "bench"],

    "animal":    ["bird", "cat", "dog", "horse", "sheep", "cow",
                  "elephant", "bear", "zebra", "giraffe"],

    "accessory": ["backpack", "umbrella", "handbag", "tie", "suitcase"],

    "sports":    ["frisbee", "skis", "snowboard", "sports ball", "kite",
                  "baseball bat", "baseball glove", "skateboard",
                  "surfboard", "tennis racket"],

    "kitchen":   ["bottle", "wine glass", "cup", "fork", "knife",
                  "spoon", "bowl"],

    "food":      ["banana", "apple", "sandwich", "orange", "broccoli",
                  "carrot", "hot dog", "pizza", "donut", "cake"],

    "furniture": ["chair", "couch", "bed", "dining table", "toilet",
                  "potted plant"],

    "electronic":["tv", "laptop", "mouse", "remote", "keyboard",
                  "cell phone"],

    "appliance": ["microwave", "oven", "toaster", "sink", "refrigerator"],

    "indoor":    ["book", "clock", "vase", "scissors", "teddy bear",
                  "hair drier", "toothbrush"]
}

# ------------------------------------------------------------------
# 2)  Build a global (flat) list so we can look-up integer indices
#     NOTE: prepend ROOT and the 12 super-classes before leaves
# ------------------------------------------------------------------
COCO_LABELS = (
    [ROOT] +
    SUPER_CATEGORIES +
    [cls for supercat in SUPER_CATEGORIES for cls in FINE_CLASSES[supercat]]
)
NAME2IDX = {name: i for i, name in enumerate(COCO_LABELS)}

# ------------------------------------------------------------------
# 3)  Generate logic rules
# ------------------------------------------------------------------
K = []

# 3-a) Composition   fine  ⇒ coarse
for supercat, children in FINE_CLASSES.items():
    for child in children:
        K.append(
            CompositionRule(
                parent=NAME2IDX[supercat],
                children=[NAME2IDX[child]]
            )
        )

# 3-b) Composition   coarse ⇒ ROOT
for supercat in SUPER_CATEGORIES:
    K.append(
        CompositionRule(
            parent=NAME2IDX[ROOT],
            children=[NAME2IDX[supercat]]
        )
    )

# 3-c) Decomposition   coarse ⇒ OR(children)
for supercat, children in FINE_CLASSES.items():
    K.append(
        DecompositionRule(
            parent=NAME2IDX[supercat],
            children=[NAME2IDX[c] for c in children]
        )
    )

# 3-d) Decomposition   ROOT ⇒ OR(super-classes)
K.append(
    DecompositionRule(
        parent=NAME2IDX[ROOT],
        children=[NAME2IDX[s] for s in SUPER_CATEGORIES]
    )
)

# 3-e) Exclusion   siblings within each coarse group
for siblings in FINE_CLASSES.values():
    K.append(
        ExclusionRule(
            group=[NAME2IDX[s] for s in siblings]
        )
    )

# 3-f) Exclusion   among the 12 super-classes themselves
K.append(
    ExclusionRule(
        group=[NAME2IDX[s] for s in SUPER_CATEGORIES]
    )
)

# ------------------------------------------------------------------
# 4)  Public export
# ------------------------------------------------------------------
__all__ = ["COCO_LABELS", "NAME2IDX", "K"]
