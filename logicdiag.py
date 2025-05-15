import torch
import itertools
import torch.nn.functional as F

# ----------------------------
# 1) RULE CLASSES
# ----------------------------

class CompositionRule:
    """
    If all children are 1 ⇒ parent must be 1.
    Fuzzy truth: 1 – max_i (1 – p(child_i)) – (1 – p(parent))
    Boolean consistency: (AND children = 1) ⇒ parent = 1
    """
    def __init__(self, parent, children):
        self.parent = parent
        self.children = children

    def evaluate(self, probs):
        # probs: [B, C]
        p_parent = probs[:, self.parent]
        p_children = probs[:, self.children]  # [B, n_children]
        # fuzzy AND ≈ min, fuzzy implication a⇒b ≈ 1 – a + b
        conj = p_children.min(dim=1)[0]
        return 1 - conj + p_parent  # ∈ [0,1]

    def is_satisfied(self, O):
        # O: binary [B, C]
        all_child_ones = O[:, self.children].all(dim=1)
        parent_ones   = O[:, self.parent].bool()
        return (~all_child_ones) | parent_ones  # [B]


class DecompositionRule:
    """
    If parent=1 ⇒ at least one child =1.
    Fuzzy truth: 1 – (1 – p(parent)) – min_i (1 – p(child_i))
    Boolean: (parent=1) ⇒ OR children =1
    """
    def __init__(self, parent, children):
        self.parent = parent
        self.children = children

    def evaluate(self, probs):
        p_parent = probs[:, self.parent]
        p_children = probs[:, self.children]
        disj = p_children.max(dim=1)[0]
        return 1 - (1 - p_parent) - (1 - disj)

    def is_satisfied(self, O):
        parent_ones = O[:, self.parent].bool()
        any_child  = O[:, self.children].any(dim=1)
        return (~parent_ones) | any_child  # [B]


class ExclusionRule:
    """
    At most one of {a, b, ...} can be 1.
    Fuzzy: 1 – sum_i sum_j>i (p_i * p_j)
    Boolean: sum(O[:, S]) ≤ 1
    """
    def __init__(self, group):
        self.group = group

    def evaluate(self, probs):
        p_grp = probs[:, self.group]  # [B, n]
        acc = torch.zeros(probs.shape[0], device=probs.device)
        for i, j in itertools.combinations(range(len(self.group)), 2):
            acc += p_grp[:, i] * p_grp[:, j]
        return 1 - acc

    def is_satisfied(self, O):
        s = O[:, self.group].sum(dim=1)
        return s <= 1  # [B]


# ----------------------------
# 2) CONFLICT RESOLUTION PIPELINE
# ----------------------------

def compute_fuzzy_conflict(o_logits, K):
    """
    Returns:
      probs: sigmoid(o_logits)
      O: binarized pseudo-labels {0,1}
      c: conflict degree ∈[0,1] for each class
    """
    probs = torch.sigmoid(o_logits)
    O = (probs > 0.5).float()

    truths = []
    for rule in K:
        t = rule.evaluate(probs)        # [B]
        truths.append(t)
    truths = torch.stack(truths, dim=1)  # [B, |K|]

    c = 1 - truths.mean(dim=1, keepdim=True)  # [B,1]
    c = c.expand_as(O)                         # [B,C]
    return probs, O, c


def all_consistent(O, K):
    """
    Boolean check: all rules satisfied for every batch item.
    """
    ok = []
    for rule in K:
        sat = rule.is_satisfied(O)  # [B] bool
        ok.append(sat)
    ok = torch.stack(ok, dim=1)     # [B, |K|]
    return ok.all(dim=1)            # [B] bool per item


def minimal_diagnoses(O, K, max_size=2):
    """
    Find all minimal subsets of indices whose flip restores consistency.
    Returns list of tuples (indices_to_flip).
    """
    B, C = O.shape
    diagnoses = []
    for size in range(1, max_size+1):
        for subset in itertools.combinations(range(C), size):
            O2 = O.clone()
            O2[:, list(subset)] = 1 - O2[:, list(subset)]
            good = all_consistent(O2, K)
            if good.all():
                if not any(set(sm).issubset(subset) for sm in diagnoses):
                    diagnoses.append(subset)
        if diagnoses:
            break
    return diagnoses


def sample_diagnosis(probs, O, c, diagnoses):
    """
    Sample one diagnosis ω for each batch element, per Eq.7–8.
    """
    B, C = probs.shape
    P_t = probs * (1-c) * O + (1-probs)*(1-c)*(1-O)  # [B,C]

    weights = []
    for ω in diagnoses:
        mask = torch.zeros_like(P_t, dtype=torch.bool)
        mask[:, list(ω)] = True
        w_flip = (1 - P_t)[mask].view(len(ω), B).prod(dim=0)
        w_keep = P_t[~mask].view(C-len(ω), B).prod(dim=0)
        weights.append((w_flip * w_keep).unsqueeze(0))  # [1,B]
    W = torch.cat(weights, dim=0)  # [num_diag, B]

    probs_w = F.softmax(W, dim=0)  # normalize over diagnoses
    idx    = torch.multinomial(probs_w, num_samples=1, dim=0).squeeze(0)  # [B]
    chosen = [diagnoses[i] for i in idx.tolist()]
    return chosen


def resolve_conflicts(o_logits, K, max_diag_size=2):
    """
    End-to-end: returns fixed pseudo-labels O_fixed [B,C]
    """
    probs, O, c = compute_fuzzy_conflict(o_logits, K)
    if all_consistent(O, K).all():
        return O  # no conflict

    diags = minimal_diagnoses(O, K, max_size=max_diag_size)
    chosen = sample_diagnosis(probs, O, c, diags)

    O_fixed = O.clone()
    for b, ω in enumerate(chosen):
        for j in ω:
            O_fixed[b, j] = 1 - O_fixed[b, j]
    return O_fixed

# ----------------------------
# 3) USAGE EXAMPLE
# ----------------------------
#
# from logic_diag import CompositionRule, DecompositionRule, ExclusionRule, resolve_conflicts
#
# # Define rules
# K = [
#     CompositionRule(parent=0, children=[1,2]),
#     DecompositionRule(parent=1, children=[3,4]),
#     ExclusionRule(group=[2,5,6])
# ]