"""Count-based detection of SynthID-Text from (k-token context, next token) events.

All texts are flattened into one token array. A "level" is a context definition (a tuple
of offsets back from the predicted token, e.g. (1, 2, 3, 4) = the 4 previous tokens). For
each level every position gets an event id (context + token) and a context id, and
per-fold / per-class counts are stored, so that "counts from every fold except f" is a
subtraction. That keeps nested cross-validation cheap: for outer fold f, the inner loop
leaves out one more fold g and counts from the remaining three.

Scores per event (see event_stat):
    condz  z-score of the event's watermarked count a against its expectation given the
           watermarked share of its context (the detector of the paper)
    z      the same z-score without the context: a against b ("no conditioning" ablation)
A text's score combines its events: "len" = sum / number of eligible positions (unseen
events count as zero), "sqrt" = sum / sqrt(number of seen events).
"""

import numpy as np
from sklearn.metrics import roc_auc_score

FOLDS = 5
SEED = 1234
P1 = np.uint64(0x9E3779B97F4A7C15)
P2 = np.uint64(0xC2B2AE3D27D4EB4F)
P3 = np.uint64(0x165667B19E3779F9)


class Data:
    """n watermarked texts followed by their n unwatermarked twins, as one token array.

    ids:  list of 2n token-id lists (watermarked first, then twins in the same order)
    fold: fold (0..FOLDS-1) of each of the 2n texts; -1 = never counted, never scored
    """

    def __init__(self, ids, fold):
        n = len(ids) // 2
        self.n, self.T = n, 2 * n
        self.y = np.r_[np.ones(n, int), np.zeros(n, int)]
        self.lens = np.array([len(x) for x in ids])
        self.tok = np.concatenate([np.asarray(x, dtype=np.uint64) for x in ids])
        self.tix = np.repeat(np.arange(self.T), self.lens)
        self.pos = np.concatenate([np.arange(l) for l in self.lens])
        self.fold = np.asarray(fold)
        self.pfold = self.fold[self.tix]
        self.py = self.y[self.tix].astype(int)
        self.levels = {}

    def level(self, offsets, dedup="ctx"):
        """Event / context ids and counts for one context definition.
        dedup='ctx': only the first occurrence of each context in a text is used
        (SynthID adds no new evidence for a repeated context)."""
        key = (tuple(offsets), dedup)
        if key in self.levels:
            return self.levels[key]
        N = len(self.tok)
        h = np.zeros(N, dtype=np.uint64)
        with np.errstate(over="ignore"):
            for o in offsets:
                h = h * P1 + np.roll(self.tok, o) + np.uint64(1)
            ev = h * P2 + self.tok * P3 + np.uint64(7)
            tsalt = self.tix.astype(np.uint64) * np.uint64(0xD6E8FEB86659FD93)
        valid = self.pos >= max(offsets)
        vi = np.flatnonzero(valid)
        if dedup == "none":
            sel = valid.copy()
        else:
            with np.errstate(over="ignore"):
                k = (h if dedup == "ctx" else ev)[vi] ^ tsalt[vi]
            _, first = np.unique(k, return_index=True)
            sel = np.zeros(N, dtype=bool)
            sel[vi[first]] = True
        L = {"offsets": tuple(offsets), "sel": sel}
        for name, arr in (("E", ev), ("C", h)):
            _, inv = np.unique(arr[vi], return_inverse=True)
            full = np.full(N, -1, dtype=np.int64)
            full[vi] = inv
            U = int(inv.max()) + 1
            cnt = np.zeros((FOLDS, 2, U), dtype=np.int32)
            for f in range(FOLDS):
                for c in (0, 1):
                    m = sel & (self.pfold == f) & (self.py == c)
                    cnt[f, c] = np.bincount(full[m], minlength=U)
            L[name], L["cnt" + name] = full, cnt
            L["tot" + name] = cnt.sum(0)
        self.levels[key] = L
        return L

    def counts(self, L, excl, which="E"):
        """(watermarked counts, unwatermarked counts) per unique event / context, from all
        folds except those in excl."""
        c = L["tot" + which].copy()
        for f in excl:
            c -= L["cnt" + which][f]
        return c[1], c[0]


def event_stat(a, b, A, B, stat, alpha, c=1.0):
    a, b = a.astype(float), b.astype(float)
    if stat == "z":  # no conditioning: a against b
        return (a - b) / np.sqrt(a + b + alpha)
    if stat == "condz":  # a against its expectation given the context's watermarked share
        p = (A + c) / (A + B + 2.0 * c) if c > 0 else A / np.maximum(A + B, 1)
        return (a - (a + b) * p) / np.sqrt((a + b) * p * (1 - p) + alpha)
    raise ValueError(stat)


def level_scores(D, L, excl, texts_mask, stat="condz", alpha=1.0, mc=2, agg="len", c=1.0):
    """Scores of the texts in texts_mask, with counts from all folds not in excl.
    Returns (score per text [T], number of seen events per text [T])."""
    a_u, b_u = D.counts(L, excl, "E")
    idx = np.flatnonzero(L["sel"] & texts_mask[D.tix])
    a, b = a_u[L["E"][idx]], b_u[L["E"][idx]]
    A = B = None
    if stat == "condz":
        A_u, B_u = D.counts(L, excl, "C")
        A, B = A_u[L["C"][idx]], B_u[L["C"][idx]]
    keep = (a + b) >= mc
    s = event_stat(a[keep], b[keep], None if A is None else A[keep], None if B is None else B[keep],
                   stat, alpha, c)
    t = D.tix[idx]
    tot = np.bincount(t[keep], weights=s, minlength=D.T)
    m = np.bincount(t[keep], minlength=D.T).astype(float)
    nev = np.bincount(t, minlength=D.T).astype(float)
    if agg == "sqrt":
        out = np.divide(tot, np.sqrt(m), out=np.zeros(D.T), where=m > 0)
    elif agg == "len":
        out = np.divide(tot, nev, out=np.zeros(D.T), where=nev > 0)
    else:
        raise ValueError(agg)
    return out, m


def oof(D, cfg, excl_extra=(), folds=None):
    """Out-of-fold scores for a config dict: each fold scored with counts from the others."""
    L = D.level(cfg["offsets"], cfg.get("dedup", "ctx"))
    out = np.zeros(D.T)
    for f in (range(FOLDS) if folds is None else folds):
        mask = D.fold == f
        s, _ = level_scores(D, L, (f,) + tuple(excl_extra), mask, cfg.get("stat", "condz"),
                            cfg.get("alpha", 1.0), cfg.get("mc", 2), cfg.get("agg", "len"), cfg.get("c", 1.0))
        out[mask] = s[mask]
    return out


def cfg_scorer(D, cfg):
    return lambda excl_extra, folds: oof(D, cfg, excl_extra, folds)


def ctx(k):
    return tuple(range(1, k + 1))


def wins(scores, n):
    """1 where the watermarked text outscores its twin, 0.5 on ties."""
    return (scores[:n] > scores[n:]).astype(float) + 0.5 * (scores[:n] == scores[n:])


def nested_select(D, scorers, criterion="auc"):
    """scorers: dict name -> fn(excl_extra, folds) returning scores on those folds.
    For each outer fold f, an inner CV over the other folds g (counts from all but f and g)
    gives inner out-of-fold scores on the training texts; the config with the best inner
    AUROC is applied to fold f with counts from all training folds. Test scores are
    standardised with the mean / std of that config's inner scores, so that folds that
    picked different configs are on one scale."""
    out = np.zeros(D.T)
    chosen = []
    for f in range(FOLDS):
        trm = D.fold != f
        best, best_v, best_in = None, -1, None
        for name, fn in scorers.items():
            inner = np.zeros(D.T)
            for g in range(FOLDS):
                if g == f:
                    continue
                s = fn((f,), (g,))
                inner[D.fold == g] = s[D.fold == g]
            if criterion == "auc":
                v = roc_auc_score(D.y[trm], inner[trm])
            else:  # paired accuracy on training pairs
                v = wins(inner, D.n)[trm[:D.n]].mean()
            if v > best_v:
                best, best_v, best_in = name, v, inner
        s = scorers[best]((), (f,))
        mu, sd = best_in[trm].mean(), best_in[trm].std() or 1.0
        out[D.fold == f] = (s[D.fold == f] - mu) / sd
        chosen.append((best, round(float(best_v), 4)))
    return out, chosen
