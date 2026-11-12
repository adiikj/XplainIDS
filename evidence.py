"""
Evidence for each alert type. Everything is in SHAP units of P(attack).

    KNOWN   p >= t1              top-k features with the largest POSITIVE phi_i (what pushed XGBoost towards "attack")
    NOVEL   p < t1 and s >= t2   top-k features with the largest |phi_i - phi_hat_i|, where phi_hat is the
                                 autoencoder's reconstruction: "what a normal-looking explanation would be"

Floor (decision 11 Nov 2026): a feature is listed only if its |deviation| (NOVEL) or phi (KNOWN) is at least
FLOOR x the largest one, so near-identical pairs like -0.094 vs -0.076 are not shown; at least 1, at most k.
The KNOWN_VARIANT flag was dropped: an AE trained on benign SHAP vectors finds almost every attack unusual, so it
fired on nearly all KNOWN alerts (10,584 test flows on NSL-KDD official, seed 0) and carried no information.

phi is the original SHAP vector (never the clipped detector input); phi_hat is the reconstruction mapped back to SHAP
units with InputTransform.inverse_shap. Categorical values are decoded back to their names (e.g. flag = S0).
"""

import numpy as np

K = 5
FLOOR = 0.2
ALERT_NAMES = {1: 'KNOWN', 2: 'NOVEL'}


def decode_value(name, value, categories):
    """Original value of a feature for display: category name for encoded columns, plain number otherwise."""
    if categories and name in categories:
        code = int(value)
        return categories[name][code] if 0 <= code < len(categories[name]) else 'unseen'
    v = float(value)
    return int(v) if v.is_integer() else round(v, 6)


def known_evidence(phi, x_raw, names, categories, k=K):
    """Top-k features by positive SHAP contribution (features pushing towards benign are left out)."""
    order = [j for j in np.argsort(-phi) if phi[j] > 0][:k]
    if order:
        order = [j for j in order if phi[j] >= FLOOR * phi[order[0]]]
    return [dict(name=names[j], raw_value=decode_value(names[j], x_raw[j], categories), phi=round(float(phi[j]), 4),
                 direction='towards attack') for j in order]


def novel_evidence(phi, phi_hat, x_raw, names, categories, k=K):
    """Top-k features where the flow's explanation deviates most from the autoencoder's normal-looking reconstruction."""
    dev = phi - phi_hat
    order = np.argsort(-np.abs(dev))[:k]
    order = [j for j in order if abs(dev[j]) >= FLOOR * abs(dev[order[0]])]
    return [dict(name=names[j], raw_value=decode_value(names[j], x_raw[j], categories), phi=round(float(phi[j]), 4),
                 phi_hat=round(float(phi_hat[j]), 4), deviation=round(float(dev[j]), 4),
                 direction='contributes more than in normal-looking flows' if dev[j] > 0
                 else 'contributes less than in normal-looking flows') for j in order]


def evidence_line(e):
    """One human-readable line, e.g. 'serror_rate = 1.0 contributed +0.31; normal-looking flows show about +0.02'."""
    line = f"{e['name']} = {e['raw_value']} contributed {e['phi']:+.3f}"
    if 'phi_hat' in e:
        line += f"; normal-looking flows show about {e['phi_hat']:+.3f}"
    return line


def build_record(alert_id, alert_type, p, s, t1, t2, phi, phi_hat, x_raw, names, categories):
    """
    AlertRecord as a plain dict (the Pydantic schema in llm/schema.py will mirror it). No raw traffic beyond the
    evidence features' own values is included.
    """
    names = list(names)
    if alert_type == 1:
        kind, ev = 'KNOWN', known_evidence(phi, x_raw, names, categories)
    elif alert_type == 2:
        kind = 'NOVEL'
        if phi_hat is None:
            raise ValueError("NOVEL evidence needs a detector with reconstruct() on SHAP input (ae_small, pca)")
        ev = novel_evidence(phi, phi_hat, x_raw, names, categories)
    else:
        raise ValueError("benign flows get no alert record")
    return dict(alert_id=str(alert_id), alert_type=kind, p=round(float(p), 6), t1=round(float(t1), 6),
                s=round(float(s), 6), t2=round(float(t2), 6), evidence=ev,
                consistency_flag=None)   # SHAP-LIME agreement flag: added with the LLM layer
