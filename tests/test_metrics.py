from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, auc, f1_score, precision_recall_curve


def test_metrics_basic():
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    assert 0.0 <= acc <= 1.0
    assert 0.0 <= f1 <= 1.0

    # pr curve
    scores = np.array([0.2, 0.8, 0.4, 0.1, 0.7])
    prec, rec, _ = precision_recall_curve(y_true, scores)
    val = auc(rec, prec)
    assert 0.0 <= val <= 1.0
