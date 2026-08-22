import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import brier_score_loss, log_loss

def evaluate_va_prob_magnitude_filtering_negative_only(
    intervals,
    y_true,
    magnitude_threshold=None,
    retain_fraction=0.9,
    threshold=0.5,
    pred_method="collapse",
):
    """
    Evaluate Venn-Abers intervals using probability-magnitude filtering, but only
    among predicted-negative examples.

    This is useful when predicted negatives are the high-risk group because an
    incorrect negative prediction corresponds to a false negative.

    Filtering rule:
        1. Convert VA interval to a single probability p_hat.
        2. Predict class using threshold.
        3. Identify predicted negatives only: y_pred == 0.
        4. Rank predicted negatives by probability magnitude:
               prob_magnitude = max(p_hat, 1 - p_hat)
           For predicted negatives, this is equivalent to 1 - p_hat.
        5. Reject the least confident predicted negatives.
        6. Retain all predicted positives.

    Parameters
    ----------
    intervals : array-like of shape (n_samples, 2)
        Venn-Abers intervals stored as [[p0, p1], [p0, p1], ...].

    y_true : array-like of shape (n_samples,) or (n_samples, 1)
        Ground-truth binary labels in {0,1}.

    magnitude_threshold : float or None
        If provided, predicted-negative examples with probability magnitude below
        this threshold are rejected. Predicted positives are always retained.

    retain_fraction : float or None
        Desired approximate overall retain fraction. If magnitude_threshold is
        None, the function rejects approximately (1 - retain_fraction) of the
        full dataset, but only from the predicted-negative subset. If there are
        fewer predicted negatives than required, all predicted negatives are
        rejected and all predicted positives are retained.

    threshold : float, default=0.5
        Classification threshold for positive class.

    pred_method : {"collapse", "midpoint"}, default="collapse"
        How to convert the VA interval into a single probability:
            - "collapse": p = p1 / (1 - p0 + p1)
            - "midpoint": p = 0.5 * (p0 + p1)

    Returns
    -------
    results : dict
        Dictionary containing retained/rejected masks, probabilities, labels,
        metrics, and filtered composition.
    """

    import numpy as np
    from sklearn.metrics import brier_score_loss, log_loss

    # ----------------------------
    # 1) Input cleaning
    # ----------------------------
    intervals = np.asarray(intervals, dtype=float)
    y_true = np.asarray(y_true, dtype=int).reshape(-1)

    if intervals.ndim != 2 or intervals.shape[1] != 2:
        raise ValueError("intervals must have shape (n_samples, 2), e.g. [[p0, p1], ...].")

    if len(y_true) != len(intervals):
        raise ValueError("intervals and y_true must have the same number of samples.")

    if not np.all(np.isin(y_true, [0, 1])):
        raise ValueError("y_true must contain only 0/1 labels.")

    if magnitude_threshold is None and retain_fraction is None:
        raise ValueError("Provide either magnitude_threshold or retain_fraction.")

    if magnitude_threshold is not None and retain_fraction is not None:
        raise ValueError("Provide only one of magnitude_threshold or retain_fraction, not both.")

    if retain_fraction is not None:
        if not (0 < retain_fraction <= 1):
            raise ValueError("retain_fraction must be in (0, 1].")

    n_samples = len(y_true)

    # Enforce p0 <= p1
    p0 = np.minimum(intervals[:, 0], intervals[:, 1])
    p1 = np.maximum(intervals[:, 0], intervals[:, 1])

    # ----------------------------
    # 2) Point prediction and confidence score
    # ----------------------------
    width = p1 - p0

    if pred_method == "collapse":
        denom = np.maximum(1.0 - p0 + p1, 1e-15)
        p_hat = p1 / denom
    elif pred_method == "midpoint":
        p_hat = 0.5 * (p0 + p1)
    else:
        raise ValueError("pred_method must be 'collapse' or 'midpoint'.")

    p_hat = np.clip(p_hat, 1e-15, 1 - 1e-15)

    y_pred = (p_hat >= threshold).astype(int)

    # Probability magnitude / confidence in predicted class
    prob_magnitude = np.maximum(p_hat, 1.0 - p_hat)

    # Only predicted negatives are eligible for rejection
    negative_pred_mask = y_pred == 0
    positive_pred_mask = y_pred == 1

    negative_indices = np.where(negative_pred_mask)[0]

    # ----------------------------
    # 3) Choose filtering rule among predicted negatives only
    # ----------------------------
    reject_mask = np.zeros(n_samples, dtype=bool)

    if magnitude_threshold is not None:
        # Reject predicted negatives below threshold.
        # Predicted positives are retained regardless.
        reject_mask = negative_pred_mask & (prob_magnitude < magnitude_threshold)

    else:
        # Approximate desired overall retain fraction by rejecting from negatives only.
        n_reject_requested = int(np.ceil((1.0 - retain_fraction) * n_samples))

        # Cannot reject more examples than there are predicted negatives.
        n_reject = min(n_reject_requested, len(negative_indices))

        if n_reject > 0:
            # Among predicted negatives, least confident first.
            # For negatives, low prob_magnitude means p_hat closer to threshold.
            neg_magnitudes = prob_magnitude[negative_indices]
            sorted_negative_indices = negative_indices[np.argsort(neg_magnitudes)]

            reject_idx = sorted_negative_indices[:n_reject]
            reject_mask[reject_idx] = True

        if np.sum(reject_mask) > 0:
            magnitude_threshold = float(np.max(prob_magnitude[reject_mask]))
        else:
            magnitude_threshold = np.nan

    retain_mask = ~reject_mask

    # ----------------------------
    # 4) Metric helper
    # ----------------------------
    def _subset_metrics(mask, label):
        n = int(np.sum(mask))

        if n == 0:
            return {
                "subset": label,
                "count": 0,
                "fraction": 0.0,
                "accuracy": np.nan,
                "risk": np.nan,
                "sensitivity": np.nan,
                "specificity": np.nan,
                "precision": np.nan,
                "prevalence": np.nan,
                "brier": np.nan,
                "log_loss": np.nan,
                "mean_width": np.nan,
                "median_width": np.nan,
                "mean_p_hat": np.nan,
                "median_p_hat": np.nan,
                "mean_prob_magnitude": np.nan,
                "median_prob_magnitude": np.nan,
                "tp": 0,
                "fp": 0,
                "tn": 0,
                "fn": 0,
            }

        yt = y_true[mask]
        yp = y_pred[mask]
        ph = p_hat[mask]
        wd = width[mask]
        mag = prob_magnitude[mask]

        tp = int(np.sum((yt == 1) & (yp == 1)))
        fp = int(np.sum((yt == 0) & (yp == 1)))
        tn = int(np.sum((yt == 0) & (yp == 0)))
        fn = int(np.sum((yt == 1) & (yp == 0)))

        acc = float(np.mean(yp == yt))
        risk = float(1.0 - acc)

        P = tp + fn
        N = tn + fp

        sensitivity = float(tp / P) if P > 0 else np.nan
        specificity = float(tn / N) if N > 0 else np.nan
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else np.nan
        prevalence = float(np.mean(yt))

        try:
            brier = float(brier_score_loss(yt, ph))
        except ValueError:
            brier = np.nan

        try:
            ll = float(log_loss(yt, ph, labels=[0, 1]))
        except ValueError:
            ll = np.nan

        return {
            "subset": label,
            "count": n,
            "fraction": float(n / n_samples),
            "accuracy": acc,
            "risk": risk,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "precision": precision,
            "prevalence": prevalence,
            "brier": brier,
            "log_loss": ll,
            "mean_width": float(np.mean(wd)),
            "median_width": float(np.median(wd)),
            "mean_p_hat": float(np.mean(ph)),
            "median_p_hat": float(np.median(ph)),
            "mean_prob_magnitude": float(np.mean(mag)),
            "median_prob_magnitude": float(np.median(mag)),
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
        }

    overall_metrics = _subset_metrics(np.ones(n_samples, dtype=bool), "overall")
    retained_metrics = _subset_metrics(retain_mask, "retained")
    rejected_metrics = _subset_metrics(reject_mask, "rejected")

    # ----------------------------
    # 5) Filtered examples composition
    # ----------------------------
    filtered_count = int(np.sum(reject_mask))

    filtered_tp = int(np.sum(reject_mask & (y_true == 1) & (y_pred == 1)))
    filtered_fp = int(np.sum(reject_mask & (y_true == 0) & (y_pred == 1)))
    filtered_tn = int(np.sum(reject_mask & (y_true == 0) & (y_pred == 0)))
    filtered_fn = int(np.sum(reject_mask & (y_true == 1) & (y_pred == 0)))

    filtered_composition = {
        "filtered_count": filtered_count,
        "filtered_fraction": float(filtered_count / n_samples),
        "filtered_tp": filtered_tp,
        "filtered_fp": filtered_fp,
        "filtered_tn": filtered_tn,
        "filtered_fn": filtered_fn,
        "filtered_predicted_positive": int(np.sum(reject_mask & (y_pred == 1))),
        "filtered_predicted_negative": int(np.sum(reject_mask & (y_pred == 0))),
        "fraction_filtered_tp": float(filtered_tp / filtered_count) if filtered_count > 0 else np.nan,
        "fraction_filtered_fp": float(filtered_fp / filtered_count) if filtered_count > 0 else np.nan,
        "fraction_filtered_tn": float(filtered_tn / filtered_count) if filtered_count > 0 else np.nan,
        "fraction_filtered_fn": float(filtered_fn / filtered_count) if filtered_count > 0 else np.nan,
    }

    # ----------------------------
    # 6) Return
    # ----------------------------
    results = {
        "filtering_rule": {
            "uncertainty_score": "prob_magnitude = max(p_hat, 1 - p_hat)",
            "rule": (
                "rank predicted-negative examples only by probability magnitude; "
                "reject least confident predicted negatives; retain all predicted positives"
            ),
            "threshold": float(threshold),
            "magnitude_threshold": float(magnitude_threshold) if not np.isnan(magnitude_threshold) else np.nan,
            "retain_fraction_requested": float(retain_fraction) if retain_fraction is not None else None,
            "retain_fraction_actual": retained_metrics["fraction"],
            "reject_fraction_actual": rejected_metrics["fraction"],
            "n_predicted_negative": int(np.sum(negative_pred_mask)),
            "n_predicted_positive": int(np.sum(positive_pred_mask)),
            "n_rejected": int(np.sum(reject_mask)),
            "pred_method": pred_method,
        },
        "overall": overall_metrics,
        "retained": retained_metrics,
        "rejected": rejected_metrics,
        "filtered_composition": filtered_composition,
        "filtering_quality": {
            "mean_prob_magnitude_retained": retained_metrics["mean_prob_magnitude"],
            "mean_prob_magnitude_rejected": rejected_metrics["mean_prob_magnitude"],
            "mean_width_retained": retained_metrics["mean_width"],
            "mean_width_rejected": rejected_metrics["mean_width"],
        },
        "arrays": {
            # Full arrays
            "p0": p0,
            "p1": p1,
            "width": width,
            "p_hat": p_hat,
            "prob_magnitude": prob_magnitude,
            "y_true": y_true,
            "y_pred": y_pred,
            "negative_pred_mask": negative_pred_mask,
            "positive_pred_mask": positive_pred_mask,
            "retain_mask": retain_mask,
            "reject_mask": reject_mask,

            # Retained examples
            "retained_p0": p0[retain_mask],
            "retained_p1": p1[retain_mask],
            "retained_width": width[retain_mask],
            "retained_p_hat": p_hat[retain_mask],
            "retained_probs": p_hat[retain_mask],
            "retained_prob_magnitude": prob_magnitude[retain_mask],
            "retained_y_true": y_true[retain_mask],
            "retained_y_pred": y_pred[retain_mask],

            # Rejected examples
            "rejected_p0": p0[reject_mask],
            "rejected_p1": p1[reject_mask],
            "rejected_width": width[reject_mask],
            "rejected_p_hat": p_hat[reject_mask],
            "rejected_probs": p_hat[reject_mask],
            "rejected_prob_magnitude": prob_magnitude[reject_mask],
            "rejected_y_true": y_true[reject_mask],
            "rejected_y_pred": y_pred[reject_mask],
        },
    }

    return results

def evaluate_va_expected_cost_filtering(
    intervals,
    y_true,
    cost_fn,
    threshold=0.5,
    reject_fraction=0.20,
    pred_method="collapse",
    n_curve_points=50
):
    """
    Evaluate Venn-Abers intervals by filtering examples with highest expected
    threshold-action cost.

    For each example:
        1. Convert VA interval to a single probability p_hat.
        2. Predict class using chosen threshold.
        3. Compute cost_fn(p_hat), e.g. compute_single_prediction_cost(p_hat).
        4. If predicted positive, use expected_cost_if_positive.
           If predicted negative, use expected_cost_if_negative.
        5. Remove the top reject_fraction highest expected action-cost examples.

    Parameters
    ----------
    intervals : array-like of shape (n_samples, 2)
        Venn-Abers intervals stored as [[p0, p1], [p0, p1], ...].
        Order can be either [lower, upper] or [upper, lower]; it is corrected internally.

    y_true : array-like of shape (n_samples,) or (n_samples, 1)
        Ground-truth binary labels in {0,1}.

    cost_fn : callable
        Function taking a single probability and returning a dictionary containing:
            - expected_cost_if_positive
            - expected_cost_if_negative

        Example:
            compute_single_prediction_cost(p)

    threshold : float, default=0.5
        Classification threshold used to decide whether each example is predicted
        positive or negative.

    reject_fraction : float, default=0.10
        Fraction of examples to remove. reject_fraction=0.10 removes the top 10%
        highest expected action-cost examples.

    pred_method : {"collapse", "midpoint"}, default="collapse"
        How to convert the VA interval into a single probability:
            - "collapse": p = p1 / (1 - p0 + p1)
            - "midpoint": p = 0.5 * (p0 + p1)

    n_curve_points : int, default=50
        Number of points for risk / accuracy / cost coverage curves.

    plot : bool, default=True
        If True, produce plots.

    axes : tuple of matplotlib axes, optional
        Optional tuple/list of 4 axes:
            (ax_risk, ax_acc, ax_cost, ax_hist)

    Returns
    -------
    results : dict
        Dictionary containing:
            - retained / rejected masks
            - retained probabilities and ground truths
            - expected costs
            - overall / retained / rejected metrics
            - counts of filtered TP, FP, TN, FN
            - curve arrays
            - optional figure / axes
    """

    # ----------------------------
    # 1) Input cleaning
    # ----------------------------
    intervals = np.asarray(intervals, dtype=float)
    y_true = np.asarray(y_true, dtype=int).reshape(-1)

    if intervals.ndim != 2 or intervals.shape[1] != 2:
        raise ValueError("intervals must have shape (n_samples, 2), e.g. [[p0, p1], ...].")

    if len(y_true) != len(intervals):
        raise ValueError("intervals and y_true must have the same number of samples.")

    if not np.all(np.isin(y_true, [0, 1])):
        raise ValueError("y_true must contain only 0/1 labels.")

    if cost_fn is None or not callable(cost_fn):
        raise ValueError("cost_fn must be callable, e.g. compute_single_prediction_cost.")

    if not (0 <= reject_fraction < 1):
        raise ValueError("reject_fraction must be in [0, 1).")

    # Enforce p0 <= p1
    p0 = np.minimum(intervals[:, 0], intervals[:, 1])
    p1 = np.maximum(intervals[:, 0], intervals[:, 1])

    n_samples = len(y_true)

    # ----------------------------
    # 2) Point prediction
    # ----------------------------
    width = p1 - p0

    if pred_method == "collapse":
        denom = np.maximum(1.0 - p0 + p1, 1e-15)
        p_hat = p1 / denom
    elif pred_method == "midpoint":
        p_hat = 0.5 * (p0 + p1)
    else:
        raise ValueError("pred_method must be 'collapse' or 'midpoint'.")

    p_hat = np.clip(p_hat, 1e-15, 1 - 1e-15)

    y_pred = (p_hat >= threshold).astype(int)

    prob_magnitude = np.maximum(p_hat, 1.0 - p_hat)

    # ----------------------------
    # 3) Compute per-example expected action cost
    # ----------------------------
    cost_outputs = []
    expected_cost_if_positive = np.empty(n_samples, dtype=float)
    expected_cost_if_negative = np.empty(n_samples, dtype=float)

    for i, p in enumerate(p_hat):
        out = cost_fn(float(p))
        cost_outputs.append(out)

        if "expected_cost_if_positive" not in out:
            raise KeyError("cost_fn output must contain 'expected_cost_if_positive'.")

        if "expected_cost_if_negative" not in out:
            raise KeyError("cost_fn output must contain 'expected_cost_if_negative'.")

        expected_cost_if_positive[i] = float(out["expected_cost_if_positive"])
        expected_cost_if_negative[i] = float(out["expected_cost_if_negative"])

    # Chosen expected cost under threshold-based action
    expected_action_cost = np.where(
        y_pred == 1,
        expected_cost_if_positive,
        expected_cost_if_negative,
    )

    # ----------------------------
    # 4) Filter top X% highest expected action costs
    # ----------------------------
    n_reject = int(np.ceil(reject_fraction * n_samples))

    reject_mask = np.zeros(n_samples, dtype=bool)

    if n_reject > 0:
        # Descending order by cost
        reject_idx = np.argsort(-expected_action_cost)[:n_reject]
        reject_mask[reject_idx] = True

    retain_mask = ~reject_mask

    cost_threshold = (
        float(np.min(expected_action_cost[reject_mask]))
        if np.sum(reject_mask) > 0
        else np.inf
    )

    # ----------------------------
    # 5) Metric helper
    # ----------------------------
    def _subset_metrics(mask, label):
        n = int(np.sum(mask))

        if n == 0:
            return {
                "subset": label,
                "count": 0,
                "fraction": 0.0,
                "accuracy": np.nan,
                "risk": np.nan,
                "sensitivity": np.nan,
                "specificity": np.nan,
                "precision": np.nan,
                "prevalence": np.nan,
                "brier": np.nan,
                "log_loss": np.nan,
                "mean_width": np.nan,
                "median_width": np.nan,
                "mean_p_hat": np.nan,
                "median_p_hat": np.nan,
                "mean_prob_magnitude": np.nan,
                "median_prob_magnitude": np.nan,
                "mean_expected_action_cost": np.nan,
                "median_expected_action_cost": np.nan,
                "min_expected_action_cost": np.nan,
                "max_expected_action_cost": np.nan,
                "total_expected_action_cost": np.nan,
                "tp": 0,
                "fp": 0,
                "tn": 0,
                "fn": 0,
            }

        yt = y_true[mask]
        yp = y_pred[mask]
        ph = p_hat[mask]
        wd = width[mask]
        mag = prob_magnitude[mask]
        ec = expected_action_cost[mask]

        tp = int(np.sum((yt == 1) & (yp == 1)))
        fp = int(np.sum((yt == 0) & (yp == 1)))
        tn = int(np.sum((yt == 0) & (yp == 0)))
        fn = int(np.sum((yt == 1) & (yp == 0)))

        acc = float(np.mean(yp == yt))
        risk = float(1.0 - acc)

        P = tp + fn
        N = tn + fp

        sensitivity = float(tp / P) if P > 0 else np.nan
        specificity = float(tn / N) if N > 0 else np.nan
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else np.nan
        prevalence = float(np.mean(yt))

        brier = float(brier_score_loss(yt, ph))

        try:
            ll = float(log_loss(yt, ph, labels=[0, 1]))
        except ValueError:
            ll = np.nan

        return {
            "subset": label,
            "count": n,
            "fraction": float(n / n_samples),
            "accuracy": acc,
            "risk": risk,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "precision": precision,
            "prevalence": prevalence,
            "brier": brier,
            "log_loss": ll,
            "mean_width": float(np.mean(wd)),
            "median_width": float(np.median(wd)),
            "mean_p_hat": float(np.mean(ph)),
            "median_p_hat": float(np.median(ph)),
            "mean_prob_magnitude": float(np.mean(mag)),
            "median_prob_magnitude": float(np.median(mag)),
            "mean_expected_action_cost": float(np.mean(ec)),
            "median_expected_action_cost": float(np.median(ec)),
            "min_expected_action_cost": float(np.min(ec)),
            "max_expected_action_cost": float(np.max(ec)),
            "total_expected_action_cost": float(np.sum(ec)),
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
        }

    overall_metrics = _subset_metrics(np.ones(n_samples, dtype=bool), "overall")
    retained_metrics = _subset_metrics(retain_mask, "retained")
    rejected_metrics = _subset_metrics(reject_mask, "rejected")

    # ----------------------------
    # 6) Filtered examples composition
    # ----------------------------
    filtered_tp = int(np.sum(reject_mask & (y_true == 1) & (y_pred == 1)))
    filtered_fp = int(np.sum(reject_mask & (y_true == 0) & (y_pred == 1)))
    filtered_tn = int(np.sum(reject_mask & (y_true == 0) & (y_pred == 0)))
    filtered_fn = int(np.sum(reject_mask & (y_true == 1) & (y_pred == 0)))

    filtered_count = int(np.sum(reject_mask))

    filtered_composition = {
        "filtered_count": filtered_count,
        "filtered_fraction": float(filtered_count / n_samples),
        "filtered_tp": filtered_tp,
        "filtered_fp": filtered_fp,
        "filtered_tn": filtered_tn,
        "filtered_fn": filtered_fn,
        "filtered_predicted_positive": int(np.sum(reject_mask & (y_pred == 1))),
        "filtered_predicted_negative": int(np.sum(reject_mask & (y_pred == 0))),
        "fraction_filtered_tp": float(filtered_tp / filtered_count) if filtered_count > 0 else np.nan,
        "fraction_filtered_fp": float(filtered_fp / filtered_count) if filtered_count > 0 else np.nan,
        "fraction_filtered_tn": float(filtered_tn / filtered_count) if filtered_count > 0 else np.nan,
        "fraction_filtered_fn": float(filtered_fn / filtered_count) if filtered_count > 0 else np.nan,
    }

    # ----------------------------
    # 10) Return
    # ----------------------------
    results = {
        "filtering_rule": {
            "filter_score": "expected_action_cost",
            "rule": (
                "predict using threshold; use expected_cost_if_positive for predicted positives "
                "and expected_cost_if_negative for predicted negatives; reject highest costs"
            ),
            "threshold": float(threshold),
            "reject_fraction_requested": float(reject_fraction),
            "reject_fraction_actual": rejected_metrics["fraction"],
            "retain_fraction_actual": retained_metrics["fraction"],
            "n_rejected": int(np.sum(reject_mask)),
            "cost_threshold": cost_threshold,
            "pred_method": pred_method,
        },
        "overall": overall_metrics,
        "retained": retained_metrics,
        "rejected": rejected_metrics,
        "filtered_composition": filtered_composition,
        "filtering_quality": {
            "mean_expected_action_cost_retained": retained_metrics["mean_expected_action_cost"],
            "mean_expected_action_cost_rejected": rejected_metrics["mean_expected_action_cost"],
            "mean_width_retained": retained_metrics["mean_width"],
            "mean_width_rejected": rejected_metrics["mean_width"],
            "mean_prob_magnitude_retained": retained_metrics["mean_prob_magnitude"],
            "mean_prob_magnitude_rejected": rejected_metrics["mean_prob_magnitude"],
        },
        "arrays": {
            # Full arrays
            "p0": p0,
            "p1": p1,
            "width": width,
            "p_hat": p_hat,
            "prob_magnitude": prob_magnitude,
            "y_true": y_true,
            "y_pred": y_pred,
            "expected_cost_if_positive": expected_cost_if_positive,
            "expected_cost_if_negative": expected_cost_if_negative,
            "expected_action_cost": expected_action_cost,
            "retain_mask": retain_mask,
            "reject_mask": reject_mask,

            # Remaining / retained examples
            "retained_p0": p0[retain_mask],
            "retained_p1": p1[retain_mask],
            "retained_width": width[retain_mask],
            "retained_p_hat": p_hat[retain_mask],
            "retained_probs": p_hat[retain_mask],
            "retained_prob_magnitude": prob_magnitude[retain_mask],
            "retained_y_true": y_true[retain_mask],
            "retained_y_pred": y_pred[retain_mask],
            "retained_expected_cost_if_positive": expected_cost_if_positive[retain_mask],
            "retained_expected_cost_if_negative": expected_cost_if_negative[retain_mask],
            "retained_expected_action_cost": expected_action_cost[retain_mask],

            # Filtered / rejected examples
            "rejected_p0": p0[reject_mask],
            "rejected_p1": p1[reject_mask],
            "rejected_width": width[reject_mask],
            "rejected_p_hat": p_hat[reject_mask],
            "rejected_probs": p_hat[reject_mask],
            "rejected_prob_magnitude": prob_magnitude[reject_mask],
            "rejected_y_true": y_true[reject_mask],
            "rejected_y_pred": y_pred[reject_mask],
            "rejected_expected_cost_if_positive": expected_cost_if_positive[reject_mask],
            "rejected_expected_cost_if_negative": expected_cost_if_negative[reject_mask],
            "rejected_expected_action_cost": expected_action_cost[reject_mask],
        },
        "cost_outputs": cost_outputs,
    }

    return results




import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import brier_score_loss, log_loss


def evaluate_va_prob_magnitude_filtering(
    intervals,
    y_true,
    magnitude_threshold=None,
    retain_fraction=.9,
    pred_method="collapse",
    n_curve_points=50
):
    """
    Evaluate Venn-Abers intervals when they are used to filter uncertain examples
    by probability magnitude rather than interval width.

    Higher probability magnitude means higher confidence:
        prob_magnitude = max(p_hat, 1 - p_hat)

    Examples with prob_magnitude >= threshold are retained.

    Returns the retained / remaining probabilities and corresponding ground truths:
        results["arrays"]["retained_p_hat"]
        results["arrays"]["retained_y_true"]
    """

    # ----------------------------
    # 1) Input cleaning
    # ----------------------------
    intervals = np.asarray(intervals, dtype=float)
    y_true = np.asarray(y_true, dtype=int).reshape(-1)

    if intervals.ndim != 2 or intervals.shape[1] != 2:
        raise ValueError("intervals must have shape (n_samples, 2), e.g. [[p0, p1], ...].")

    if len(y_true) != len(intervals):
        raise ValueError("intervals and y_true must have the same number of samples.")

    if not np.all(np.isin(y_true, [0, 1])):
        raise ValueError("y_true must contain only 0/1 labels.")

    # Enforce p0 <= p1
    p0 = np.minimum(intervals[:, 0], intervals[:, 1])
    p1 = np.maximum(intervals[:, 0], intervals[:, 1])

    # ----------------------------
    # 2) Point prediction and confidence score
    # ----------------------------
    width = p1 - p0

    if pred_method == "collapse":
        denom = np.maximum(1.0 - p0 + p1, 1e-15)
        p_hat = p1 / denom
    elif pred_method == "midpoint":
        p_hat = 0.5 * (p0 + p1)
    else:
        raise ValueError("pred_method must be 'collapse' or 'midpoint'.")

    p_hat = np.clip(p_hat, 1e-15, 1 - 1e-15)
    y_pred = (p_hat >= 0.5).astype(int)

    # Probability magnitude / confidence in predicted class
    prob_magnitude = np.maximum(p_hat, 1.0 - p_hat)

    # ----------------------------
    # 3) Choose filtering threshold
    # ----------------------------
    if magnitude_threshold is None and retain_fraction is None:
        raise ValueError("Provide either magnitude_threshold or retain_fraction.")

    if magnitude_threshold is not None and retain_fraction is not None:
        raise ValueError("Provide only one of magnitude_threshold or retain_fraction, not both.")

    if retain_fraction is not None:
        if not (0 < retain_fraction <= 1):
            raise ValueError("retain_fraction must be in (0, 1].")

        # Retain the top retain_fraction by probability magnitude.
        # Reject the least confident examples directly.
        n_samples = len(y_true)
        n_reject = int(np.ceil((1.0 - retain_fraction) * n_samples))
        
        order = np.argsort(prob_magnitude)  # lowest confidence first
        
        reject_mask = np.zeros(n_samples, dtype=bool)
        
        if n_reject > 0:
            reject_idx = order[:n_reject]
            reject_mask[reject_idx] = True
        
        retain_mask = ~reject_mask
        
        magnitude_threshold = (
            float(np.max(prob_magnitude[reject_mask]))
            if np.sum(reject_mask) > 0
            else np.nan
        )

    #retain_mask = prob_magnitude >= magnitude_threshold
    #reject_mask = ~retain_mask

    # ----------------------------
    # 4) Metric helper
    # ----------------------------
    def _subset_metrics(mask, label):
        n = int(np.sum(mask))

        if n == 0:
            return {
                "subset": label,
                "count": 0,
                "fraction": 0.0,
                "accuracy": np.nan,
                "risk": np.nan,
                "sensitivity": np.nan,
                "specificity": np.nan,
                "precision": np.nan,
                "prevalence": np.nan,
                "brier": np.nan,
                "log_loss": np.nan,
                "mean_width": np.nan,
                "median_width": np.nan,
                "mean_p_hat": np.nan,
                "mean_prob_magnitude": np.nan,
                "median_prob_magnitude": np.nan,
                "tp": 0,
                "fp": 0,
                "tn": 0,
                "fn": 0,
            }

        yt = y_true[mask]
        yp = y_pred[mask]
        ph = p_hat[mask]
        wd = width[mask]
        mag = prob_magnitude[mask]

        tp = int(np.sum((yt == 1) & (yp == 1)))
        fp = int(np.sum((yt == 0) & (yp == 1)))
        tn = int(np.sum((yt == 0) & (yp == 0)))
        fn = int(np.sum((yt == 1) & (yp == 0)))

        acc = float(np.mean(yp == yt))
        risk = float(1.0 - acc)

        P = tp + fn
        N = tn + fp

        sensitivity = float(tp / P) if P > 0 else np.nan
        specificity = float(tn / N) if N > 0 else np.nan
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else np.nan
        prevalence = float(np.mean(yt))

        brier = float(brier_score_loss(yt, ph))
        ll = float(log_loss(yt, ph, labels=[0, 1]))

        return {
            "subset": label,
            "count": n,
            "fraction": float(n / len(y_true)),
            "accuracy": acc,
            "risk": risk,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "precision": precision,
            "prevalence": prevalence,
            "brier": brier,
            "log_loss": ll,
            "mean_width": float(np.mean(wd)),
            "median_width": float(np.median(wd)),
            "mean_p_hat": float(np.mean(ph)),
            "mean_prob_magnitude": float(np.mean(mag)),
            "median_prob_magnitude": float(np.median(mag)),
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
        }

    overall_metrics = _subset_metrics(np.ones(len(y_true), dtype=bool), "overall")
    retained_metrics = _subset_metrics(retain_mask, "retained")
    rejected_metrics = _subset_metrics(reject_mask, "rejected")


    # ----------------------------
    # 8) Return
    # ----------------------------
    results = {
        "filtering_rule": {
            "uncertainty_score": "prob_magnitude = max(p_hat, 1 - p_hat)",
            "magnitude_threshold": float(magnitude_threshold),
            "retain_fraction": retained_metrics["fraction"],
            "pred_method": pred_method,
        },
        "overall": overall_metrics,
        "retained": retained_metrics,
        "rejected": rejected_metrics,
        "filtering_quality": {
            "mean_prob_magnitude_retained": retained_metrics["mean_prob_magnitude"],
            "mean_prob_magnitude_rejected": rejected_metrics["mean_prob_magnitude"],
            "mean_width_retained": retained_metrics["mean_width"],
            "mean_width_rejected": rejected_metrics["mean_width"],
        },
        "arrays": {
            # Full arrays
            "p0": p0,
            "p1": p1,
            "width": width,
            "p_hat": p_hat,
            "prob_magnitude": prob_magnitude,
            "y_true": y_true,
            "y_pred": y_pred,
            "retain_mask": retain_mask,
            "reject_mask": reject_mask,

            # Remaining / retained examples
            "retained_p0": p0[retain_mask],
            "retained_p1": p1[retain_mask],
            "retained_width": width[retain_mask],
            "retained_p_hat": p_hat[retain_mask],
            "retained_prob_magnitude": prob_magnitude[retain_mask],
            "retained_y_true": y_true[retain_mask],
            "retained_y_pred": y_pred[retain_mask],

            # Rejected examples, useful for checking what was filtered out
            "rejected_p0": p0[reject_mask],
            "rejected_p1": p1[reject_mask],
            "rejected_width": width[reject_mask],
            "rejected_p_hat": p_hat[reject_mask],
            "rejected_prob_magnitude": prob_magnitude[reject_mask],
            "rejected_y_true": y_true[reject_mask],
            "rejected_y_pred": y_pred[reject_mask],
        },
    }

    return results