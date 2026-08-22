import numpy as np

from sklearn.isotonic import IsotonicRegression

def predict_interval(test_input,preds_AF,gts_AF):
    preds_AF_app = np.copy(preds_AF)
    preds_AF_app = np.concatenate((preds_AF_app,np.expand_dims(test_input,axis=0)),axis=0)
    gts_AF_app_0 = np.copy(gts_AF)
    gts_AF_app_1 = np.copy(gts_AF)
    gts_AF_app_0 = np.concatenate((gts_AF_app_0,np.expand_dims(0,axis=0)),axis=0)
    gts_AF_app_1 = np.concatenate((gts_AF_app_1,np.expand_dims(1,axis=0)),axis=0)
    iso_low = IsotonicRegression().fit(preds_AF_app, gts_AF_app_0)
    iso_high = IsotonicRegression().fit(preds_AF_app, gts_AF_app_1)
    p_low = iso_low.predict(np.expand_dims(test_input,axis=0))
    p_high = iso_high.predict(np.expand_dims(test_input,axis=0))
    return p_low, p_high

def compute_af_screening_cost(
    calib_pred,
    calib_true,
    test_pred,
    test_true,
    threshold=0.5,
    predictions_are_probabilities=True,
    se_sp_source="test",        # "calib" or "test"
    prevalence_source="test",    # "calib" or "test"
    tp_mode="net",             # "gross" or "net"
    # Default economic parameters from the text
    C_device=300.0,
    C_nurse=76.15,
    C_confirm=614.0,
    C_phys=76.15,
    p_stroke_missed_af=0.037,#.025
    C_stroke_event=44929.0,
    C_OAC=2211.90,
    p_bleed_af=0.029,
    C_bleed_event=39236.33,
    C_TN=0.0,
    return_details=True,
):
    """
    Compute the static AF screening cost:

        C_bar = C_screen
              + pi * [Se * C_TP + (1 - Se) * C_FN]
              + (1 - pi) * [(1 - Sp) * C_FP + Sp * C_TN]

    Parameters
    ----------
    calib_pred, calib_true : array-like
        Predictions and ground truths for the calibration set.

    test_pred, test_true : array-like
        Predictions and ground truths for the test set.

    threshold : float, default=0.5
        Threshold used if predictions are probabilities.

    predictions_are_probabilities : bool, default=True
        If True, threshold predictions at `threshold`.
        If False, predictions are assumed to already be binary {0,1}.

    se_sp_source : {"calib", "test"}, default="calib"
        Which split to use to estimate sensitivity and specificity.

    prevalence_source : {"calib", "test"}, default="test"
        Which split to use to estimate prevalence pi.

    tp_mode : {"gross", "net"}, default="gross"
        - "gross": use full C_TP
        - "net": set C_TP = 0

    Economic defaults are set from the values in your text.

    Returns
    -------
    dict or float
        If return_details=True, returns a dict with cost, metrics, counts, and parameters.
        Otherwise returns only the scalar expected cost.
    """

    def _to_numpy_1d(arr, name):
        arr = np.asarray(arr).ravel()
        if arr.ndim != 1:
            raise ValueError(f"{name} must be 1D after flattening.")
        return arr

    def _to_binary(pred, true, split_name):
        pred = _to_numpy_1d(pred, f"{split_name}_pred")
        true = _to_numpy_1d(true, f"{split_name}_true").astype(int)

        if pred.shape[0] != true.shape[0]:
            raise ValueError(f"{split_name}_pred and {split_name}_true must have the same length.")

        if not np.all(np.isin(true, [0, 1])):
            raise ValueError(f"{split_name}_true must contain only 0/1 labels.")

        if predictions_are_probabilities:
            y_hat = (pred >= threshold).astype(int)
        else:
            y_hat = pred.astype(int)
            if not np.all(np.isin(y_hat, [0, 1])):
                raise ValueError(f"{split_name}_pred must contain only 0/1 labels when "
                                 f"predictions_are_probabilities=False.")
        return y_hat, true

    def _confusion_and_metrics(y_hat, y_true):
        TP = int(np.sum((y_true == 1) & (y_hat == 1)))
        FN = int(np.sum((y_true == 1) & (y_hat == 0)))
        TN = int(np.sum((y_true == 0) & (y_hat == 0)))
        FP = int(np.sum((y_true == 0) & (y_hat == 1)))

        P = TP + FN
        N = TN + FP

        Se = TP / P if P > 0 else np.nan
        Sp = TN / N if N > 0 else np.nan
        pi = np.mean(y_true) if len(y_true) > 0 else np.nan

        return {
            "TP": TP, "FN": FN, "TN": TN, "FP": FP,
            "Se": Se, "Sp": Sp, "pi": pi
        }

    # Convert predictions to binary decisions
    calib_hat, calib_true = _to_binary(calib_pred, calib_true, "calib")
    test_hat, test_true = _to_binary(test_pred, test_true, "test")

    # Metrics for each split
    calib_stats = _confusion_and_metrics(calib_hat, calib_true)
    test_stats = _confusion_and_metrics(test_hat, test_true)

    # Select metric source
    if se_sp_source == "calib":
        Se = calib_stats["Se"]
        Sp = calib_stats["Sp"]
    elif se_sp_source == "test":
        Se = test_stats["Se"]
        Sp = test_stats["Sp"]
    else:
        raise ValueError("se_sp_source must be 'calib' or 'test'.")

    if prevalence_source == "calib":
        pi = calib_stats["pi"]
    elif prevalence_source == "test":
        pi = test_stats["pi"]
    else:
        raise ValueError("prevalence_source must be 'calib' or 'test'.")

    # Derived costs from the text
    C_screen = C_device + C_nurse
    C_FP = C_confirm + C_phys
    C_FN = p_stroke_missed_af * C_stroke_event
    C_follow = C_phys
    C_bleed = p_bleed_af * C_bleed_event
    C_TP_gross = C_confirm + C_phys + C_OAC + C_follow + C_bleed

    if tp_mode == "gross":
        C_TP = C_TP_gross
    elif tp_mode == "net":
        C_TP = 0.0
        #C_screen = 0.0
    else:
        raise ValueError("tp_mode must be 'gross' or 'net'.")

    # Static expected cost formula
    expected_cost = (
        C_screen
        + pi * (Se * C_TP + (1.0 - Se) * C_FN)
        + (1.0 - pi) * ((1.0 - Sp) * C_FP + Sp * C_TN)
    )

    if not return_details:
        return float(expected_cost)

    return {
        "expected_cost": float(expected_cost),
        "used_for_Se_Sp": se_sp_source,
        "used_for_prevalence": prevalence_source,
        "tp_mode": tp_mode,
        "Se": float(Se),
        "Sp": float(Sp),
        "pi": float(pi),
        "calib_stats": calib_stats,
        "test_stats": test_stats,
        "cost_parameters": {
            "C_screen": float(C_screen),
            "C_TP": float(C_TP),
            "C_TP_gross": float(C_TP_gross),
            "C_FP": float(C_FP),
            "C_FN": float(C_FN),
            "C_TN": float(C_TN),
            "C_device": float(C_device),
            "C_nurse": float(C_nurse),
            "C_confirm": float(C_confirm),
            "C_phys": float(C_phys),
            "C_OAC": float(C_OAC),
            "C_follow": float(C_follow),
            "C_bleed": float(C_bleed),
            "p_stroke_missed_af": float(p_stroke_missed_af),
            "C_stroke_event": float(C_stroke_event),
            "p_bleed_af": float(p_bleed_af),
            "C_bleed_event": float(C_bleed_event),
        }
    }


import numpy as np

def compute_single_prediction_cost(
    pred,
    true_label=None,
    threshold=0.5,
    predictions_are_probabilities=True,
    tp_mode="net",
    C_device=300.0,
    C_nurse=76.15,
    C_confirm=614.0,
    C_phys=76.15,
    p_stroke_missed_af=0.037,#.025
    C_stroke_event=44929.0,
    C_OAC=2211.90,
    p_bleed_af=0.029,
    C_bleed_event=39236.33,
    C_TN=0.0,
    include_screen_cost_for_all=True,
    return_details=True,
):
    """
    Compute cost for a single prediction.

    Parameters
    ----------
    pred : float or int
        Either predicted probability (if predictions_are_probabilities=True)
        or binary prediction {0,1}.

    true_label : int or None
        If provided (0/1), returns realised cost for this prediction.
        If None and pred is a probability, returns expected cost for each action
        and for the threshold-based action.

    include_screen_cost_for_all : bool
        If True, C_screen is added regardless of decision.
        If False, C_screen is only added when predicted positive.

    Returns
    -------
    dict or float
    """

    # Derived costs
    C_screen = C_device + C_nurse
    C_FP = C_confirm + C_phys
    C_FN = p_stroke_missed_af * C_stroke_event
    C_follow = C_phys
    C_bleed = p_bleed_af * C_bleed_event
    C_TP_gross = C_confirm + C_phys + C_OAC + C_follow + C_bleed

    if tp_mode == "gross":
        C_TP = C_TP_gross
    elif tp_mode == "net":
        C_TP = 0.0
    else:
        raise ValueError("tp_mode must be 'gross' or 'net'.")

    # Case 1: true label is known -> realised cost
    if true_label is not None:
        if true_label not in (0, 1):
            raise ValueError("true_label must be 0 or 1.")

        if predictions_are_probabilities:
            y_hat = int(pred >= threshold)
            p = float(pred)
        else:
            y_hat = int(pred)
            if y_hat not in (0, 1):
                raise ValueError("Binary prediction must be 0 or 1.")
            p = None

        if true_label == 1 and y_hat == 1:
            event_cost = C_TP
            outcome = "TP"
        elif true_label == 1 and y_hat == 0:
            event_cost = C_FN
            outcome = "FN"
        elif true_label == 0 and y_hat == 1:
            event_cost = C_FP
            outcome = "FP"
        else:
            event_cost = C_TN
            outcome = "TN"

        total_cost = event_cost + (C_screen if include_screen_cost_for_all or y_hat == 1 else 0.0)

        if not return_details:
            return float(total_cost)

        return {
            "mode": "realised",
            "predicted_probability": p,
            "predicted_label": y_hat,
            "true_label": int(true_label),
            "outcome": outcome,
            "cost": float(total_cost),
            "event_cost": float(event_cost),
            "C_screen": float(C_screen),
            "cost_parameters": {
                "C_TP": float(C_TP),
                "C_FP": float(C_FP),
                "C_FN": float(C_FN),
                "C_TN": float(C_TN),
            }
        }

    # Case 2: true label unknown -> expected cost using individual probability
    if not predictions_are_probabilities:
        raise ValueError(
            "If true_label is None, pred should usually be a probability "
            "(predictions_are_probabilities=True)."
        )

    p = float(pred)
    if not (0.0 <= p <= 1.0):
        raise ValueError("Probability pred must be between 0 and 1.")

    # Expected cost if action is positive
    EC_positive = p * C_TP + (1.0 - p) * C_FP
    if include_screen_cost_for_all:
        EC_positive += C_screen
    #else:
    #    EC_positive += C_screen

    # Expected cost if action is negative
    EC_negative = p * C_FN + (1.0 - p) * C_TN
    if include_screen_cost_for_all:
        EC_negative += C_screen

    # Threshold-based action
    y_hat = int(p >= threshold)
    EC_threshold_action = EC_positive if y_hat == 1 else EC_negative

    # Cost-minimising action
    optimal_action = 1 if EC_positive <= EC_negative else 0
    EC_optimal = min(EC_positive, EC_negative)

    if not return_details:
        return float(EC_threshold_action)

    return {
        "mode": "expected",
        "predicted_probability": p,
        "threshold": float(threshold),
        "threshold_based_label": y_hat,
        "expected_cost_if_positive": float(EC_positive),
        "expected_cost_if_negative": float(EC_negative),
        "expected_cost_threshold_action": float(EC_threshold_action),
        "cost_minimising_action": int(optimal_action),
        "expected_cost_optimal_action": float(EC_optimal),
        "cost_parameters": {
            "C_screen": float(C_screen),
            "C_TP": float(C_TP),
            "C_FP": float(C_FP),
            "C_FN": float(C_FN),
            "C_TN": float(C_TN),
        }
    }