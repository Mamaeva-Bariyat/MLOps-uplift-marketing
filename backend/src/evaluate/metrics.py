import numpy as np


def auuc_score(y_true, uplift_pred, treatment):
    order = np.argsort(-uplift_pred)
    y_true = np.asarray(y_true)[order]
    treatment = np.asarray(treatment)[order]

    n_treat = treatment.sum()
    n_control = len(treatment) - n_treat

    if n_treat == 0 or n_control == 0:
        return 0.0

    cum_gain_t = np.cumsum(y_true * treatment)
    cum_gain_c = np.cumsum(y_true * (1 - treatment))
    uplift_curve = (cum_gain_t / n_treat) - (cum_gain_c / n_control)

    treated_fraction = np.cumsum(treatment) / n_treat
    return np.trapz(uplift_curve, treated_fraction)