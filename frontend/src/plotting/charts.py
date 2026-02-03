import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_qini_curve(y_true, uplift_pred, treatment):
    """Plot Qini curve."""
    order = np.argsort(-uplift_pred)
    y_sorted = np.asarray(y_true)[order]
    t_sorted = np.asarray(treatment)[order]
    cum_treated = np.cumsum(t_sorted)
    cum_control = np.cumsum(1 - t_sorted)
    cum_response_t = np.cumsum(y_sorted * t_sorted) / np.maximum(cum_treated, 1)
    cum_response_c = np.cumsum(y_sorted * (1 - t_sorted)) / np.maximum(cum_control, 1)
    cum_uplift = cum_response_t - cum_response_c
    treated_fraction = cum_treated / cum_treated[-1] if cum_treated[-1] > 0 else np.zeros(len(cum_uplift))

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.plot(treated_fraction, cum_uplift, label='Model', linewidth=3)
    ax.plot([0, 1], [0, cum_uplift[-1]], 'k--', label='Ideal')
    ax.plot([0, 1], [0, 0], 'k-', label='Random')
    ax.set_title('Qini Curve')
    ax.set_xlabel('Fraction of Treated Customers')
    ax.set_ylabel('Cumulative Uplift')
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig


def plot_uplift_distribution(df):
    """Plot uplift score distribution."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df['uplift_score'], bins=50, kde=True, ax=ax)
    ax.set_title('Uplift Score Distribution')
    return fig