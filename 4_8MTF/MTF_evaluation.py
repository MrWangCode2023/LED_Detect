import numpy as np


def mtf_evaluation(mtf_curve):
    cutoff_freq = np.argmax(mtf_curve < 0.1) if np.any(mtf_curve < 0.1) else len(mtf_curve) - 1
    mtf50_freq = np.argmax(mtf_curve < 0.5) if np.any(mtf_curve < 0.5) else len(mtf_curve) - 1
    mtf10_freq = np.argmax(mtf_curve < 0.1) if np.any(mtf_curve < 0.1) else len(mtf_curve) - 1
    auc = np.trapz(mtf_curve)
    return {
        'Cutoff Frequency (lp/mm)': cutoff_freq,
        'MTF50 Frequency (lp/mm)': mtf50_freq,
        'MTF10 Frequency (lp/mm)': mtf10_freq,
        'Area Under Curve (AUC)': auc
    }