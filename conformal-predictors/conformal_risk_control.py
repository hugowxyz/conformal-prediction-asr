from scipy.optimize import brentq
import numpy as np

n=500 # number of calibration points
alpha = 0.1 # 1-alpha is the desired false negative rate

idx = np.array([1] * n + [0] * (sgmd.shape[0]-n)) > 0
np.random.shuffle(idx)
cal_sgmd, val_sgmd = sgmd[idx,:], sgmd[~idx,:]
cal_gt_masks, val_gt_masks = gt_masks[idx], gt_masks[~idx]

def false_negative_rate(pred_masks, true_masks):
    return 1-((pred_masks * true_masks).sum(axis=1).sum(axis=1)/true_masks.sum(axis=1).sum(axis=1)).mean()

def lamhat_threshold(lam): 
    return false_negative_rate(cal_sgmd>=lam, cal_gt_masks) - ((n+1)/n*alpha - 1/(n+1))

lamhat = brentq(lamhat_threshold, 0, 1)
predicted_masks = val_sgmd >= lamhat