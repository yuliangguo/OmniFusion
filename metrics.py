import torch

#==========================
# Depth Prediction Metrics
#==========================

def abs_rel_error(pred, gt, mask):
    '''Compute absolute relative difference error'''
    return ((pred[mask>0] - gt[mask>0]).abs() / gt[mask>0]).mean()

def sq_rel_error(pred, gt, mask):
    '''Compute squared relative difference error'''
    return (((pred[mask>0] - gt[mask>0]) ** 2) / gt[mask>0]).mean()

def lin_rms_sq_error(pred, gt, mask):
    '''Compute the linear RMS error except the final square-root step'''
    return ((pred[mask>0] - gt[mask>0]) ** 2).mean()

def log_rms_sq_error(pred, gt, mask):
    '''Compute the log RMS error except the final square-root step'''
    mask = (mask > 0) & (pred > 1e-7) & (gt > 1e-7) # Compute a mask of valid values
    return ((pred[mask].log() - gt[mask].log()) ** 2).mean()

def delta_inlier_ratio(pred, gt, mask, degree=1):
    '''Compute the delta inlier rate to a specified degree (def: 1)'''
    return (torch.max(pred[mask>0] / gt[mask>0], gt[mask>0] / pred[mask>0]) < (1.25 ** degree)).float().mean()