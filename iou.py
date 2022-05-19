# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch, numpy as np

#Classes relabelled {-100,0,1,...,19}.
#Predictions will all be in the set {0,1,...,19}

#VALID_CLASS_IDS = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
VALID_CLASS_IDS = np.array([0, 1])
CLASS_LABELS = ['beam', 'board', 'bookcase', 'ceiling', 'chair', 'clutter', 'column', 'door', 'floor', 'sofa', 'table', 'wall', 'window']
UNKNOWN_ID = -100
N_CLASSES = len(CLASS_LABELS)

def printout(flog, data):
    flog.write(data + '\n')

def confusion_matrix(pred_ids, gt_ids):
    assert pred_ids.shape == gt_ids.shape, (pred_ids.shape, gt_ids.shape)
    idxs= gt_ids>=0
    return np.bincount(pred_ids[idxs]*13+gt_ids[idxs],minlength=169).reshape((13,13)).astype(np.ulonglong)

def get_iou(label_id, confusion):
    # true positives
    tp = np.longlong(confusion[label_id, label_id])
    # false positives
    fp = np.longlong(confusion[label_id, :].sum()) - tp
    # false negatives
    fn = np.longlong(confusion[:, label_id].sum()) - tp

    denom = (tp + fp + fn)
    if denom == 0:
        return float('nan')
    return (float(tp) / denom, tp, denom)

def evaluate(pred_ids,gt_ids,flog=None):
    print('evaluating', gt_ids.size, 'points...')
    confusion=confusion_matrix(pred_ids,gt_ids)
    class_ious = {}
    mean_iou = 0
    for i in range(N_CLASSES):
        label_name = CLASS_LABELS[i]
        class_ious[label_name] = get_iou(i, confusion)
        mean_iou+=class_ious[label_name][0]/13

    print('classes          IoU')
    print('----------------------------')
    for i in range(N_CLASSES):
        label_name = CLASS_LABELS[i]
        if flog is not None:
            printout(flog, '{0:<14s}: {1:>5.3f}   ({2:>6d}/{3:<6d})'.format(label_name, class_ious[label_name][0], class_ious[label_name][1], class_ious[label_name][2]))
        print('{0:<14s}: {1:>5.3f}   ({2:>6d}/{3:<6d})'.format(label_name, class_ious[label_name][0], class_ious[label_name][1], class_ious[label_name][2]))
    print('mean IOU: %f' % mean_iou)
    return mean_iou