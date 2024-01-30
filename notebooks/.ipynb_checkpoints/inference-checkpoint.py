from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from object_detection.utils import utils_frcnn
import os

###################################################################################################
# Specify the base path and path to model predictions

base_path = "/p/project/hai_hmgu/workspace/harsha/object_detection/data/orgaquant"
pred_json = f"{base_path}/predictions/2024-01-30_14-57-02/model_predictions.json"

###################################################################################################

# Get the predicted bboxes and load the ground truth bounding boxes for testset

image_name,pred_bb,pred_scores = utils_frcnn.load_preds('/p/project/hai_hmgu/workspace/harsha/object_detection/data/predictions/2024-01-30_14-57-02/model_predictions.json')
test_df = pd.read_csv(f'{base_path}/test_labels.csv',sep = ',')
gt_bb = []
for img in image_name:
    full_image_path = f'{base_path}/test/{img}'
    tempdf = test_df[test_df.path == full_image_path]
    boxes = [tempdf.iloc[i,:4].tolist() for i in range(len(tempdf))]
    gt_bb.append(boxes)


##################################################################################################
# Inference on grid of box score threshold and save precision,recall values for different box score thresholds

print('Starting to compute metrics for different box_score_thresh values for Test set')
# Compute eval metrics for different box_score_thresh values

thresh_grid = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

tp_coll = []
fp_coll = []
fn_coll = []
prec_coll = []
rec_coll = []
for i in thresh_grid:
    bst_pred_bb = utils_frcnn.bboxes_bst(pred_bb,pred_scores,i)
    TP,FP,FN,TPFLAG = utils_frcnn.evalMetrics(gt_bb,bst_pred_bb,0.5)
    TTP = np.sum(TP)
    TFP = np.sum(FP)
    TFN = np.sum(FN)
    PREC = TTP / (TTP + TFP)
    RECAL = TTP / (TTP + TFN)
    print(f'For box_score_thresh {i}, Total TPs = {TTP}, Total FPs = {TFP}, Total FNs = {TFN}, Overall Precision = {PREC}, Overall Recall = {RECAL} ')

    tp_coll.append(TTP)
    fp_coll.append(TFP)
    fn_coll.append(TFN)
    prec_coll.append(PREC)
    rec_coll.append(RECAL)

test_prec_rec_arr = np.zeros((2,len(prec_coll)))
test_prec_rec_arr[0,:] = prec_coll
test_prec_rec_arr[1,:] = rec_coll

if not os.path.exists(f"{base_path}/metrics"):
    os.makedirs(f"{base_path}/metrics",exist_ok=True)
   
np.save(f'{base_path}/metrics/test_prec_rec_arr.npy',test_prec_rec_arr)
print('Metrics are computed and saved...')

    