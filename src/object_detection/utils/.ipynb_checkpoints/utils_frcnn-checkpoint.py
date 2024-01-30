from object_detection.datamodules import frcnn_dataset
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import torch
import json


def resize(img_rgb,rescaling_factor):
    """Function resizes the input rgb image by a input rescaling factor

    Args:
        img_rgb (array): Input RGB image in array format
        rescaling_factor (float): Rescaling factor

    Returns:
        array: Return resized image
    """
    img_reshape = image = np.transpose(img_rgb,(2,0,1))
    img_tensor = torch.tensor(img_reshape)
    res = transforms.Resize(size=(int(rescaling_factor*image.shape[1])))(img_tensor)
    img_arr = res.numpy()
    img_arr = np.moveaxis(img_arr,0,-1)
    return img_arr
    

def apply_nms(orig_prediction, iou_thresh=0.2):
    """Function applies non max suppression to iteratively remove lower scoring boxes which have an IoU greater than iou_threshold with another (higher scoring) box.

    Args:
        orig_prediction (Dictionary): Model output for each i9nput image.
        iou_thresh (float, optional): IOU threshold. Defaults to 0.2.

    Returns:
        final_prediction: Dictionary containing boxes, scores and labels after nms.
    """
    
    # torchvision returns the indices of the bboxes to keep
    keep = torchvision.ops.nms(orig_prediction['boxes'], orig_prediction['scores'], iou_thresh)
    
    final_prediction = orig_prediction
    final_prediction['boxes'] = final_prediction['boxes'][keep]
    final_prediction['scores'] = final_prediction['scores'][keep]
    final_prediction['labels'] = final_prediction['labels'][keep]
    
    return final_prediction

def check_annotdf(df,path_input):
    """Function to correct the order of bounding box coordinates and remove the rows with negative coordinates

    Args:
        df (Dataframe): Annotations for input images

    Returns:
        df: Corrected annotation dataframe
    """

    rmvind = df[(df.iloc[:,:4].values <= 0)].index.tolist()
    df.drop(rmvind,axis=0,inplace=True)
    df = df.reset_index(drop=True)

    return df

def prediction(data_test,model):
    """Function to perform inference on test dataset and postprocess predictions from FRCNN.

    Args:
        data_test (Dataset): Test dataset.
        model (model class): Pretrained FRCNN mode.

    Returns:
        GTBB (List): List of GT bounding boxes.
        PREDBB (List): List of respective FRCNN predicted bounding boxes.
        IMG (List): List of input test images.
        GTMASK (List): List of respective GT masks.
        PREDMASK (List): List of respective predicted masks
        SCORE (List): List of scores obtained from FRCNN for each predicted bounding boxes.
    """

    GTBB = []
    PREDBB = []
    IMG = []
    SCORE = []

    # Set the pretrained model to evaluation mode to perform prediction
    model = model.eval()
    
    for i in range(len(data_test)):

        images, targets = data_test[i]
        images = [images.to('cuda')]
        targets = [{k: v.to('cuda') for k, v in targets.items()}]
        output = model(images)[0]

        # Uncomment to apply nms
        output = apply_nms(output, iou_thresh=0.2)
        
        
        IMG.append(images[0].cpu().detach().numpy())
        GTBB.append(targets[0]['boxes'].cpu().detach())
        PREDBB.append(output['boxes'].cpu().detach()) 
        SCORE.append(output['scores'].cpu().detach().numpy()) 

        
    return GTBB, PREDBB, IMG, SCORE


def bboxes_bst(pred_boxes, pred_scores, thresh):
    """Function to return bounding boxes greater than input confidence score threshold.
    
    Args:
        pred_boxes (List): List of predicted bounding boxes.
        pred_scores (List): List of confidence scores for each of predicted bounding boxes.
        
    Returns:
        bboxes (List): List of bounding boxes greater than input confidence score threshold.
    """
    bboxes = []
    for i in range(len(pred_boxes)):
        temp_boxes = []
        temp_scores = []
        temp_labels = []
        for j in range(len(pred_boxes[i])):
            if pred_scores[i][j] > thresh:
                temp_boxes.append(pred_boxes[i][j])
        bboxes.append(temp_boxes)
    return bboxes

def load_preds(json_path):
    """Function to load predictions from the json file.

    Args:
        json_path (str): Path to json file.

    Returns:
        image_name (List): List of image names in testset.
        bbox_collected (List): List of predicted bounding boxes.
        score_collected (List): List of confidence scores for predicted bounding boxes.
    """
    f = open(json_path)
    preds = json.load(f)
    bbox_collected = []
    score_collected = []
    image_name = preds.keys()
    for k1 in preds.keys():
        temp_box = []
        temp_score = []
        for k2 in preds[k1]:
            box = preds[k1][k2]['bbox']
            x1 = float(box[0])
            y1 = float(box[1])
            x2 = float(box[2])
            y2 = float(box[3])
            temp_box.append([x1, y1, x2, y2])
            temp_score.append(float(preds[k1][k2]['score']))
        bbox_collected.append(temp_box)  
        score_collected.append(temp_score) 
    return image_name, bbox_collected,score_collected

def get_iou(a, b, epsilon=1e-5, intersection_check=False):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    width =  (x2 - x1)
    height = (y2 - y1)

    if (width < 0) or (height < 0):
        if intersection_check:
            return 0.0, False
        else:
            return 0.0
    area_overlap = width * height

    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    area_combined = area_a + area_b - area_overlap

    iou = area_overlap / (area_combined + epsilon)
    if intersection_check:
        return iou, bool(area_overlap)
    else:
        return iou


def calc_conditions(gt_boxes, pred_boxes, iou_thresh=0.5):
    """Function to compute evaluation metrics (TP,FP,FN) based on IOU score.

    Args:
        gt_boxes (List): List of GT bounding boxes.
        pred_boxes (List): List of model predicted bounding boxes.
        iou_thresh (float, optional): Threshold to decide if detection is TP or FP. Defaults to 0.5.

    Returns:
        tp,fp,fn,tpflag (int,int,int,array): Count of number of TPs, FPs, FNs and the flag position of TPs.
    """
    gt_class_ids_ = np.zeros(len(gt_boxes))
    pred_class_ids_ = np.zeros(len(pred_boxes))

    tp, fp, fn = 0, 0, 0
    TPFlag = np.zeros((len(gt_class_ids_),len(pred_class_ids_)))
    for i in range(len(gt_class_ids_)):
        iou = []
        
        for j in range(len(pred_class_ids_)):
            now_iou = get_iou(gt_boxes[i], pred_boxes[j])
            if now_iou >= iou_thresh:
                iou.append(now_iou)
                gt_class_ids_[i] = 1
                pred_class_ids_[j] = 1
            else:
                iou.append(0)

        
        if any(iou):
            tp += 1
            ind = iou.index(max(iou))
            TPFlag[i,ind] = 1
            fp += len(iou) - 1 - iou.count(0)

        
    fn += np.count_nonzero(np.array(gt_class_ids_) == 0)
    fp += np.count_nonzero(np.array(pred_class_ids_) == 0)
    
    return tp, fp, fn, TPFlag
    

def evalMetrics(GTBB,BBColl,iou_thresh):
    """Function to compute and output the evaluated metrics.

    Args:
        GTBB (List): List of GT bounding boxes.
        BBColl (_type_): List of predicted bounding boxes.
        iou_thresh (_type_): IOU threshold.

    Returns:
        TP,FP,FN,TPFLAG (List,List,List,List): TPs,FPs,FNs of all the images stored in a list.
    """
    TP = []
    FP = []
    FN = []
    TPFLAG = []
    for i in range(len(GTBB)):
        if len(GTBB[i]) == 0:
            TP.append(0)
            FP.append(len(BBColl[i]))
            FN.append(0)
            TPFLAG.append([])
        else:
            tp,fp,fn,TPFlag = calc_conditions(GTBB[i],BBColl[i],iou_thresh)
            TP.append(tp)
            FP.append(fp)
            FN.append(fn)
            TPFLAG.append(TPFlag)
            
    return TP,FP,FN,TPFLAG


def bbox_to_rect(bbox, color):
    """Function to convert bounding box to matplotlib format.

    Args:
        bbox (List): List of bounding box coordinates
        color (String): Bounding box color name

    Returns:
        Bounding box in matplotlib format
    """


    return plt.Rectangle(
        xy=(bbox[0], bbox[1]), width=bbox[2]-bbox[0], height=bbox[3]-bbox[1],
        fill=False, edgecolor=color, linewidth=2)
