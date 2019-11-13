from DataSet_custom import ModelConfig
from mrcnn import model as modellib
from mrcnn import visualize
from mrcnn import utils

import os
import shutil
import numpy as np
import pandas as pd
import gc
import copy
import cv2
import math

from keras import backend as K

import time

def Test_Dataset(TaskID, savePath, classList, modelConfig, modelPath, testData, 
                 saveFig = False, saveMask = False, IoU_thr = 0):
    
    MODEL_DIR = ""
    if os.path.isfile(modelPath):
        MODEL_DIR = os.path.dirname(modelPath)
    elif os.path.isdir(modelPath):
        MODEL_DIR = modelPath
    else:
        print ("No exist path : ",modelPath)

    SAVE_DIR = savePath + "/Test_" + TaskID
    if(not os.path.isdir(SAVE_DIR)):
        os.makedirs(SAVE_DIR)
    
    Mask_DIR = None
    if (saveMask):
        Mask_DIR = SAVE_DIR + "/Mask"
        if(not os.path.isdir(Mask_DIR)):
            os.makedirs(Mask_DIR)
    Figure_DIR = None
    if (saveFig):
        Figure_DIR = SAVE_DIR + "/Fig" # Fig: overlay,contour,bbox,score, ...
        if(not os.path.isdir(Figure_DIR)):
            os.makedirs(Figure_DIR)

    print("prepare Data...")
    testData.prepare()
    print("prepare Data Done !")

    print("load Model...")
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=modelConfig)
    if os.path.isdir(modelPath):
        modelPath = model.find_last()
    model.load_weights(modelPath, by_name=True)
    print("load Model Done !")

    
    Classes = copy.deepcopy(classList)
    Classes.insert(0, "BG")
    print ("Classes : ", Classes)
    
    ##############################################################
    # Run detection
    print("Test Run...")
    
    for i, image_id in enumerate(testData._image_ids) :
        
        testImage, testFileName = testData.load_image_custom(image_id)
        gt_masks, gt_class_ids = testData.load_mask_custom(image_id, testImage.shape)
        
        start_T = time.time()
        ##########################################################     
        # Detection
        result = model.detect([testImage], verbose=0)[0] # batch=1
        
        spend_T = time.time() - start_T
        ######################################################            
        # Display & Save 
        if (saveFig) :
            saveFileName = os.path.splitext(testFileName)[0] + ".png"
            if (testImage.shape[2]>1) :
                testImage = testImage[:,:,0] # artery phase
            visualize.save_result_figures(testImage, result, Classes, saveFileName, \
                                        truemasks = gt_masks, truemasks_class_id = gt_class_ids,\
                                        maskDir = Mask_DIR, figDir = Figure_DIR)
        
        if (gt_masks is None) : 
            continue
           
        ##########################################################            
        # Evaluation
        bboxes = result['rois']
        class_ids = result['class_ids']
        masks = result['masks']
        scores = result['scores']
        
        if(len(bboxes) == 0):
            continue

        FP_count = 0
        Detected = [False] * gt_masks.shape[2] # for TP & FN
        IoUs = []
        for i_pred in range(len(bboxes)):
            # y1, x1, y2, x2
            bbox = bboxes[i_pred]
            class_id = class_ids[i_pred]
            mask = masks[:,:,i_pred]
            score = scores[i_pred]
            
            if class_id == 0:
                continue
            
            truemasks = gt_masks[:,:,np.where(gt_class_ids==class_id)[0]]
            FP = True
            for i_true in range(truemasks.shape[2]):
                truemask = gt_masks[:,:,np.where(gt_class_ids==class_id)[0][i_true]]
                truemask = np.asarray(truemask, dtype = "uint8")
                _, truemask_binary = cv2.threshold(truemask, 0, 1, cv2.THRESH_BINARY)

                true_bbox = utils.extract_bboxes(np.expand_dims(truemask_binary,-1))[0] #1 truemask - 1 instance
                if not np.any(true_bbox):
                    # no ground-truth
                    continue

                mask = np.asarray(mask)
                iou = utils.compute_iou(true_bbox, np.asarray([bbox]), 
                                        (true_bbox[2]-true_bbox[0])*(true_bbox[3]-true_bbox[1]), 
                                        np.array([(bbox[2]-bbox[0])*(bbox[3]-bbox[1])]))[0]
                
                if iou>IoU_thr:
                    # TP : over IoU-threshold
                    Detected[np.where(gt_class_ids==class_id)[0][i_true]] = True 
                    IoUs.append(iou)
                    FP = False
                    
                    
            if FP:
                FP_count += 1
                
        
        TP_count = len(np.where(Detected)[0])
        Detected[:] = [not x for x in Detected]
        FN_count = len(np.where(Detected)[0])
        
        print ("{0}  :  TP({1}), IoU({2}) / FP({3}) / FN({4}) ".format(testFileName, TP_count, IoUs, FP_count, FN_count))
        
        
    del model
    gc.collect()
    K.clear_session()

    print("Test Done ! : ", TaskID)
    
    