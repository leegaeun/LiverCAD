"""
Mask R-CNN
Configurations and data loading code for the synthetic Shapes dataset.
This is a duplicate of the code in the noteobook train_shapes.ipynb for easy
import into other notebooks, such as inspect_model.ipynb.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import fnmatch
import numpy as np
import cv2

from mrcnn import config
from mrcnn import utils
import os

import SimpleITK as sitk



class ModelConfig(config.Config):
    
    NAME = "Liver"  # Override in sub-classes

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    
    BACKBONE = "resnet101"
    
    PretainedModelPath = ""

    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
#     RPN_ANCHOR_RATIOS = [0.5, 1, 2]
    
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    
    IMAGE_CHANNEL_COUNT = 3
    
    MEAN_PIXEL = np.array([127.0, 127.0, 127.0])
    
    DETECTION_MIN_CONFIDENCE = 0.85

    
    def __init__(self, classCount):
        """Set values of computed attributes."""
        self.NUM_CLASSES = 1+classCount
        self.STEPS_PER_EPOCH = self.STEPS_PER_EPOCH / self.IMAGES_PER_GPU
        self.VALIDATION_STEPS = self.VALIDATION_STEPS / self.IMAGES_PER_GPU
        super(ModelConfig, self).__init__()

        
        
        
class Dataset(utils.Dataset):
    """Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """
    def set_dataset(self, imagePath, maskPath, dirDicts, classDicts):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        dirDicts: key=dirname, val=classname
        classDicts: key=classname, val=classid
        """
        
        if(imagePath is None or not os.path.isdir(imagePath)):
            print ("Wrong ImagePath")
            return
        
        for classname in classDicts:
            self.add_class("Liver", classDicts[classname], classname)
        print ("class_info", self.class_info)
        
        for file in os.listdir(imagePath):
            index = len(self.image_info)
            imageFilePath = imagePath + "/" + file
            
            maskFilePaths = []
            maskClasses = []
            if (maskPath):
                for dirname in dirDicts:
                    maskPath_cl = maskPath+'/'+dirname
                    if not os.path.isdir(maskPath_cl):
                        continue
                    for maskFile in os.listdir(maskPath_cl):
                        if fnmatch.fnmatch(maskFile, os.path.splitext(file)[0]+'_*'):
                            maskFilePaths.append(maskPath_cl + "/" + maskFile)
                            maskClasses.append(classDicts[dirDicts[dirname]])
            
            self.add_image("Liver", image_id=index, path=imageFilePath, maskPaths=maskFilePaths, maskClasses=maskClasses)
                
        
    def read_image(self, filePath):
        """Depending on the extension.
        Shape : h, w, channel
        """
        if filePath.endswith(".dcm"):
            image = sitk.ReadImage(filePath)
            image = sitk.GetArrayFromImage(image).astype("int16")
            image = np.expand_dims(image[0,:,:], -1)
        elif filePath.endswith(".png"):
            image = cv2.imread(filePath)
            image = np.array(image, dtype = "int16")
        elif filePath.endswith(".mha"):
            image = sitk.ReadImage(filePath)
            image = sitk.GetArrayFromImage(image).astype("int16")
            image = np.transpose(image,(1,2,0))
        return image
        
    
    def load_image_custom(self, image_id):
        """Generate an image from the specs of the given image ID.
        Typically this function loads the image from a file, but
        in this case it generates the image on the fly from the
        specs in image_info.
        
        + The lowest directory in 'filePath' must be phase.
        """
        
        info = self.image_info[image_id]
        filePath = info["path"]
        
        filename = os.path.basename(filePath)
        filePath = os.path.dirname(os.path.dirname(filePath))
        
        image = []
        image.append(self.read_image(filePath + "/artery/" + filename)[:,:,0]) # artery phase
        image.append(self.read_image(filePath + "/portal/" + filename)[:,:,0]) # portal-venous phase
        image.append(self.read_image(filePath + "/delay/" + filename)[:,:,0]) # delay phase
        image = np.transpose(image,(1,2,0))
        
        return image, filename
    
    
    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "shapes":
            return info["shapes"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask_custom(self, image_id, image_shape):
        """Generate instance masks for shapes of the given image ID.
            image_shape : h, w, channel
        """
        info = self.image_info[image_id]
        filePaths = info['maskPaths']
        classes = info['maskClasses']
        
        masks = []
        class_ids = []
        if(len(image_shape)==3):
            image_shape = image_shape[:2]
        
        # 1 filePath -- 1 class 
        for i, filePath in enumerate(filePaths):
            
            if filePath.endswith(".png"):
                mask = cv2.imread(filePath, 0)
                mask = np.asarray(mask, dtype = "uint8")
                
            masks.append(mask)
            class_ids.append(classes[i])
            
        if len(masks)==0 :
            masks.append(np.zeros(image_shape, dtype = "uint8"))
            class_ids.append(0)
        
        image = np.stack(masks, axis=2)
        class_ids = np.array(class_ids, dtype=np.int32)
        return image, class_ids
    
    
    def showImginfo(self):
        print(len(self.image_info))
        print(self.image_info)

        