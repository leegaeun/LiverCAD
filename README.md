# LiverCAD
Source codes for detecting hepatic malignancy in multiphase CT scans. (ver.1)<br/>
Automatic detection of hepatic malignancy with trained weights in multiphase CT scan.<br/>


## Environments
Ubuntu 18.04 LTS, Python 3.6.7, Keras 2.1.6, TensorFlow 1.9, CUDA 9.0, cuDNN 7.0 <br/>
<br/>
<br/>

## Trained weights
[Mask-RCNN](https://arxiv.org/pdf/1703.06870.pdf) was used to detect hepatic malignancy. Pre-trained weights with [MS COCO](https://ttic.uchicago.edu/~mmaire/papers/pdf/coco_eccv2014.pdf) were fine-tuned using multiphase CT scans including malignancies. <br/>
[*weight/MaskRCNN_20190902_230139.hdf5*](./weight)<br/>
<br/>
<br/>

## Hepatic Malignancy Detection
Codes for detecting hepatic malignancy with the trained weights are implemented in [*Run_test.ipynb*](./Run_test.ipynb).<br/>
<br/>
<p>
<img src="https://user-images.githubusercontent.com/17020746/68740110-a92fd280-062d-11ea-813f-2659629ae564.png" width="30%">    <img src="https://user-images.githubusercontent.com/17020746/68740198-e09e7f00-062d-11ea-9320-e14bfa929da0.png" width="30%">    <img src="https://user-images.githubusercontent.com/17020746/68740192-de3c2500-062d-11ea-97f3-a9d7d7aca680.png" width="30%">
</p>
When the above multiphase CT scan (arterial-, portal venous-, delayed phase from left) is input,<br/>
malignancy is detected as shown below. White is ground-truth and Red is the predicted detection box.<br/>
<br/>

<p>
<img src="https://user-images.githubusercontent.com/17020746/68740581-bc8f6d80-062e-11ea-97dd-16685ad219bf.png" width="50%">
</p>

<br/>


There is a learned weight of the conversion from B10f to B70f in the *weight* directory,<br/>
In the *data/noncontrast* directory, there are test2 example image in addition to test1. You can convert it yourself.<br/>
<br/>
