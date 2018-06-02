This is the TensorFlow implementation of the saliency project "Saliency-based deep convolutional neural network for no-reference image quality assessment".  

This implementation is based on the LIVE dataset, that is the label is DMOS [0,99], the lower the better.

Pretrained model: [link](https://drive.google.com/file/d/1UnQYLgEgVSMv6lqWhAseUWgg3CEc-zMy/view?usp=sharing).

Use the predict.py file to load the pretrained model and predict the quality of an input image.
The input image should be locally normalized (see the paper for more information).
