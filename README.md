# Semantic-Segmentation
Classified &amp; clustered every pixel in cityscape dataset image (gtFine labels) belonging to same object class in modified Google DeepLabV3 architecture. Enhanced boundary detection of the Image Net (ResNet) with atrous conv &amp; Atrous Spatial Pyramid Pooling (ASPP) with variable dilation rates by using end to end Holistic Edge Detection(HED) technique. Added Squeeze &amp; Excitation (SE) block to improve channel attention &amp; also tested softmax in stead of sigmoid for last layer of SE.

![sem-seg.gif](https://github.com/SatishKumarAnbalagan/Semantic-Segmentation/blob/master/results/sem_seg.gif)

File Description:

preprocessHED.py: pretrained HED for preprocessing inputs

[train/test]_vanilla.py: train DeepLabsV3 

[train/test]_DLSE.py: train DeepLabsV3 with Squeeze and Excitation Networks

[train/test]_DLSE_SF.py: train DeepLabsV3 with Squeeze and Excitation Networks using Softmax

[train/test]_DLSE_HED.py: train DeepLabsV3 with Squeeze and Excitation Networks with Hollistic Edge Detection Preprocessing

bgenerator.py: batch generator used in running models

.h5: weights for corresponding models

train/val.txt: text file used in the batch generator 

**If running DLSE_HED make sure to preprocess inputs first

Running Preprocessing:
python preprocessHED.py --model bsds500 --in ./tobeprocesseddirectory --out ./processeddirectory


Running Models:
Before running models a .txt file needs to be generated for the batch generator with format:
image1.png, image1label.png, image1hedprocessed.png
image2.png, image2label.png, image2hedprocessed.png
.
.
.
*if not using HED omit last column and make commented changes in bgenerator.py

Files with "train" or "test" in the beginning are models, run without any input arguments 
ex: python train_DLSE.py


