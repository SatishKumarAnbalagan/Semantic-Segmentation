# Semantic-Segmentation
Classified &amp; clustered every pixel in cityscape dataset image (gtFine labels) belonging to same object class in modified Google DeepLabV3 architecture. Enhanced boundary detection of the Image Net (ResNet) with atrous conv &amp; Atrous Spatial Pyramid Pooling (ASPP) with variable dilation rates by using end to end Holistic Edge Detection(HED) technique. Added Squeeze &amp; Excitation (SE) block to improve channel attention &amp; also tested softmax in stead of sigmoid for last layer of SE.

![sem-seg.gif](https://github.com/SatishKumarAnbalagan/Semantic-Segmentation/blob/master/results/sem_seg.gif)

DeepLabV3 encoder-decoder architecture

![DeepLabV3_en-de_arch.jpg](https://github.com/SatishKumarAnbalagan/Semantic-Segmentation/blob/master/results/DeepLabV3_en-de_arch.jpg)

Atrous Spatial Pyramid Pooling (ASPP) 

![ASPP.jpg](https://github.com/SatishKumarAnbalagan/Semantic-Segmentation/blob/master/results/ASPP.jpg)

Squeeze and Excitation Blocks

![SE.jpg](https://github.com/SatishKumarAnbalagan/Semantic-Segmentation/blob/master/results/SE.jpg)

Holistic Edge Detection Preprocessing

![HED1.png](https://github.com/SatishKumarAnbalagan/Semantic-Segmentation/blob/master/results/HED1.png)
![comparison.gif](https://github.com/SatishKumarAnbalagan/Semantic-Segmentation/blob/master/results/comparison.gif)

File Description:

preprocessHED.py: pretrained HED for preprocessing inputs

![aachen_000001_000019_leftImg8bit_boundary.png](https://github.com/SatishKumarAnbalagan/Semantic-Segmentation/blob/master/results/aachen_000001_000019_leftImg8bit_boundary.png)

[train/test]_vanilla.py: train DeepLabsV3

Traget :

![V1.jpg](https://github.com/SatishKumarAnbalagan/Semantic-Segmentation/blob/master/results/V1.jpg)

Predicted :

![V2.jpg](https://github.com/SatishKumarAnbalagan/Semantic-Segmentation/blob/master/results/V2.jpg)

[train/test]_DLSE.py: train DeepLabsV3 with Squeeze and Excitation Networks

Traget :

![DLSE-T.png](https://github.com/SatishKumarAnbalagan/Semantic-Segmentation/blob/master/results/DLSE-T.png)

Predicted :

![DLSE-P.png](https://github.com/SatishKumarAnbalagan/Semantic-Segmentation/blob/master/results/DLSE-P.png)

[train/test]_DLSE_SF.py: train DeepLabsV3 with Squeeze and Excitation Networks using Softmax

Traget :

![DLSE-SF-T.png](https://github.com/SatishKumarAnbalagan/Semantic-Segmentation/blob/master/results/DLSE-SF-T.png)

Predicted :

![DLSE-SF-P.png](https://github.com/SatishKumarAnbalagan/Semantic-Segmentation/blob/master/results/DLSE-SF-P.png)

[train/test]_DLSE_HED.py: train DeepLabsV3 with Squeeze and Excitation Networks with Hollistic Edge Detection Preprocessing

Traget :

![DLSE-HED-T.png](https://github.com/SatishKumarAnbalagan/Semantic-Segmentation/blob/master/results/DLSE-HED-T.png)

Predicted :

![DLSE-HED-P.png](https://github.com/SatishKumarAnbalagan/Semantic-Segmentation/blob/master/results/DLSE-HED-P.png)

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


