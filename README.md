# MARD-Net: Multi-Scale Attentive Residual Dense Network for Single Image Rain Removal

[Paper Download](https://openaccess.thecvf.com/content/ACCV2020/html/Chen_Multi-scale_Attentive_Residual_Dense_Network_for_Single_Image_Rain_Removal_ACCV_2020_paper.html)

[Xiang Chen](https://cxtalk.github.io/), [Yufeng Huang](https://dzx.sau.edu.cn/info/1031/1169.htm) and Lei Xu

This work has been accepted by ACCV 2020. 

## Abstract
Single image deraining is an urgent yet challenging task since rain streaks severely degrade the image quality and hamper the practical application. The investigation on rain removal has thus been attracting, while the performances of existing deraining have limitations owing to over smoothing effect, poor generalization capability and rain intensity varies both in spatial locations and color channels. To address these issues, we proposed a Multi-scale Attentive Residual Dense Network called MARD-Net in end-to-end manner, to exactly extract the negative rain streaks from rainy images while precisely preserving the image details. The architecture of modified dense network can be used to exploit the rain streaks details representation through feature reuse and propagation. Further, the Multi-scale Attentive Residual Block (MARB) is involved in the dense network to guide the rain streaks feature extraction and representation capability. Since contextual information is very critical for deraining, MARB first uses different convolution kernels along with fusion to extract multi-scale rain features and employs feature attention module to identify rain streaks regions and color channels, as well as has the skip connections to aggregate features at multiple levels and accelerate convergence. The proposed method is extensively evaluated on several frequent-use synthetic and real-world datasets. The quantitative and qualitative results show that the designed framework performs better than the recent state of-the-art deraining approaches on promoting the rain removal performance and preserving image details under various rain streaks cases.

## Requirements
- CUDA 9.0
- Python 3.7
- Pytorch 1.4.0
- Torchvision 0.2.0

## Dataset
Please download the following datasets:

* Rain100L [[dataset](http://www.icst.pku.edu.cn/struct/Projects/joint_rain_removal.html)]
* Rain100H [[dataset](http://www.icst.pku.edu.cn/struct/Projects/joint_rain_removal.html)]
* Rain1400 [[dataset](https://xueyangfu.github.io/projects/cvpr2017.html)]
* Rain12 [[dataset](http://yu-li.github.io/paper/li_cvpr16_rain.zip)]

## Setup
```
$ git clone https://github.com/cxtalk/MARD-Net.git
$ cd config
```  

## Training
```
$ python train.py
```

## Testing
```
$ python test.py
``` 


## Citation
```
@inproceedings{Chen_2020_ACCV,
	author    = {Xiang, Chen and Yufeng, Huang and Lei, Xu},
	title     = {Multi-Scale Attentive Residual Dense Network for Single Image Rain Removal},
	booktitle = {Proceedings of the Asian Conference on Computer Vision (ACCV)},
	month     = {November},
	year      = {2020},
}
```

## Contact

If you are interested in our work or have any questions, please directly contact my github.
Email: cx@lowlevelcv.com
