# Generalizable Face Landmarking Guided by Conditional Face Warping (CVPR 2024)
This is the official repository for the following paper:

>**Generalizable Face Landmarking Guided by Conditional Face Warping** [[paper]](https://openreview.net/pdf?id=wB2R7QQncw) [[arxiv]](https://arxiv.org/abs/2404.12322) [[project page]](https://plustwo0.github.io/project-face-landmarker/)<br>
 <br>Jiayi Liang*, Haotian Liu*, Hongteng Xu, Dixin Luo<br>
 Accepted by CVPR 2024.


![Scheme](/assets/scheme.png "Learning Scheme")

# Install

```commandline
pip install -r requirements.txt
```

# Model
Our proposed framework mainly contains two parts: **face warper** and **landmark detector**. 
They are trained in an alternative optimization framework. 
- The face warper aims to deform real human faces according to stylized facial images, generating warped faces and corresponding warping fields. 
- The face landmarker, as the key module of the warper, predicts the ending points of the warping fields and thus provides us with pseudo landmarks for the stylized facial images. 

In our implmentation, we employ SLPT as our backbone and locate the model in ```Landmark2``` folder. 
For the reproduction on other detectors, substitute the ```Landmark2``` folder with target model and make modifications in ```train.py```.


# Data Preparation

## Source Domain
Download images and annotations of 300-W from [ibug](https://ibug.doc.ic.ac.uk/resources/300-W/).

We select frontal faces from the trainset of 300W as our training data, and list of image path ```300W_frontal_train_list.txt``` can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1LPYKxb2e-7a7Ovy2tPBPKad74HxxGZRG?usp=sharing).

Your directory should be like:
   ```
     Dataset
     │
     └──300W
        │
        └───300W_frontal_train_list.txt
        └───frontal_train
            └───261068_1.jpg
            │
            └───...
        └───frontal_train_label
            └───261068_1.jpg.npy
            │
            └───...
        └───test_list.txt
        └───test_list_common.txt
        └───test_list_challenge.txt
        └───lfpw
            └───trainset
            └───testset
                └───image_0001.png
                └───image_0001.pts
                │
                └───...
        └───helen
            │
            └───...
        └───ibug
            │
            └───...

   ```

## Target Domain
- Download CariFace according to [CariFace Dataset](https://github.com/Juyong/CaricatureFace).
The split of training and testing set can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1_oQhYCOJastWGhy6tFDUEivgj5GMLUCe?usp=sharing).

- Other domains like Artistic-Faces can be retrieved from [Link](https://github.com/papulke/face-of-art). 
The split of training and testing set can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1XN8_DfrbT7rfWbxx0oHpzqXOpGPCWcn7?usp=sharing).
> Please note that ArtiFace contains a total of 160 images. Under the GZSL (Unseen ArtiFace) scenario, the test set size of ArtiFace is 160 images; under the DA (300W->ArtiFace) and GZSL (Unseen CariFace) scenarios, the test set of ArtiFace only contains 32 images. <br> Therefore, please modify the specified test set list in ```Artistic.py``` to ```test_list.txt``` (32 images) or ```test_list_all.txt``` (160 images) according to the different circumstances.

Your directory should be like:
   ```
     Dataset
     │
     └──CariFace_dataset
        │
        └───images
            └───00005.jpg
            │
            └───...
        └───landmarks
            └───00005.jpg.npy
            │
            └───...
        └───train_list.txt
        └───test_list.txt
     │
     └──AF_dataset
        │
        └───images
            └───0.png
            │
            └───...
        └───landmarks
            └───0.png.npy
            │
            └───...
        └───train_list.txt
        └───test_list.txt
        └───test_list_all.txt
   ```

# Train

## Load Pretrained Model
Download source-pretrained weights ```model_best.pth``` from [Google Drive](https://drive.google.com/drive/folders/1a-9oT2GB-IthCeJbnoLbn1kp7uJyzkMi?usp=sharing) and move it into folder ```Landmark2```.


## Training Begin!
 
```python
python train.py --src_data path/to/source/data --tgt_data path/to/target/data --pretrain_path path/to/pretrained/checkpoint
```


# Inference
Download our [model](https://drive.google.com/drive/folders/1a-9oT2GB-IthCeJbnoLbn1kp7uJyzkMi?usp=sharing) and test on the CariFace by running:
```python
python test.py --checkpoint path/to/model/weights
```

Further, to test on ArtiFace, download [checkpoint](https://drive.google.com/drive/folders/1a-9oT2GB-IthCeJbnoLbn1kp7uJyzkMi?usp=sharing) and inference:
```python
python test_Artistic.py --checkpoint path/to/model/weights
```

# Citation
If our work is helpful for your research, please cite our paper:


# Aknowledgement
