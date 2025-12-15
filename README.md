# IMRL_Project

## Setup Environment  ([Reference](https://github.com/Yux1angJi/DIFF))

For this project, we used python 3.8.18. We recommend setting up a new virtual
environment:

```shell
python -m venv ~/venv/diff
source ~/venv/diff/bin/activate
```

In that environment, the requirements can be installed with:

```shell
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.3.7  # requires the other packages to be installed first
```

## Dataset
**Camvid** 
Teacher: 512x384  [download](https://drive.google.com/uc?export=download&id=1_WGz38Zd2va4UrFnjiF1TbO96PkawQPI)
Student: 384x288  [download](https://drive.google.com/uc?export=download&id=1bxPicvecOYFyAgmDWadz-3-RZOp6A33_)

**KITTI:** 
[download](https://drive.google.com/uc?export=download&id=1z19HfHcolqCrMrxMWWMmp9nQLJLpIQCD)

## Prompt
1. Label: Use the class label itself directly as the prompt.
2. Sentence: Use a sentence-level prompt that describes each class.
   (Example): All classes in an urban driving scene are "Building", "Bicyclist", "Fence"...... Tell me about the general sentence that describe each classes.
4. Bag of words: Use 4 general descriptive words and 1 label name for each class.
   (Example): All classes in an urban driving scene are "Building", "Bicyclist", "Fence"...... Tell me about the 5 general words that describe each class.
5. Orthogonal concep:Use 4 unique concept words and 1 label name for each class. The concept words must be orthogonal (distinct) from those of other classes.
   (Example): All classes in an urban driving scene are "Building", "Bicyclist", "Fence"...... Tell me about the 5 closest concept words that describe each class, which must be orthogonala to other classes.
## Training Config
### [Teacher](configs/Teacher) 
- Camvid: input resolution [384, 384]
- Kitti: input resolution [384, 384]
- [Teacher backbone config](/home/yeonwoo3/DIFF/mmseg/models/backbones/diff/configs/diff_config.yaml)
   if use only Image, ->  do_mask_step: False 
### [Student](configs/Student) 
- Train the student model without knowledge distillation (KD)
- Camvid: input resolution [288, 288]
- Kitti: input resolution [192, 192]
- ImageNet Pre-trained weights [download](https://drive.google.com/uc?export=download&id=1Ociq6VZ9MECrCe-hld7C8dh2XgdzsV8y)
   Once you download the weights, create a folder named pretrained and place the weights inside it.
### [KD](configs/KD) 
- [Camvid] Teacher input size: 384x384 / Student input size: 192x192
- [Kitti] Teacher input size: 384x384 / Student input size: 192x192
- kd_type can be one of the following: = kl, mse, at, sp, gram

## Training
After specifying the configuration paths in experiments.sh, run:
```shell
bash experiments.sh 
```

