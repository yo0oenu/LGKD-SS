# IMRL_Project

## Setup Environment  [참고](https://github.com/Yux1angJi/DIFF)

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
Teacher: 1280x384 [download]()
Student: 640x192  [download]()

## Prompt
1. Label: 클래스 라벨을 그대로 프롬프트로 사용
2. Sentence: 해당 클래스에 대해 설명하는 문장 단위 프롬프트
   제작예시: All classes in an urban driving scene are "Building", "Bicyclist", "Fence"...... Tell me about the general sentence that describe each classes.
4. Bag of words: 해당 클래스에 대해 설명하는 일반적인 단어 표현 4개와 라벨명 1개
   제작예시: All classes in an urban driving scene are "Building", "Bicyclist", "Fence"...... Tell me about the 5 general words that describe each class.
5. Orthogonal concep:각 클래스에 대한 교유의 단어 표현 4개와 라벨명 1개
   제작예시: All classes in an urban driving scene are "Building", "Bicyclist", "Fence"...... Tell me about the 5 closest concept words that describe each class, which must be orthogonala to other classes.
## Training Config
### [Teacher](configs/Teacher) 
- Camvid: input resolution [384, 384]
- Kitti: input resolution [384, 384]
- [Teacher backbone config](/home/yeonwoo3/DIFF/mmseg/models/backbones/diff/configs/diff_config.yaml)
   [only Image] do_mask_step: False 
### [Student](configs/Student) 
- KD 없이 student 학습
- Camvid: input resolution [288, 288]
- Kitti: input resolution [192, 192]
- ImageNet Pre-trained 가중치 [download](https://drive.google.com/uc?export=download&id=1xPxDrs1k0ywO7Sz-H2A1pgtTGAWlFZkt)
- "pretrained" 이름의 폴더를 만든 이후, 해당 폴더 안에 가중치를 넣어놓으시면 됩니다.
### [KD](configs/KD) 
- [Camvid] Teacher input size: 384x384 / Student input size: 192x192
- [Kitti] Teacher input size: 384x384 / Student input size: 192x192
- response 기반 KD Loss는 MSE와 KL-Divergence 두 개가 구현되어 있습니다.
  [코드](mmseg/models/segmentors/encoder_decoder.py) 448번 줄에서 kd_mse_loss(mse)를 사용할 지, kd_kl_loss(kl-divergence)를 사용할 것인지 하드코딩 해야됩니다..

## Training
experiments.sh에 config 경로를 넣은 후, 아래 코드 실행
```shell
bash experiments.sh 
```






