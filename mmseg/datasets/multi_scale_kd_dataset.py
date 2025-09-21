from .builder import DATASETS
from mmseg.datasets.pipelines import Compose, Resize
from mmseg.datasets.builder import build_dataset
from mmcv.parallel import DataContainer
import torch.nn.functional as F
import torch

@DATASETS.register_module()
class MultiScaleKDDataset:
    '''Teacher augmentation -> Student resolution'''
    def __init__(self,
                 base_dataset_cfg,
                 train_pipeline,
                 test_pipeline=None,
                 student_resolution = (288, 288),
                 teacher_resolution = (384, 384),
                 test_mode = False,
                 multi_scale = True):   
                 
        # Base dataset 생성
        self.base_dataset = build_dataset(base_dataset_cfg)
        self.train_pipeline = Compose(train_pipeline)
        self.test_pipeline = Compose(test_pipeline) if test_pipeline else None
        
        self.student_resolution = student_resolution
        self.teacher_resolution = teacher_resolution
        self.test_mode = test_mode
        self.multi_scale = multi_scale
        
        #Dataset 속성
        self.CLASSES = self.base_dataset.CLASSES
        self.PALETTE = self.base_dataset.PALETTE
    
    def __len__(self):
        return len(self.base_dataset)
        
    def __getitem__(self, idx):
        base_data = self.base_dataset[idx]
        
        if self.test_mode and self.test_pipeline:
            # Test mode: Student만 사용 (원래 축소된 이미지)
            student_data = self.test_pipeline(base_data)
            return {
                'img': student_data['img'],
                'img_metas': student_data.get('img_metas', [])
            }
        else:
            # Train mode
            teacher_data = self.train_pipeline(base_data)
            
            if self.multi_scale:
                # MultiScale=True: Teacher와 Student의 resolution을 다르게 설정
                teacher_img = teacher_data['img']
                teacher_gt = teacher_data['gt_semantic_seg']
                
                # Teacher 이미지/GT에서 텐서 추출
                teacher_img_tensor = teacher_img.data if hasattr(teacher_img, 'data') else teacher_img
                teacher_gt_tensor = teacher_gt.data if hasattr(teacher_gt, 'data') else teacher_gt
                
                # 텐서 차원 확인 및 수정 (F.interpolate는 4D 텐서 필요)
                if teacher_img_tensor.dim() == 3:
                    teacher_img_tensor = teacher_img_tensor.unsqueeze(0)
                if teacher_gt_tensor.dim() == 3:
                    teacher_gt_tensor = teacher_gt_tensor.unsqueeze(0)
                
                # Student 해상도로 resize
                if self.student_resolution != self.teacher_resolution:
                    student_img_tensor = F.interpolate(
                        teacher_img_tensor,
                        size=self.student_resolution,
                        mode='bilinear',
                        align_corners=False
                    )
                    student_gt_tensor = F.interpolate(
                        teacher_gt_tensor.float(),
                        size=self.student_resolution,
                        mode='nearest'
                    ).long()
                else:
                    student_img_tensor = teacher_img_tensor
                    student_gt_tensor = teacher_gt_tensor
                
                # 차원 복원 (원래 형태로)
                if student_img_tensor.shape[0] == 1:
                    student_img_tensor = student_img_tensor.squeeze(0)
                if student_gt_tensor.shape[0] == 1:
                    student_gt_tensor = student_gt_tensor.squeeze(0)
                
                # DataContainer 형태로 복원
                student_img = DataContainer(student_img_tensor, stack=True)
                student_gt = DataContainer(student_gt_tensor, stack=True)
                
                return {
                    'teacher_img': teacher_img,
                    'student_img': student_img,
                    'teacher_gt': teacher_gt,
                    'student_gt': student_gt,
                    'img_metas': teacher_data.get('img_metas', [])
                }
            else:
                return {
                    'teacher_img': teacher_data['img'],
                    'student_img': teacher_data['img'],  # 같은 이미지
                    'teacher_gt': teacher_data['gt_semantic_seg'],
                    'student_gt': teacher_data['gt_semantic_seg'],  # 같은 GT
                    'img_metas': teacher_data.get('img_metas', [])
                }
