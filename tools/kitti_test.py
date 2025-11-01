import os
from mmseg.apis import inference_segmentor, init_segmentor
from mmseg.datasets import build_dataset
from mmseg.datasets.pipelines import Compose
import mmcv
import numpy as np
import cv2
from tqdm import tqdm

#PYTHONPATH=$(pwd):$PYTHONPATH python /home/yeonwoo3/DIFF/tools/camvid_test_2fold.py

# K-Fold 설정: [(학습fold, 평가fold, config, checkpoint), ...]
fold_configs = [
    {
        'name': 'fold1_model_on_fold2_data',
        'train_fold': 'fold1',
        'test_fold': 'fold2',
        'config_file': '/home/yeonwoo3/DIFF/work_dirs/kd/Label_Teacher/MSE/MSE_0.01_Multi_LabelTeacher_pre_student/fold1/camvid_DIFF2Seg_512t384s_mse_fold1.py',
        'checkpoint_file': '/home/yeonwoo3/DIFF/work_dirs/kd/Label_Teacher/MSE/MSE_0.01_Multi_LabelTeacher_pre_student/fold1/best_mIoU_iter_23000_student_only.pth',
        'base_img_dir': '/home/yeonwoo3/DATA/camvid/384x288_fold_p/fold2/images',
        'base_gt_dir': '/home/yeonwoo3/DATA/camvid/384x288_fold_p/fold2/ann',
        'save_dir': '/home/yeonwoo3/DIFF/work_dirs/kd/Label_Teacher/MSE/MSE_0.01_Multi_LabelTeacher_pre_student/fold1/test'  #segmentation map
    },
    {
        'name': 'fold2_model_on_fold1_data',
        'train_fold': 'fold2',
        'test_fold': 'fold1',
        'config_file': '/home/yeonwoo3/DIFF/work_dirs/kd/Label_Teacher/MSE/MSE_0.01_Multi_LabelTeacher_pre_student/fold2/camvid_DIFF2Seg_512t384s_mse_fold2.py',
        'checkpoint_file': '/home/yeonwoo3/DIFF/work_dirs/kd/Label_Teacher/MSE/MSE_0.01_Multi_LabelTeacher_pre_student/fold2/best_mIoU_iter_27000_student_only.pth',
        'base_img_dir': '/home/yeonwoo3/DATA/camvid/384x288_fold_p/fold1/images',
        'base_gt_dir': '/home/yeonwoo3/DATA/camvid/384x288_fold_p/fold1/ann',
        'save_dir': '/home/yeonwoo3/DIFF/work_dirs/kd/Label_Teacher/MSE/MSE_0.01_Multi_LabelTeacher_pre_student/fold2/test'
    }
]

def kitti_palette():
    return [
        [128, 128, 128],   # 0: sky
        [128, 0, 0],       # 1: building
        [128, 64, 128],    # 2: road
        [0, 0, 192],       # 3: sidewalk
        [64, 64, 128],     # 4: fence
        [128, 128, 0],     # 5: vegetation
        [192, 192, 128],   # 6: pole
        [64, 0, 128],      # 7 : car
        [192, 128, 128],   # 8: sign
        [64, 64, 0],       # 9: pedestrian
        [0, 128, 192]      # 10: cyclist
]



# 각 fold별 결과 저장
all_fold_metrics = []
all_fold_class_ious = []
all_fold_class_accs = []

# 각 fold 평가
for fold_config in fold_configs:
    print("\n" + "="*80)
    print(f"평가 시작: {fold_config['name']}")
    print(f"학습 fold: {fold_config['train_fold']}, 평가 fold: {fold_config['test_fold']}")
    print("="*80 + "\n")
    
    config_file = fold_config['config_file']
    checkpoint_file = fold_config['checkpoint_file']
    base_img_dir = fold_config['base_img_dir']
    base_gt_dir = fold_config['base_gt_dir']
    save_dir = fold_config['save_dir']
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 모델 및 파이프라인 초기화
    cfg = mmcv.Config.fromfile(config_file)
    print("--- test pipeline ---")
    print(cfg.data.test.pipeline)
    print("------------------------------------")
    model = init_segmentor(cfg, checkpoint_file, device='cuda:0')
    
    # 이미지 리스트 수집
    img_list = []  
    for split in ['train', 'val']:
        split_img_dir = os.path.join(base_img_dir, split)
        split_gt_dir = os.path.join(base_gt_dir, split)
        if os.path.exists(split_img_dir):
            for f in os.listdir(split_img_dir):
                if f.endswith('.png'):
                    img_path = os.path.join(split_img_dir, f)
                    gt_path = os.path.join(split_gt_dir, f)
                    img_list.append((img_path, gt_path, f))
    
    print(f"총 {len(img_list)}개")
    
    # 직접 테스트 파이프라인 생성
    cfg.model.test_cfg.mode = 'whole'
    
    # 예측 및 저장
    results = []
    
    for img_path, gt_path, fname in tqdm(img_list, desc=f"{fold_config['name']} 예측 중"):
        try:
            result = inference_segmentor(model, img_path)
            if isinstance(result, list) and len(result) > 0:
                results.append(result[0])
            else:
                results.append(None)
            
            # 시각화 저장
            if isinstance(result, list) and len(result) > 0:
                pred_mask = np.array(result[0], dtype=np.uint8)
                palette = np.array(kitti_palette(), dtype=np.uint8)
                if pred_mask.max() >= len(palette):
                    pred_mask = np.clip(pred_mask, 0, len(palette)-1)
                color_mask = palette[pred_mask]
                save_path = os.path.join(save_dir, fname)
                cv2.imwrite(save_path, cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR))
        except Exception as e:
            print(f"Error processing {fname}: {e}")
            results.append(None)
    
    # 평가
    if os.path.exists(base_gt_dir):
        from mmseg.core.evaluation import eval_metrics
        preds = []
        gts = []
        
        for i, (img_path, gt_path, fname) in enumerate(tqdm(img_list, desc=f"{fold_config['name']} 평가 중")):
            if not os.path.exists(gt_path):
                continue
            gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
            if gt is None:
                continue
            pred = results[i]
            if pred is None:
                continue
            
            if len(pred.shape) > 2:
                pred = pred[0]
            if len(gt.shape) > 2:
                continue
            
            preds.append(pred)
            gts.append(gt)
        
        if preds and gts:
            if not hasattr(model, 'CLASSES') or model.CLASSES is None:
                num_classes = len(set([v for arr in gts for v in np.unique(arr)]))
            else:
                num_classes = len(model.CLASSES)
            
            metrics = eval_metrics(preds, gts, num_classes=num_classes, ignore_index=255)
            
            print(f"\n[{fold_config['name']}] 평가 결과:")
            print(f"총 평가 이미지: {len(preds)}개")
            
            # 결과 저장
            fold_result = {
                'name': fold_config['name'],
                'num_images': len(preds)
            }
            
            if 'mIoU' in metrics:
                fold_result['mIoU'] = metrics['mIoU']
                print(f"mIoU: {metrics['mIoU']*100:.2f}")
            elif 'IoU' in metrics:
                miou = float(np.nanmean(metrics['IoU']))
                fold_result['mIoU'] = miou
                print(f"mIoU (mean IoU): {miou*100:.2f}")
            
            if 'mAcc' in metrics:
                fold_result['mAcc'] = metrics['mAcc']
                print(f"mPA (mean Pixel Accuracy): {metrics['mAcc']*100:.2f}")
            
            if 'aAcc' in metrics:
                fold_result['aAcc'] = metrics['aAcc']
                print(f"aAcc: {metrics['aAcc']*100:.2f}")
            
            all_fold_metrics.append(fold_result)
            
            # 클래스별 IoU, Acc 저장
            if 'IoU' in metrics and 'Acc' in metrics:
                all_fold_class_ious.append(metrics['IoU'])
                all_fold_class_accs.append(metrics['Acc'])
                
                if hasattr(model, 'CLASSES') and model.CLASSES is not None:
                    print("\n클래스별 IoU:")
                    for i, (cls_iou, cls_acc) in enumerate(zip(metrics['IoU'], metrics['Acc'])):
                        if i < len(model.CLASSES):
                            print(f"{model.CLASSES[i]}: IoU={cls_iou*100:.2f}, Acc={cls_acc*100:.2f}")
        else:
            print(f"[{fold_config['name']}] 오류: 평가할 예측/GT 쌍이 없습니다.")

# K-Fold 최종 평균 결과 출력
print("\n" + "="*80)
print("K-FOLD 교차 검증 최종 결과")
print("="*80 + "\n")

if all_fold_metrics:
    # mIoU 평균
    miou_values = [m['mIoU'] for m in all_fold_metrics if 'mIoU' in m]
    if miou_values:
        avg_miou = np.mean(miou_values)
        print(f"평균 mIoU: {avg_miou*100:.2f}%")
        print(f"  - fold1 모델 (fold2 평가): {miou_values[0]*100:.2f}")
        print(f"  - fold2 모델 (fold1 평가): {miou_values[1]*100:.2f}")
    
    # mAcc 평균
    macc_values = [m['mAcc'] for m in all_fold_metrics if 'mAcc' in m]
    if macc_values:
        avg_macc = np.mean(macc_values)
        print(f"\n평균 mPA: {avg_macc*100:.2f}%")
        print(f"  - fold1 모델 (fold2 평가): {macc_values[0]*100:.2f}")
        print(f"  - fold2 모델 (fold1 평가): {macc_values[1]*100:.2f}")
    
    # aAcc 평균
    aacc_values = [m['aAcc'] for m in all_fold_metrics if 'aAcc' in m]
    if aacc_values:
        avg_aacc = np.mean(aacc_values)
        print(f"\n평균 aAcc: {avg_aacc*100:.2f}%")
        print(f"  - fold1 모델 (fold2 평가): {aacc_values[0]*100:.2f}")
        print(f"  - fold2 모델 (fold1 평가): {aacc_values[1]*100:.2f}")
    
    # 클래스별 평균 IoU
    if all_fold_class_ious and hasattr(model, 'CLASSES') and model.CLASSES is not None:
        avg_class_ious = np.mean(all_fold_class_ious, axis=0)
        avg_class_accs = np.mean(all_fold_class_accs, axis=0)
        print("\n클래스별 평균 IoU (K-Fold):")
        for i, (cls_iou, cls_acc) in enumerate(zip(avg_class_ious, avg_class_accs)):
            if i < len(model.CLASSES):
                print(f"{model.CLASSES[i]}: IoU={cls_iou*100:.2f}, Acc={cls_acc*100:.2f}")

print("\n" + "="*80)
print("평가 완료!")
print("="*80)