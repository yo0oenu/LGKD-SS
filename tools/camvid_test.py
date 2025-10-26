import os
from mmseg.apis import inference_segmentor, init_segmentor
from mmseg.datasets import build_dataset
from mmseg.datasets.pipelines import Compose
import mmcv
import numpy as np
import cv2
from tqdm import tqdm

#PYTHONPATH=$(pwd):$PYTHONPATH python /home/yeonwoo3/DIFF/tools/camvid_test.py

# 사용자 설정
config_file = '/home/yeonwoo3/DIFF/work_dirs/kd/sim_pre_0.05_Multi_LabelTeacher/fold2/camvid_DIFF2Seg_512t384s_gram_fold2.py' 
checkpoint_file = '/home/yeonwoo3/DIFF/work_dirs/kd/sim_pre_0.05_Multi_LabelTeacher/fold2/best_mIoU_iter_30000.pth'
img_dir = '/home/yeonwoo3/DATA/camvid/384x288_fold_p/fold2/images/val'  #
gt_dir = '/home/yeonwoo3/DATA/camvid/384x288_fold_p/fold2/ann/val'  #
save_dir = '/home/yeonwoo3/DIFF/work_dirs/kd/sim_pre_0.05_Multi_LabelTeacher/fold2/test'

os.makedirs(save_dir, exist_ok=True)

# 모델 및 파이프라인 초기화
cfg = mmcv.Config.fromfile(config_file)
print("--- DEBUG: 실제 적용되는 test pipeline ---")
print(cfg.data.test.pipeline)
print("------------------------------------")
model = init_segmentor(cfg, checkpoint_file, device='cuda:0')

# 이미지 리스트: .png 파일만 사용
img_list = [f for f in os.listdir(img_dir) if f.endswith('.png')]

# 직접 테스트 파이프라인 생성 (scale_factor 문제 해결)
cfg.model.test_cfg.mode = 'whole'  # 테스트 모드 설정

# 예측 및 저장
results = []  # 결과 저장 리스트

#CamVid
#def camvid_palette():
#    return [
#        [128, 128, 128],  # 0: sky
#        [128, 0, 0],      # 1: building
#        [192, 192, 128],  # 2: pole
#        [128, 64, 128],   # 3: road
#        [0, 0, 192],      # 4: sidewalk
#        [128, 128, 0],    # 5: tree
#        [192, 128, 128],  # 6: signsymbol
#        [64, 64, 128],    # 7: fence
#        [64, 0, 128],     # 8: car
#        [64, 64, 0],      # 9: pedestrian
#        [0, 128, 192]     # 10: bicyclist
#    ]


def camvid_palette():
    return [
        [128, 0, 0],   #0: Building
        [0, 128, 192], #1: Bicyclist
        [64, 64, 128], #2: Fence
        [192, 192, 128], #Pole
        [64, 64, 0],     #Pedestrian
        [128, 64, 128],  #Road
        [128, 128, 128],  #Sky
        [192, 128, 128],  #SignSymbol
        [128, 128, 0],    #Tree
        [0, 0, 192],      #Sidewalk
        [64, 0, 128]      # 10: Car
    ]


for fname in tqdm(img_list):
    img_path = os.path.join(img_dir, fname)
    try:
        # inference_segmentor 사용 (scale_factor 문제 해결)
        result = inference_segmentor(model, img_path)
        # 디버깅: 예측 결과 타입, 길이, shape 출력
        if isinstance(result, list) and len(result) > 0:
            print(f"{fname} 예측 결과 타입: {type(result)}, 길이: {len(result)}, shape: {np.array(result[0]).shape}")
            results.append(result[0])  # 결과 저장
        else:
            print(f"{fname} 예측 결과 없음 또는 비정상: {result}")
            results.append(None)
        # 시각화 저장 (camvid_palette 사용)
        if isinstance(result, list) and len(result) > 0:
            pred_mask = np.array(result[0], dtype=np.uint8)
            palette = np.array(camvid_palette(), dtype=np.uint8)
            if pred_mask.max() >= len(palette):
                print(f"Warning: 예측 클래스 인덱스가 PALETTE 범위를 벗어남! max={pred_mask.max()}, palette_len={len(palette)})")
                pred_mask = np.clip(pred_mask, 0, len(palette)-1)
            color_mask = palette[pred_mask]
            save_path = os.path.join(save_dir, fname)
            cv2.imwrite(save_path, cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR))
    except Exception as e:
        print(f"Error processing {fname}: {e}")
        results.append(None)

#print(f"시각화 결과가 {save_dir}에 저장되었습니다.")

# (선택) mIoU 등 평가 직접 계산 예시
if os.path.isdir(gt_dir):
    from mmseg.core.evaluation import eval_metrics
    preds = []
    gts = []
    print(f"\n이미지 폴더: {img_dir}, 파일 수: {len(img_list)}")
    print(f"GT 폴더: {gt_dir}, 파일 수: {len(os.listdir(gt_dir))}")
    if len(img_list) > 0:
        sample_img = os.path.join(img_dir, img_list[0])
        # GT 파일명: 확장자 제거 후 _L.png로 변환
        base_name = os.path.splitext(img_list[0])[0]
        gt_fname = base_name + '_L.png'
        sample_gt = os.path.join(gt_dir, gt_fname)
        print(f"샘플 이미지 경로: {sample_img}, 존재: {os.path.exists(sample_img)}")
        print(f"샘플 GT 경로: {sample_gt}, 존재: {os.path.exists(sample_gt)}")
    for i, fname in enumerate(tqdm(img_list)):
        # GT 파일명 매칭: 확장자 제거 후 _L.png로 변환
        base_name = os.path.splitext(fname)[0]
        gt_fname = base_name + '_L.png'
        gt_path = os.path.join(gt_dir, gt_fname)
        if not os.path.exists(gt_path):
            print(f"GT 파일 없음: {gt_path}")
            continue
        gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
        if gt is None:
            print(f"GT 파일 로드 실패: {gt_path}")
            continue
        pred = results[i]
        if pred is None:
            print(f"예측 결과 없음: {fname}")
            continue
        # 디버깅: shape, dtype, min/max 값 출력
        print(f"{fname} pred shape: {np.array(pred).shape}, dtype: {np.array(pred).dtype}, min: {np.array(pred).min()}, max: {np.array(pred).max()}")
        print(f"{fname} gt shape: {gt.shape}, dtype: {gt.dtype}, min: {gt.min()}, max: {gt.max()}")
        if len(pred.shape) > 2:
            print(f"[경고] pred shape이 2D가 아님: {pred.shape}, 첫번째 채널만 사용")
            pred = pred[0]
        if len(gt.shape) > 2:
            print(f"GT {gt_path}가 단일 채널이 아닙니다! {gt.shape}")
            continue
        print(f"[DEBUG] preds에 추가: pred shape={pred.shape}, dtype={pred.dtype}; gt shape={gt.shape}, dtype={gt.dtype}")
        preds.append(pred)
        gts.append(gt)
    if preds and gts:
        print(f"preds 개수: {len(preds)}, gts 개수: {len(gts)}")
        if preds:
            print(f"[DEBUG] preds[0] shape: {np.array(preds[0]).shape}, dtype: {np.array(preds[0]).dtype}")
        if gts:
            print(f"[DEBUG] gts[0] shape: {np.array(gts[0]).shape}, dtype: {np.array(gts[0]).dtype}")
        if not hasattr(model, 'CLASSES') or model.CLASSES is None:
            print("[경고] model.CLASSES가 정의되어 있지 않습니다. config나 모델 정의를 확인하세요.")
            num_classes = len(set([v for arr in gts for v in np.unique(arr)]))
            print(f"GT에서 추정한 클래스 개수: {num_classes}")
        else:
            num_classes = len(model.CLASSES)
        print(f"num_classes: {num_classes}")
        metrics = eval_metrics(preds, gts, num_classes=num_classes, ignore_index=255)
        print("\n평가 결과:")
        print(metrics)
        if 'mIoU' in metrics:
            print(f"mIoU: {metrics['mIoU']}")
        elif 'IoU' in metrics:
            miou = float(np.nanmean(metrics['IoU']))
            print(f"mIoU (mean IoU): {miou}")
        if 'mAcc' in metrics:
            print(f"mPA (mean Pixel Accuracy): {metrics['mAcc']}")
        for key in ['mean_iou', 'Mean IoU', 'iou']:
            if key in metrics:
                print(f"{key}: {metrics[key]}")
        for key in ['mean_acc', 'Mean Acc', 'acc']:
            if key in metrics:
                print(f"{key}: {metrics[key]}")
        if 'aAcc' in metrics:
            print("aAcc:", metrics['aAcc'])
        if 'IoU' in metrics and 'Acc' in metrics and hasattr(model, 'CLASSES') and model.CLASSES is not None:
            print("\n클래스별 IoU:")
            for i, (cls_iou, cls_acc) in enumerate(zip(metrics['IoU'], metrics['Acc'])):
                if i < len(model.CLASSES):
                    print(f"{model.CLASSES[i]}: IoU={cls_iou:.4f}, Acc={cls_acc:.4f}")
    else:
        print("[오류] 평가할 예측/GT 쌍이 없습니다. preds/gts가 비어 있습니다. 파일 경로, GT 매칭, 예측 결과를 확인하세요.")
        print(f"preds 개수: {len(preds)}, gts 개수: {len(gts)}")