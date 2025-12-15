import os
from mmseg.apis import inference_segmentor, init_segmentor
from mmseg.datasets import build_dataset
from mmseg.datasets.pipelines import Compose
import mmcv
import numpy as np
import cv2
from tqdm import tqdm



fold_configs = [
    {
        'name': 'fold1_model_on_fold2_data',
        'train_fold': 'fold1',
        'test_fold': 'fold2',
        'config_file': '',
        'checkpoint_file': '',
        'base_img_dir': '',
        'base_gt_dir': '',
        'save_dir': ''  #segmentation map
    },
    {
        'name': 'fold2_model_on_fold1_data',
        'train_fold': 'fold2',
        'test_fold': 'fold1',
        'config_file': '',
        'checkpoint_file': '',
        'base_img_dir': '',
        'base_gt_dir': '',
        'save_dir': ''
    }
]

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


all_fold_metrics = []
all_fold_class_ious = []
all_fold_class_accs = []


for fold_config in fold_configs:
    print("\n" + "="*80)
    print(f"eval start: {fold_config['name']}")
    print(f"training fold: {fold_config['train_fold']}, eval fold: {fold_config['test_fold']}")
    print("="*80 + "\n")
    
    config_file = fold_config['config_file']
    checkpoint_file = fold_config['checkpoint_file']
    base_img_dir = fold_config['base_img_dir']
    base_gt_dir = fold_config['base_gt_dir']
    save_dir = fold_config['save_dir']
    
    os.makedirs(save_dir, exist_ok=True)
    

    cfg = mmcv.Config.fromfile(config_file)
    print("--- test pipeline ---")
    print(cfg.data.test.pipeline)
    print("------------------------------------")
    model = init_segmentor(cfg, checkpoint_file, device='cuda:0')
    

    img_list = []  
    for split in ['train', 'val']:
        split_img_dir = os.path.join(base_img_dir, split)
        split_gt_dir = os.path.join(base_gt_dir, split)
        if os.path.exists(split_img_dir):
            for f in os.listdir(split_img_dir):
                if f.endswith('.png'):
                    img_path = os.path.join(split_img_dir, f)
                    base_name = os.path.splitext(f)[0]
                    gt_fname = base_name + '_L.png'
                    gt_path = os.path.join(split_gt_dir, gt_fname)
                    img_list.append((img_path, gt_path, f))
    
    print(f"총 {len(img_list)}개")
    

    cfg.model.test_cfg.mode = 'whole'
    
    
    results = []
    
    for img_path, gt_path, fname in tqdm(img_list, desc=f"{fold_config['name']} predicting"):
        try:
            result = inference_segmentor(model, img_path)
            if isinstance(result, list) and len(result) > 0:
                results.append(result[0])
            else:
                results.append(None)
            

            if isinstance(result, list) and len(result) > 0:
                pred_mask = np.array(result[0], dtype=np.uint8)
                palette = np.array(camvid_palette(), dtype=np.uint8)
                if pred_mask.max() >= len(palette):
                    pred_mask = np.clip(pred_mask, 0, len(palette)-1)
                color_mask = palette[pred_mask]
                save_path = os.path.join(save_dir, fname)
                cv2.imwrite(save_path, cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR))
        except Exception as e:
            print(f"Error processing {fname}: {e}")
            results.append(None)
    
 
    if os.path.exists(base_gt_dir):
        from mmseg.core.evaluation import eval_metrics
        preds = []
        gts = []
        
        for i, (img_path, gt_path, fname) in enumerate(tqdm(img_list, desc=f"{fold_config['name']} evalutating")):
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
            
            print(f"\n[{fold_config['name']}] result:")
            print(f"total eval images : {len(preds)}")

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
            
            if 'IoU' in metrics and 'Acc' in metrics:
                all_fold_class_ious.append(metrics['IoU'])
                all_fold_class_accs.append(metrics['Acc'])
                
                if hasattr(model, 'CLASSES') and model.CLASSES is not None:
                    print("\nclass IoU:")
                    for i, (cls_iou, cls_acc) in enumerate(zip(metrics['IoU'], metrics['Acc'])):
                        if i < len(model.CLASSES):
                            print(f"{model.CLASSES[i]}: IoU={cls_iou*100:.2f}, Acc={cls_acc*100:.2f}")
        else:
            print(f"[{fold_config['name']}] error: no pred/gt pair to eval")

print("\n" + "="*80)
print("K-FOLD final result")
print("="*80 + "\n")

if all_fold_metrics:
    
    miou_values = [m['mIoU'] for m in all_fold_metrics if 'mIoU' in m]
    if miou_values:
        avg_miou = np.mean(miou_values)
        print(f"avg mIoU: {avg_miou*100:.2f}%")
        print(f"  - fold1  (fold2 eval): {miou_values[0]*100:.2f}")
        print(f"  - fold2  (fold1 eval): {miou_values[1]*100:.2f}")
    
 
    macc_values = [m['mAcc'] for m in all_fold_metrics if 'mAcc' in m]
    if macc_values:
        avg_macc = np.mean(macc_values)
        print(f"\navg mPA: {avg_macc*100:.2f}%")
        print(f"  - fold1  (fold2 eval): {macc_values[0]*100:.2f}")
        print(f"  - fold2  (fold1 eval): {macc_values[1]*100:.2f}")
    
 
    aacc_values = [m['aAcc'] for m in all_fold_metrics if 'aAcc' in m]
    if aacc_values:
        avg_aacc = np.mean(aacc_values)
        print(f"\navg aAcc: {avg_aacc*100:.2f}%")
        print(f"  - fold1 (fold2 eval): {aacc_values[0]*100:.2f}")
        print(f"  - fold2 (fold1 eval): {aacc_values[1]*100:.2f}")
    
  
    if all_fold_class_ious and hasattr(model, 'CLASSES') and model.CLASSES is not None:
        avg_class_ious = np.mean(all_fold_class_ious, axis=0)
        avg_class_accs = np.mean(all_fold_class_accs, axis=0)
        print("\nclass avg IoU (K-Fold):")
        for i, (cls_iou, cls_acc) in enumerate(zip(avg_class_ious, avg_class_accs)):
            if i < len(model.CLASSES):
                print(f"{model.CLASSES[i]}: IoU={cls_iou*100:.2f}, Acc={cls_acc*100:.2f}")

print("\n" + "="*80)
print("done!")
print("="*80)
