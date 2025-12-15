#!/usr/bin/env python
"""
KD Checkpoint Converter:

Usage:
    # single file
    python tools/student_weight_convert.py work_dirs/kd/xxx/iter_30000.pth
    
    # directory entire
    python tools/student_weight_convert.py work_dirs/kd/xxx/
    
    # without backup
    python tools/student_weight_convert.py /home/yeonwoo3/DIFF/work_dirs/kd/Sentence_teacher/mse_0.01/fold2/best_mIoU_iter_29000.pth --no-backup
    
    # all recursive
    python tools/student_weight_convert.py /home/yeonwoo3/DIFF/work_dirs/kd/sim_pre_0.1_Multi_LabelTeacher/ --recursive
"""

import argparse
import os
import shutil
from pathlib import Path
import torch
from tqdm import tqdm


def convert_to_student_only(ckpt_path, backup=True, verbose=True):
    """
    
    Args:
        ckpt_path (str): checkpoint
        backup (bool)
        verbose (bool)
    
    Returns:
        tuple: (success, original_size_mb, new_size_mb)
    """
    ckpt_path = Path(ckpt_path)
    
    if not ckpt_path.exists():
        print(f" File not found: {ckpt_path}")
        return False, 0, 0
    

    if '_backup' in ckpt_path.name or '_student' in ckpt_path.name:
        if verbose:
            print(f" Skipping (backup/student file): {ckpt_path.name}")
        return False, 0, 0
    
    try:

        original_size = os.path.getsize(ckpt_path) / (1024 ** 2)
        

        if verbose:
            print(f"\n Loading: {ckpt_path.name}")
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        

        if 'state_dict' not in checkpoint:
            print(f" No 'state_dict' found in {ckpt_path.name}")
            return False, original_size, 0
        
 
        original_state = checkpoint['state_dict']
        teacher_keys = [k for k in original_state.keys() 
                       if 'teacher' in k.lower()]
        
        if len(teacher_keys) == 0:
            if verbose:
                print(f"✓ Already student-only (no teacher weights found)")
            return False, original_size, original_size
        

        student_state = {k: v for k, v in original_state.items() 
                        if k not in teacher_keys}
        
        checkpoint['state_dict'] = student_state
        

        if backup:
            backup_path = ckpt_path.with_name(ckpt_path.stem + '_backup.pth')
            if backup_path.exists():
                if verbose:
                    print(f"Backup already exists: {backup_path.name}")
            else:
                shutil.copy2(ckpt_path, backup_path)
                if verbose:
                    print(f"Backup created: {backup_path.name}")
        

        torch.save(checkpoint, ckpt_path)
        

        new_size = os.path.getsize(ckpt_path) / (1024 ** 2)
        saved = original_size - new_size
        
        if verbose:
            print(f" Converted: {ckpt_path.name}")
            print(f"   - Removed: {len(teacher_keys)} teacher parameters")
            print(f"   - Kept: {len(student_state)} student parameters")
            print(f"   - Size: {original_size:.1f} MB → {new_size:.1f} MB "
                  f"(saved {saved:.1f} MB, {saved/original_size*100:.1f}%)")
        
        return True, original_size, new_size
        
    except Exception as e:
        print(f"Error converting {ckpt_path}: {e}")
        return False, 0, 0


def convert_directory(directory, backup=True, recursive=False, pattern='*.pth'):
    """    
    Args:
        directory (str)
        backup (bool)
        recursive (bool)
        pattern (str)
    """
    directory = Path(directory)
    
    if not directory.exists():
        print(f"Directory not found: {directory}")
        return
    
    if recursive:
        pth_files = list(directory.rglob(pattern))
    else:
        pth_files = list(directory.glob(pattern))
    
    pth_files = [f for f in pth_files 
                 if '_backup' not in f.name and '_student' not in f.name]
    
    if len(pth_files) == 0:
        print(f"No checkpoint files found in {directory}")
        return
    
    print(f"\n{'='*70}")
    print(f"Found {len(pth_files)} checkpoint files to convert")
    print(f"{'='*70}")
    
    total_original = 0
    total_new = 0
    success_count = 0
    
    for pth_file in tqdm(pth_files, desc="Converting"):
        success, orig_size, new_size = convert_to_student_only(
            pth_file, backup=backup, verbose=False
        )
        if success:
            success_count += 1
            total_original += orig_size
            total_new += new_size
            tqdm.write(f"✓ {pth_file.name}: {orig_size:.1f}MB → {new_size:.1f}MB")
    
    print(f"\n{'='*70}")
    print(f"Conversion Summary:")
    print(f"  - Total converted: {success_count}/{len(pth_files)} files")
    print(f"  - Total space saved: {total_original - total_new:.1f} MB "
          f"({(total_original - total_new)/1024:.2f} GB)")
    print(f"  - Average reduction: {(total_original - total_new)/total_original*100:.1f}%")
    if backup:
        print(f"\n Original files backed up with '_backup.pth' suffix")
        print(f"   You can delete them after verification:")
        print(f"   find {directory} -name '*_backup.pth' -delete")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Convert KD checkpoints to student-only by removing teacher weights',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        'path',
        type=str,
        help='Path to checkpoint file or directory'
    )
    
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Do not create backup files (DANGEROUS!)'
    )
    
    parser.add_argument(
        '--recursive', '-r',
        action='store_true',
        help='Recursively process all subdirectories'
    )
    
    parser.add_argument(
        '--pattern',
        type=str,
        default='*.pth',
        help='File pattern to match (default: *.pth)'
    )
    
    args = parser.parse_args()
    
    path = Path(args.path)
    backup = not args.no_backup
    
    if path.is_file():
        convert_to_student_only(path, backup=backup, verbose=True)
    elif path.is_dir():
        convert_directory(path, backup=backup, recursive=args.recursive, 
                         pattern=args.pattern)
    else:
        print(f"Invalid path: {path}")


if __name__ == '__main__':
    main()
