#!/usr/bin/env python
"""
KD Checkpoint Converter: Teacher weights 제거 및 Student weights만 저장

기존 KD checkpoint를 student-only checkpoint로 변환합니다.
원본은 _backup.pth로 백업되고, 원본 파일명에 student weight만 저장됩니다.

Usage:
    # 단일 파일 변환
    python tools/student_weight_convert.py work_dirs/kd/xxx/iter_30000.pth
    
    # 디렉토리 전체 변환
    python tools/student_weight_convert.py work_dirs/kd/xxx/
    
    # 백업 없이 덮어쓰기 (주의!)
    python tools/student_weight_convert.py work_dirs/kd/xxx/ --no-backup
    
    # 재귀적으로 모든 하위 디렉토리 변환
    python tools/student_weight_convert.py work_dirs/kd/ --recursive
"""

import argparse
import os
import shutil
from pathlib import Path
import torch
from tqdm import tqdm


def convert_to_student_only(ckpt_path, backup=True, verbose=True):
    """
    KD checkpoint를 student-only로 변환
    
    Args:
        ckpt_path (str): checkpoint 파일 경로
        backup (bool): 원본을 _backup.pth로 백업할지 여부
        verbose (bool): 상세 정보 출력 여부
    
    Returns:
        tuple: (success, original_size_mb, new_size_mb)
    """
    ckpt_path = Path(ckpt_path)
    
    if not ckpt_path.exists():
        print(f" File not found: {ckpt_path}")
        return False, 0, 0
    
    # 이미 변환된 파일인지 확인
    if '_backup' in ckpt_path.name or '_student' in ckpt_path.name:
        if verbose:
            print(f" Skipping (backup/student file): {ckpt_path.name}")
        return False, 0, 0
    
    try:
        # 원본 크기 확인
        original_size = os.path.getsize(ckpt_path) / (1024 ** 2)
        
        # Checkpoint 로드
        if verbose:
            print(f"\n Loading: {ckpt_path.name}")
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        
        # State dict 확인
        if 'state_dict' not in checkpoint:
            print(f" No 'state_dict' found in {ckpt_path.name}")
            return False, original_size, 0
        
        # Teacher weights 찾기
        original_state = checkpoint['state_dict']
        teacher_keys = [k for k in original_state.keys() 
                       if 'teacher' in k.lower()]
        
        if len(teacher_keys) == 0:
            if verbose:
                print(f"✓ Already student-only (no teacher weights found)")
            return False, original_size, original_size
        
        # Student weights만 필터링
        student_state = {k: v for k, v in original_state.items() 
                        if k not in teacher_keys}
        
        checkpoint['state_dict'] = student_state
        
        # 백업 생성
        if backup:
            backup_path = ckpt_path.with_name(ckpt_path.stem + '_backup.pth')
            if backup_path.exists():
                if verbose:
                    print(f"Backup already exists: {backup_path.name}")
            else:
                shutil.copy2(ckpt_path, backup_path)
                if verbose:
                    print(f"Backup created: {backup_path.name}")
        
        # Student-only checkpoint 저장 (원본 덮어쓰기)
        torch.save(checkpoint, ckpt_path)
        
        # 새 크기 확인
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
    디렉토리 내 모든 checkpoint 변환
    
    Args:
        directory (str): 디렉토리 경로
        backup (bool): 백업 생성 여부
        recursive (bool): 하위 디렉토리도 탐색할지 여부
        pattern (str): 파일 패턴
    """
    directory = Path(directory)
    
    if not directory.exists():
        print(f"Directory not found: {directory}")
        return
    
    # 파일 찾기
    if recursive:
        pth_files = list(directory.rglob(pattern))
    else:
        pth_files = list(directory.glob(pattern))
    
    # 백업/student 파일 제외
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
            # Progress bar와 함께 간단한 정보만 출력
            tqdm.write(f"✓ {pth_file.name}: {orig_size:.1f}MB → {new_size:.1f}MB")
    
    # 최종 요약
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
        # 단일 파일 변환
        convert_to_student_only(path, backup=backup, verbose=True)
    elif path.is_dir():
        # 디렉토리 변환
        convert_directory(path, backup=backup, recursive=args.recursive, 
                         pattern=args.pattern)
    else:
        print(f"Invalid path: {path}")


if __name__ == '__main__':
    main()