#!/usr/bin/env python3
"""학습 로그를 확인하는 스크립트"""

import os
import glob
from pathlib import Path
from datetime import datetime

def find_latest_run():
    """가장 최근 학습 실행 디렉토리 찾기"""
    log_dir = Path("logs/rsl_rl/hierarchical_nav")
    if not log_dir.exists():
        return None
    
    runs = sorted(log_dir.glob("20*"), key=os.path.getmtime, reverse=True)
    return runs[0] if runs else None

def show_checkpoint_info(run_dir):
    """체크포인트 파일 정보 표시"""
    checkpoints = sorted(run_dir.glob("model_*.pt"), key=os.path.getmtime)
    if checkpoints:
        print(f"\n{'='*80}")
        print("체크포인트 파일:")
        print(f"{'='*80}\n")
        for ckpt in checkpoints[-10:]:  # 최근 10개
            size = ckpt.stat().st_size / (1024 * 1024)  # MB
            mtime = os.path.getmtime(ckpt)
            mtime_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
            # 파일명에서 iteration 번호 추출
            iter_num = ckpt.stem.split('_')[-1] if '_' in ckpt.stem else '?'
            print(f"  {ckpt.name:25s} | Iter: {iter_num:>6s} | {size:6.2f} MB | {mtime_str}")
    else:
        print("\n체크포인트 파일이 아직 생성되지 않았습니다.")

def show_tensorboard_info(run_dir):
    """TensorBoard 이벤트 파일 정보"""
    event_files = list(run_dir.glob("events.out.tfevents.*"))
    if event_files:
        latest_event = sorted(event_files, key=os.path.getmtime, reverse=True)[0]
        size = latest_event.stat().st_size / 1024  # KB
        mtime = os.path.getmtime(latest_event)
        mtime_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n{'='*80}")
        print("TensorBoard 이벤트 파일:")
        print(f"{'='*80}\n")
        print(f"  파일: {latest_event.name}")
        print(f"  크기: {size:.2f} KB")
        print(f"  수정 시간: {mtime_str}")
        print(f"\n  TensorBoard 실행:")
        print(f"    tensorboard --logdir=logs/rsl_rl/hierarchical_nav --port=6006")
        print(f"    브라우저에서 http://localhost:6006 접속")
    else:
        print("\nTensorBoard 이벤트 파일이 아직 생성되지 않았습니다.")

def show_config_info(run_dir):
    """설정 파일 정보"""
    params_dir = run_dir / "params"
    if params_dir.exists():
        config_files = list(params_dir.glob("*.yaml"))
        if config_files:
            print(f"\n{'='*80}")
            print("설정 파일:")
            print(f"{'='*80}\n")
            for config_file in config_files:
                size = config_file.stat().st_size / 1024  # KB
                print(f"  {config_file.name:30s} | {size:6.2f} KB")
                # 주요 설정 내용 일부 출력
                if config_file.name == "agent.yaml":
                    try:
                        with open(config_file, 'r') as f:
                            lines = f.readlines()[:10]  # 처음 10줄만
                            print("    주요 설정:")
                            for line in lines:
                                if ':' in line and not line.strip().startswith('#'):
                                    print(f"      {line.rstrip()}")
                    except:
                        pass

def main():
    """메인 함수"""
    run_dir = find_latest_run()
    
    if not run_dir:
        print("학습 로그 디렉토리를 찾을 수 없습니다.")
        print("logs/rsl_rl/hierarchical_nav 디렉토리가 존재하는지 확인하세요.")
        return
    
    print(f"\n{'='*80}")
    print("학습 로그 정보")
    print(f"{'='*80}")
    print(f"\n최신 학습 실행: {run_dir.name}")
    print(f"전체 경로: {run_dir.absolute()}\n")
    
    # 체크포인트 정보
    show_checkpoint_info(run_dir)
    
    # TensorBoard 정보
    show_tensorboard_info(run_dir)
    
    # 설정 파일 정보
    show_config_info(run_dir)
    
    print(f"\n{'='*80}\n")

if __name__ == "__main__":
    main()
