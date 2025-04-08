#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
from huggingface_hub import hf_hub_download
from pathlib import Path
import json

def parse_args():
    parser = argparse.ArgumentParser(description="Hugging Face 데이터셋 다운로더")
    parser.add_argument("--output_dir", type=str, default="../dataset", 
                        help="데이터셋을 저장할 디렉토리 경로")
    parser.add_argument("--hf_token", type=str, default=None,
                        help="Hugging Face API 토큰 (필요한 경우)")
    return parser.parse_args()

def download_dataset(repo_id, filename, output_dir, token=None):
    """Hugging Face Hub에서 데이터셋 파일을 다운로드합니다."""
    try:
        print(f"[INFO] '{repo_id}'에서 '{filename}' 다운로드 중...")
        output_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            token=token,
            repo_type="dataset",
            local_dir=output_dir,
            local_dir_use_symlinks=False
        )
        print(f"[SUCCESS] 파일이 성공적으로 다운로드되었습니다: {output_path}")
        return output_path
    except Exception as e:
        print(f"[ERROR] 다운로드 실패: {e}")
        return None

def main():
    args = parse_args()
    
    # 환경 변수에서 토큰 가져오기 (명령줄 인수로 제공되지 않은 경우)
    token = args.hf_token or os.environ.get("HF_TOKEN")
    
    # 출력 디렉토리 생성
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 다운로드할 데이터셋 정의
    datasets = [
        {"repo_id": "jhyun0414/raw_train_dataset", "filename": "raw_train_dataset.json"},
        {"repo_id": "jhyun0414/raw_sample_train_dataset", "filename": "raw_sample_train_dataset.json"}
    ]
    
    # 데이터셋 다운로드
    downloaded_files = []
    for dataset in datasets:
        file_path = download_dataset(
            repo_id=dataset["repo_id"],
            filename=dataset["filename"],
            output_dir=output_dir,
            token=token
        )
        if file_path:
            downloaded_files.append(file_path)
    
    # 다운로드 결과 요약
    print("\n===== 다운로드 요약 =====")
    print(f"총 다운로드 파일 수: {len(downloaded_files)}")
    for file_path in downloaded_files:
        print(f"- {file_path}")
    print("========================")

if __name__ == "__main__":
    main()
