#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from huggingface_hub import hf_hub_download
from pathlib import Path
import shutil

def load_env_file(env_path="../.env"):
    """
    .env 파일이 존재하면, KEY=VALUE 형태의 내용을 읽어
    os.environ에 주입합니다.
    """
    if os.path.isfile(env_path):
        print(f"[INFO] .env 파일을 로드합니다: {env_path}")
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                key, value = line.split("=", 1)
                os.environ[key.strip()] = value.strip()

def main():
    # 1) 환경변수에서 토큰 불러오기
    load_env_file("../.env")
    hf_token = os.environ.get("HF_TOKEN")

    # 2) 출력 경로 설정
    output_path = Path("../dataset/2_train_dataset.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 3) 다운로드 수행
    repo_id = "jhyun0414/train_dataset"
    filename = "sample_dataset_50.json"

    print(f"[INFO] Hugging Face에서 {repo_id}/{filename} 파일 다운로드 중...")

    try:
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            token=hf_token,
            repo_type="dataset",
            local_dir_use_symlinks=False
        )

        # 4) 원하는 경로로 복사 (이름 변경)
        shutil.copy(downloaded_path, output_path)
        print(f"[SUCCESS] 파일이 {output_path}로 저장되었습니다.")

    except Exception as e:
        print(f"[ERROR] 다운로드 실패: {e}")

if __name__ == "__main__":
    main()
