#!/bin/bash

# 사용 예:
#  chmod +x run_train_generation.sh
#  ./run_train_generation.sh

############################################
# 환경 변수 로드
############################################
# .env 파일에서 환경 변수 로드
if [ -f "../.env" ]; then
  source "../.env"
fi

# 환경 변수가 없으면 오류 메시지 출력 후 종료
if [ -z "$HF_TOKEN" ]; then
  echo "오류: HF_TOKEN이 설정되지 않았습니다. .env 파일을 확인하세요."
  exit 1
fi

############################################
# 사용자 지정 파라미터
############################################
MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct"

# 입력 파일과 출력 디렉토리 지정 (상대 경로 사용)
INPUT_FILE="../dataset_sample/raw_train_dataset.json"
OUTPUT_DIR="../dataset_sample"

REPEAT_COUNT=16
COMPLETION_NUMBER=16
TEMPERATURE=0.7
TOP_K=50
TOP_P=0.9
MAX_TOKENS=1024

# GPU 및 처리할 문제 범위 지정 (8개 GPU 병렬 실행)
GPU_IDS=("0")
START_NUMBERS=(0)
END_NUMBERS=(1)  # 종료번호는 미포함

############################################
# 병렬 실행
############################################
N=${#START_NUMBERS[@]}
for (( i=0; i<${N}; i++ )); do
    gpu_id=${GPU_IDS[i]}
    start_number=${START_NUMBERS[i]}
    end_number=${END_NUMBERS[i]}

    echo ">>>>>>>>>>> 실행: GPU $gpu_id, 문제번호 [${start_number} ~ ${end_number}) <<<<<<<<<<<"

    nohup python "../python/1_0325_train_dataset_generation.py" \
      --hf_token "${HF_TOKEN}" \
      --model_path "${MODEL_PATH}" \
      --repeat_count "${REPEAT_COUNT}" \
      --completion_number "${COMPLETION_NUMBER}" \
      --temperature "${TEMPERATURE}" \
      --top_k "${TOP_K}" \
      --top_p "${TOP_P}" \
      --max_tokens "${MAX_TOKENS}" \
      --input_file "${INPUT_FILE}" \
      --output_dir "${OUTPUT_DIR}" \
      --start_number "${start_number}" \
      --end_number "${end_number}" \
      --gpu_ids "${gpu_id}" \
      > "${OUTPUT_DIR}/train_${start_number}_${end_number}.log" 2>&1 &
done

wait
echo "✅ 모든 병렬 작업이 완료되었습니다!"
