#!/bin/bash

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
GPU_ID=0
MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct"

# 입력 파일 3개 지정
INPUT_FILES=(
  "../dataset_sample/raw_test_dataset.json"
)

# 결과를 저장할 출력 디렉토리
OUTPUT_DIR="../dataset_sample"

REPEAT_COUNT=64
TEMPERATURE=0.7
TOP_K=50
TOP_P=0.9
MAX_TOKENS=4096

# 각 GPU에 할당할 파일 분리 (인덱스 기준)
GPU4_INPUT_FILES=("${INPUT_FILES[0]}")
# GPU5_INPUT_FILES=("${INPUT_FILES[1]}")
# GPU6_INPUT_FILES=("${INPUT_FILES[2]}")

# GPU 4: 1번째 파일 처리
(
  for INPUT_FILE in "${GPU4_INPUT_FILES[@]}"; do
    FILENAME=$(basename "$INPUT_FILE" .json)
    nohup python ../python/2_0325_test_dataset_generation.py \
      --hf_token "$HF_TOKEN" \
      --model_path "$MODEL_PATH" \
      --repeat_count "$REPEAT_COUNT" \
      --temperature "$TEMPERATURE" \
      --top_k "$TOP_K" \
      --top_p "$TOP_P" \
      --max_tokens "$MAX_TOKENS" \
      --input_file "$INPUT_FILE" \
      --output_dir "$OUTPUT_DIR" \
      --gpu_id "$GPU_ID" \
      > "${OUTPUT_DIR}/test_${FILENAME}.log" 2>&1
  done
) &
# 모든 백그라운드 작업이 끝날 때까지 대기
wait
