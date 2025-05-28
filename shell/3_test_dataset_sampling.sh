#!/bin/bash
if [ -f "../.env" ]; then
  source "../.env"
fi

# 환경 변수가 없으면 오류 메시지 출력 후 종료
if [ -z "$HF_TOKEN" ]; then
  echo "오류: HF_TOKEN이 설정되지 않았습니다. .env 파일을 확인하세요."
  exit 1
fi

MODEL_PATH="../model_downloaded/llama_3.1_8b_instruct"

# 입력 파일 지정
INPUT_FILE="../dataset/dataset_2_raw_test_dataset/0527_final_raw_test_dataset.json"

# 결과를 저장할 출력 디렉토리
OUTPUT_DIR="../dataset/dataset_3_sampled_dataset"

# 처리할 데이터 소스 지정 (쉼표로 구분) / 안적으면 전체 다 실행
# DATA_SOURCE_LIST=""

GPU_ID=0

REPEAT_COUNT=64
TEMPERATURE=0.7
TOP_K=50
TOP_P=0.9
MAX_TOKENS=4096

# 파일명 추출
FILENAME=$(basename "$INPUT_FILE")
# 모델명 추출 (경로에서 마지막 부분만)
MODEL_NAME=$(basename "$MODEL_PATH")

# 로그 디렉토리 생성
log_dir="../logs"
mkdir -p $log_dir

#   --data_source_list "$DATA_SOURCE_LIST" \
# GPU 7에서 단일 파일 처리 (일반 실행)
python ../python/3_test_dataset_sampling.py \
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
  > "${log_dir}/test_${FILENAME}_${MODEL_NAME}_${REPEAT_COUNT}.log" 2>&1
