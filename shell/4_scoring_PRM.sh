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
# 기본값 설정

USE_RAG="yes"
USE_ORM="no"

# DATA_SOURCE_LIST='["med_qa"]' 비어있으면 전체 처리
PROCESS_SOLUTION_NUM=64

MODEL_PATHS=(
"../model_downloaded/RAG_PRM_Q_GS"
)
INPUT_JSON="/hdd3/jaehoon/sampling/dataset_re/final_raw_test_dataset_with_related_docs_results_medqa_prm_rs_sft_7k_med_qa_64.json"
# 각 모델에 할당할 GPU 번호 배열
GPUS=(7)

# 출력 디렉토리 설정
OUTPUT_DIR="../dataset_result"

# 최대 토큰 길이 설정 (다른 PRM은 1024로 RAG-PRM은 4096으로)
MAX_TOKEN_LEN=4096

# 옵션 포함 여부 설정 (yes/no)
INCLUDE_OPTIONS="no"

# 로그 디렉토리 생성
LOG_DIR="../logs"
mkdir -p "$LOG_DIR"
mkdir -p "$OUTPUT_DIR"

BASE_INPUT_NAME="$(basename "$INPUT_JSON" .json)"
# 데이터 소스 첫 번째 요소 추출
FIRST_DATA_SOURCE=$(echo $DATA_SOURCE_LIST | sed -E 's/\[\"([^\"]+)\".*/\1/')

# 각 모델에 대해 병렬 실행
for i in "${!MODEL_PATHS[@]}"; do
    MODEL_PATH="${MODEL_PATHS[$i]}"
    GPU="${GPUS[$i]}"
    MODEL_BASENAME="${MODEL_PATH##*/}"
    OUTPUT_JSON="${OUTPUT_DIR}/${MODEL_BASENAME}_${FIRST_DATA_SOURCE}_sol${PROCESS_SOLUTION_NUM}_${BASE_INPUT_NAME}.json"
    LOG_FILE="${LOG_DIR}/TEST_$(date +'%Y%m%d_%H%M%S')_${MODEL_BASENAME}.log"
    
    echo "====== 평가 설정 (Model: ${MODEL_BASENAME}, GPU: ${GPU}) ======" | tee -a "$LOG_FILE"
    echo "모델명: $MODEL_NAME" | tee -a "$LOG_FILE"
    echo "모델 경로: $MODEL_PATH" | tee -a "$LOG_FILE"
    echo "출력 파일: $OUTPUT_JSON" | tee -a "$LOG_FILE"
    echo "GPU: $GPU" | tee -a "$LOG_FILE"
    echo "처리할 솔루션 개수: $SOLUTION_NUM" | tee -a "$LOG_FILE"
    echo "RAG 사용 여부: $USE_RAG" | tee -a "$LOG_FILE"
    echo "최대 토큰 길이: $MAX_TOKEN_LEN" | tee -a "$LOG_FILE"
    echo "옵션 포함 여부: $INCLUDE_OPTIONS" | tee -a "$LOG_FILE"
    echo "ORM 사용 여부: $USE_ORM" | tee -a "$LOG_FILE"
    echo "데이터 소스: $DATA_SOURCE_LIST" | tee -a "$LOG_FILE"
    echo "처리할 솔루션 수: $PROCESS_SOLUTION_NUM" | tee -a "$LOG_FILE"
    echo "====================" | tee -a "$LOG_FILE"
    

    
#        --data_source_list "$DATA_SOURCE_LIST" \
    # 단일 GPU 번호를 --device 인자로 넘겨 해당 GPU에서 실행하도록 함.
    python ../python/2_test.py \
        --model_save_path "$MODEL_PATH" \
        --input_json_file "$INPUT_JSON" \
        --output_json_file "$OUTPUT_JSON" \
        --device "$GPU" \
        --hf_token "$HF_TOKEN" \
        --use_rag "$USE_RAG" \
        --max_token_len "$MAX_TOKEN_LEN" \
        --include_options "$INCLUDE_OPTIONS" \
        --use_orm "$USE_ORM" \
        --process_solution_num "$PROCESS_SOLUTION_NUM" 2>&1 | tee -a "$LOG_FILE" &
done

wait
echo "모든 모델 평가 완료."
