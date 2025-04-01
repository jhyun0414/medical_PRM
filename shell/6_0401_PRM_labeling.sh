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
MODEL_NAME="Llama-3.1-8B-Instruct"

# 모델 경로 배열
# MODEL_PATHS=(
#    "/hdd2/jiwoong/med_PRM/bin/Llama-3.1-8B-Instruct-er_label-2e-6-rp5.0-20250330_121753"
#    "/hdd2/jiwoong/med_PRM/bin/Llama-3.1-8B-Instruct-gemini_label-2e-6-rp5.0-20250330_125012"
#    "/hdd2/jiwoong/med_PRM/bin/Llama-3.1-8B-Instruct-hard_label-2e-6-rp5.0-20250330_121753"
#    "/hdd2/jiwoong/med_PRM/bin/Llama-3.1-8B-Instruct-prm_soft_label-2e-6-rp5.0-20250330_121753"
#)
MODEL_PATHS=("../model/Llama-3.1-8B-Instruct-er_label-2e-6-rp5.0-20250401_064751")

# 각 모델에 할당할 GPU 번호 배열 (순서대로 0, 1, 2, 3)
# GPUS=(0 1 2 3)
GPUS=(7)
# 입출력 JSON 파일 경로
INPUT_JSON="../dataset_sample/test_dataset.json"

# 로그 디렉토리 생성
LOG_DIR="../logs"
mkdir -p "$LOG_DIR"

# 각 모델에 대해 병렬 실행
for i in "${!MODEL_PATHS[@]}"; do
    MODEL_PATH="${MODEL_PATHS[$i]}"
    GPU="${GPUS[$i]}"
    MODEL_BASENAME=$(basename "$MODEL_PATH")
    OUTPUT_JSON="../dataset_sample/${MODEL_BASENAME}.json"
    LOG_FILE="${LOG_DIR}/$(date +'%Y%m%d_%H%M%S')_${MODEL_BASENAME}.log"
    
    echo "====== 평가 설정 (Model: ${MODEL_BASENAME}, GPU: ${GPU}) ======" | tee -a "$LOG_FILE"
    echo "모델명: $MODEL_NAME" | tee -a "$LOG_FILE"
    echo "모델 경로: $MODEL_PATH" | tee -a "$LOG_FILE"
    echo "출력 파일: $OUTPUT_JSON" | tee -a "$LOG_FILE"
    echo "GPU: $GPU" | tee -a "$LOG_FILE"
    echo "====================" | tee -a "$LOG_FILE"
    
    # 단일 GPU 번호를 --device 인자로 넘겨 해당 GPU에서 실행하도록 함.
    python ../python/6_0401_PRM_labeling.py \
        --model_save_path "$MODEL_PATH" \
        --input_json_file "$INPUT_JSON" \
        --output_json_file "$OUTPUT_JSON" \
        --device "$GPU" \
        --hf_token "$HF_TOKEN" \
        --process_solution_num 64 2>&1 | tee -a "$LOG_FILE" &
done

wait
echo "모든 모델 평가 완료."
