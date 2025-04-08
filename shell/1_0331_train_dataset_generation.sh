#!/bin/bash

# 사용 예:
#  chmod +x run_solution_sampling.sh
#  ./run_solution_sampling.sh

############################################
# 환경 변수 로드 (.env 파일에 HF_TOKEN 등 정의)
############################################
if [ -f "../.env" ]; then
  source "../.env"
fi

if [ -z "$HF_TOKEN" ]; then
  echo "오류: HF_TOKEN이 설정되지 않았습니다. .env 파일을 확인하세요."
  exit 1
fi

############################################
# 사용자 지정 파라미터 (필요에 따라 수정하세요)
############################################
# GPU ID들 (쉼표로 구분된 배열, 필요시 수정)
GPU_IDS=("2" "2")

# HF_TOKEN은 .env에서 로드되므로 별도 하드코딩하지 마세요.
MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct"

# 입력 파일과 출력 디렉토리/파일 지정 (상대경로 사용)
INPUT_FILE="../dataset/0_sample_raw_train_dataset.json"
OUTPUT_DIR="../dataset"
OUTPUT_FILE="${OUTPUT_DIR}/1_solution_sampling_output.json"
LOG_DIR="../logs"

REPEAT_COUNT=16
TEMPERATURE=0.7
TOP_K=50
TOP_P=0.9
MAX_TOKENS=1024

############################################
# 문제 분할 계산
############################################
GPU_COUNT=${#GPU_IDS[@]}
if [ $GPU_COUNT -eq 0 ]; then
    echo "오류: GPU ID가 지정되지 않았습니다."
    exit 1
fi

# jq 명령어를 사용하여 전체 문항 수 확인
TOTAL_PROBLEMS=$(jq length "$INPUT_FILE")
echo "총 문항 수: $TOTAL_PROBLEMS"

# 각 GPU에 할당할 문항 수 계산
PROBLEMS_PER_GPU=$((TOTAL_PROBLEMS / GPU_COUNT))
REMAINDER=$((TOTAL_PROBLEMS % GPU_COUNT))

# 각 GPU에 할당할 시작/종료 번호 계산
START_NUMBERS=()
END_NUMBERS=()
current_start=0

for ((i=0; i<GPU_COUNT; i++)); do
    problems_for_this_gpu=$PROBLEMS_PER_GPU
    if [ $i -lt $REMAINDER ]; then
        problems_for_this_gpu=$((problems_for_this_gpu + 1))
    fi

    START_NUMBERS+=($current_start)
    current_end=$((current_start + problems_for_this_gpu))
    END_NUMBERS+=($current_end)
    current_start=$current_end
done

# 임시 파일과 로그를 저장할 디렉토리 생성
mkdir -p "${OUTPUT_DIR}/temp"
mkdir -p "${LOG_DIR}"

############################################
# 병렬 실행
############################################
echo "총 ${GPU_COUNT}개의 GPU를 사용하여 문제를 분할합니다:"
for ((i=0; i<GPU_COUNT; i++)); do
    echo "GPU ${GPU_IDS[i]}: 문제번호 [${START_NUMBERS[i]} ~ ${END_NUMBERS[i]})"
done

for ((i=0; i<GPU_COUNT; i++)); do
    gpu_id=${GPU_IDS[i]}
    start_number=${START_NUMBERS[i]}
    end_number=${END_NUMBERS[i]}
    temp_output_file="${OUTPUT_DIR}/temp/output_${start_number}_${end_number}.json"

    echo ">>>>>>>>>>> 실행: GPU $gpu_id, 문제번호 [${start_number} ~ ${end_number}) <<<<<<<<<<<"

    nohup python "../python/1_solution_sampling.py" \
      --hf_token "${HF_TOKEN}" \
      --model_path "${MODEL_PATH}" \
      --repeat_count "${REPEAT_COUNT}" \
      --temperature "${TEMPERATURE}" \
      --top_k "${TOP_K}" \
      --top_p "${TOP_P}" \
      --max_tokens "${MAX_TOKENS}" \
      --input_file "${INPUT_FILE}" \
      --output_file "${temp_output_file}" \
      --start_number "${start_number}" \
      --end_number "${end_number}" \
      --gpu_ids "${gpu_id}" \
      > "${LOG_DIR}/solution_sampling_${start_number}_${end_number}.log" 2>&1 &
done

wait
echo "✅ 모든 병렬 작업이 완료되었습니다!"

############################################
# 결과 파일 합치기
############################################
echo "결과 파일들을 합치는 중..."
if [ ${GPU_COUNT} -eq 1 ]; then
    cp "${OUTPUT_DIR}/temp/output_${START_NUMBERS[0]}_${END_NUMBERS[0]}.json" "${OUTPUT_FILE}"
else
    # 첫 번째 파일에서 JSON 배열의 시작 부분 복사
    first_file="${OUTPUT_DIR}/temp/output_${START_NUMBERS[0]}_${END_NUMBERS[0]}.json"
    head -n -1 "${first_file}" > "${OUTPUT_FILE}"

    # 중간 파일들 추가 (마지막 줄 제외)
    for ((i=1; i<GPU_COUNT-1; i++)); do
        start_number=${START_NUMBERS[i]}
        end_number=${END_NUMBERS[i]}
        temp_file="${OUTPUT_DIR}/temp/output_${start_number}_${end_number}.json"
        tail -n +2 "${temp_file}" | head -n -1 >> "${OUTPUT_FILE}"
    done

    # 마지막 파일 추가 (마지막 줄 포함)
    last_file="${OUTPUT_DIR}/temp/output_${START_NUMBERS[$GPU_COUNT-1]}_${END_NUMBERS[$GPU_COUNT-1]}.json"
    tail -n +2 "${last_file}" >> "${OUTPUT_FILE}"
fi

# 임시 파일 삭제
rm -rf "${OUTPUT_DIR}/temp"

echo "✅ 결과 파일이 성공적으로 합쳐졌습니다: ${OUTPUT_FILE}"
