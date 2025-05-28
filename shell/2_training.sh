#!/usr/bin/env bash
##############################################################################
# 환경 변수 로드
##############################################################################
# .env 파일에서 환경 변수 로드
if [ -f "../.env" ]; then
  source "../.env"
fi

# 환경 변수가 없으면 오류 메시지 출력 후 종료
if [ -z "$HF_TOKEN" ]; then
  echo "오류: HF_TOKEN이 설정되지 않았습니다. .env 파일을 확인하세요."
  exit 1
fi

if [ -z "$WANDB_TOKEN" ]; then
  echo "오류: WANDB_TOKEN이 설정되지 않았습니다. .env 파일을 확인하세요."
  exit 1
fi


# 주요 setting들
use_rag="yes"
train_label="gemini_label"

max_token_len=4096

num_train_epochs=3
do_filtering="yes"

# 환경별 setting
gpu="0"
online=False


##############################################################################
# 기본 설정 변수들
##############################################################################
# Hugging Face 및 W&B 토큰 설정
hf_token="$HF_TOKEN"
wandb_token="$WANDB_TOKEN"
wandb_project="huggingface"




# 모델 경로 설정
model_name="meta-llama/Llama-3.1-8B-Instruct"
# 경로 설정
dataset_path="../dataset/1_train_dataset.json"
base_output_dir="../model_train"

# 데이터셋 파일 존재 여부 확인
if [ ! -f "$dataset_path" ]; then
  echo "오류: 데이터셋 파일($dataset_path)이 존재하지 않습니다."
  exit 1
fi

# 출력 디렉토리 존재 여부 확인 및 생성
if [ ! -d "$base_output_dir" ]; then
  echo "출력 디렉토리($base_output_dir)가 존재하지 않아 생성합니다."
  mkdir -p "$base_output_dir"
fi

# 학습 파라미터 설정
lr_scheduler_type="cosine"
per_device_train_batch_size=1
gradient_accumulation_steps=64
bf16=True
timestamp=$(date +%Y%m%d_%H%M%S)
logging_steps=1
save_steps=50000
dtype="bfloat16"
train_ratio=1.0
# 학습 라벨 및 하이퍼파라미터 설정

learning_rate=2e-6
risk_param=5.0


# 로그 디렉토리 생성
log_dir="../logs"
mkdir -p $log_dir

##############################################################################
# 학습 실행
##############################################################################
# 모델 이름 추출


# 모델 저장 경로 설정
output_dir="${base_output_dir}/${model_name}-${train_label}-filter_${do_filtering}-ep${num_train_epochs}-${timestamp}-RAG_${use_rag}"
run_name="finetune_${model_name}_${train_label}_filter_${do_filtering}_ep${num_train_epochs}_${timestamp}-RAG_${use_rag}"
log_file="${log_dir}/TRAIN_${train_label}_${learning_rate}_${timestamp}-RAG_${use_rag}.log"

# Python 스크립트 경로 확인
python_script="../python/1_train.py"

# 로그 시작 메시지
echo "1: 학습 시작" > "$log_file"


# 단일 GPU에서 학습 실행 (로그 파일에 출력 리다이렉션)
TRANSFORMERS_NO_DEEPSPEED=1 \
CUDA_VISIBLE_DEVICES=${gpu} python "$python_script" \
    --model_path "$model_name" \
    --device "$gpu" \
    --dtype "$dtype" \
    --max_token_len "$max_token_len" \
    --train_json "$dataset_path" \
    --train_ratio "$train_ratio" \
    --output_dir "$output_dir" \
    --logging_steps $logging_steps \
    --num_train_epochs $num_train_epochs \
    --learning_rate $learning_rate \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --lr_scheduler_type "$lr_scheduler_type" \
    --per_device_train_batch_size $per_device_train_batch_size \
    --bf16 $bf16 \
    --run_name "$run_name" \
    --save_steps $save_steps \
    --train_label "$train_label" \
    --risk_param "$risk_param" \
    --do_filtering "$do_filtering" \
    --use_rag "$use_rag" \
    --online $online \
    --wandb_token "$wandb_token" \
    --wandb_project "$wandb_project" \
    --hf_token "$hf_token" \
    2>&1 | tee -a "$log_file"

echo "학습이 완료되었습니다. 로그는 ${log_file}에 저장되었습니다."
