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

##############################################################################
# 기본 설정 변수들을 미리 지정합니다.
##############################################################################
# Hugging Face 및 W&B 토큰 설정
hf_token="$HF_TOKEN"
wandb_token="$WANDB_TOKEN"

model_name="meta-llama/Llama-3.1-8B-Instruct"
# GPU 설정 (각 GPU에 다른 라벨 할당)
gpus=("0, 1, 2, 3")

#####  다음에는 dataset_path 변경해야함.
dataset_path="../dataset_sample/train_dataset.json"
base_output_dir="../model"
num_train_epochs=1
lr_scheduler_type="cosine"
per_device_train_batch_size=1
gradient_accumulation_steps=64
bf16=True
timestamp=$(date +%Y%m%d_%H%M%S)
logging_steps=1
save_steps=50000
dtype="bfloat16"

# 학습에 사용할 라벨 종류와 learning rate 설정
# 각 GPU에 할당할 라벨 종류
train_labels=("er_label")
learning_rate=2e-6
# 엔트로픽 리스크 계산을 위한 hyperparameter (mu)
risk_param=5.0

# 로그 디렉토리 생성 및 로그 파일 경로 설정
log_dir="../logs"
mkdir -p $log_dir

##############################################################################
# 각 GPU에 다른 라벨로 학습 실행
##############################################################################
for i in "${!gpus[@]}"; do
    # 현재 GPU와 라벨 가져오기
    gpu="${gpus[$i]}"
    train_label="${train_labels[$i]}"
    
    # 현재 시간 갱신
    timestamp=$(date +%Y%m%d_%H%M%S)
    
    # 모델 저장 경로 설정
    output_dir="${base_output_dir}/${model_name##*/}-${train_label}-${learning_rate}-rp${risk_param}-${timestamp}"
    run_name="finetune_${model_name##*/}_${train_label}_${learning_rate}_${risk_param}_${timestamp}"
    log_file="${log_dir}/training_${train_label}_${learning_rate}_${timestamp}.log"
    
    echo "GPU ${gpu}에서 라벨 ${train_label}, 학습률: ${learning_rate}, risk_param: ${risk_param}로 학습 시작"
    
    # 백그라운드로 각 GPU에서 학습 실행
    CUDA_VISIBLE_DEVICES=${gpu} python ../python/5_PRM_finetuning.py \
        --hf_token "$hf_token" \
        --wandb_token "$wandb_token" \
        --device "$gpu" \
        --model_path "$model_name" \
        --dtype "$dtype" \
        --train_json "$dataset_path" \
        --output_dir "$output_dir" \
        --num_train_epochs $num_train_epochs \
        --learning_rate $learning_rate \
        --lr_scheduler_type "$lr_scheduler_type" \
        --per_device_train_batch_size $per_device_train_batch_size \
        --gradient_accumulation_steps $gradient_accumulation_steps \
        --bf16 $bf16 \
        --run_name "$run_name" \
        --logging_steps $logging_steps \
        --save_steps $save_steps \
        --train_label "$train_label" \
        --risk_param "$risk_param" \
        > "$log_file" 2>&1 &
    
    echo "GPU ${gpu}에서 라벨 ${train_label}로 학습 시작됨 (백그라운드 프로세스)"
done

# 모든 백그라운드 프로세스가 완료될 때까지 대기
wait

echo "모든 GPU에서 학습이 완료되었습니다."
