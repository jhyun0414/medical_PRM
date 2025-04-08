#!/usr/bin/env bash

##############################################################################
# 기본 설정 변수들을 미리 지정합니다.
##############################################################################
# .env 파일에서 환경 변수 로드
if [ -f "../.env" ]; then
  source "../.env"
fi

# 환경 변수가 없으면 기본값 사용
if [ -z "$HF_TOKEN" ]; then
  echo "경고: HF_TOKEN이 설정되지 않았습니다. 기본값을 사용합니다."
  hf_token=""
else
  hf_token="$HF_TOKEN"
fi

if [ -z "$WANDB_TOKEN" ]; then
  echo "경고: WANDB_TOKEN이 설정되지 않았습니다. 기본값을 사용합니다."
  wandb_token=""
else
  wandb_token="$WANDB_TOKEN"
fi

model_name="OpenMeditron/Meditron3-8B"
# GPU 설정 (하나의 GPU만 사용)
gpu="0"

##### 다음에는 dataset_path 변경해야함.
dataset_path="../dataset/2_train_dataset.json"
base_output_dir="../model"
num_train_epochs=1
lr_scheduler_type="cosine"
per_device_train_batch_size=1
gradient_accumulation_steps=64
bf16=True
logging_steps=1
save_steps=50000
dtype="bfloat16"

# 학습에 사용할 라벨 종류 (2종)
train_labels=("gemini_label" "prm_soft_label")
learning_rate=2e-6
# 엔트로픽 리스크 계산을 위한 hyperparameter (mu)
risk_param=5.0

# ✨ 필터링 여부 (yes / no) 셸 스크립트에서 지정
do_filtering="no"

# 로그 디렉토리 생성 및 로그 파일 경로 설정
log_dir="../logs"
mkdir -p $log_dir

# 학습에 사용되는 Python 코드 파일명
python_script="6_0408_PRM_finetuning_vanilla.py"


##############################################################################
# 각 라벨에 대해 순차적으로 학습 실행
##############################################################################
for train_label in "${train_labels[@]}"; do
    # 현재 시간 갱신
    timestamp=$(date +%Y%m%d_%H%M%S)
    
    # 모델 저장 경로 설정 (model_name과 epoch 포함)
    model_name_short="${model_name##*/}"
    output_dir="${base_output_dir}/${model_name_short}-${train_label}-${num_train_epochs}ep-${learning_rate}-rp${risk_param}-${python_script%.py}-${timestamp}"
    run_name="finetune_${model_name_short}_${train_label}_${num_train_epochs}ep_${learning_rate}_${risk_param}_${python_script%.py}_${timestamp}"
    log_file="${log_dir}/${timestamp}_training_${train_label}_${num_train_epochs}ep_${learning_rate}.log"
    
    echo "GPU ${gpu}에서 라벨 ${train_label}, 학습률: ${learning_rate}, risk_param: ${risk_param}, epochs: ${num_train_epochs}, do_filtering: ${do_filtering}로 학습 시작"
    
    # 각 라벨에 대해 순차적으로 학습 실행
    CUDA_VISIBLE_DEVICES=${gpu} python ../python/${python_script} \
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
        --do_filtering "$do_filtering" \
        > "$log_file" 2>&1
    
    echo "GPU ${gpu}에서 라벨 ${train_label}로 학습 완료"
done

echo "모든 라벨에 대한 학습이 완료되었습니다."
