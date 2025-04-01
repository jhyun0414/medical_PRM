import argparse
import torch
import re
import json
import random
import os
import math
import wandb
from datetime import datetime

# ✔ Accelerate, DeepSpeed 관련
# from accelerate import Accelerator  # 직접 커스텀 루프 시 필요

from huggingface_hub import login, create_repo
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import Dataset

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune LLM model on PRM dataset.")
    
    # Hugging Face / W&B 토큰
    parser.add_argument("--hf_token", type=str, help="Hugging Face access token.")
    parser.add_argument("--wandb_token", type=str, help="W&B access token.")
    
    # 모델 관련 설정
    parser.add_argument("--model_path", type=str, help="Path or name of the LLM model.")
    parser.add_argument("--device", type=str, help="(Not used in Accelerate) CUDA visible devices (e.g. '0,1')")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Data type (default: bfloat16).")
    
    # 데이터 관련
    parser.add_argument("--train_json", type=str, help="Path to the training JSON file.")
    # train_ratio는 사용하지 않으므로 기본값을 무시합니다.
    
    # 학습 파라미터
    parser.add_argument("--output_dir", type=str, help="Output directory.")
    parser.add_argument("--logging_steps", type=int, help="Logging steps.")
    parser.add_argument("--num_train_epochs", type=int, help="Number of epochs.")
    parser.add_argument("--learning_rate", type=float, help="Learning rate.")
    parser.add_argument("--gradient_accumulation_steps", type=int, help="Gradient accumulation steps.")
    
    parser.add_argument("--lr_scheduler_type", type=str, default="linear", help="Learning rate scheduler type. [linear, cosine, ...]")
    parser.add_argument("--per_device_train_batch_size", type=int, help="Training batch size per device.")
    parser.add_argument("--bf16", type=bool, default=True, help="Whether to use bf16 training.")
    parser.add_argument("--run_name", type=str, help="Name for the run in W&B or logs.")
    parser.add_argument("--save_steps", type=int, help="Interval steps for saving checkpoints.")
    
    parser.add_argument("--train_label", type=str, 
                       choices=["prm_soft_label", "prm_hard_label", "er_label", "gemini_label"],
                       help="Which training label to use. 'er_label' will be computed from soft label with risk_param.")
    parser.add_argument("--risk_param", type=float,
                       help="Entropic risk parameter (mu) for converting soft labels to ER labels.")
    return parser.parse_args()


def load_model(model_name="meta-llama/Llama-3.1-8B-Instruct", dtype="bfloat16"):
    """
    Accelerate + DeepSpeed 사용 시에는 device_map / os.environ 설정 없이 로드하는 게 핵심.
    """
    print("Loading model...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if dtype == "bfloat16" else torch.float16,
        attn_implementation="flash_attention_2",
        device_map="auto"
    )
    model.gradient_checkpointing_enable()  # Zero-3 환경에서도 gradient checkpointing이 유효할 수 있음
    
    # pad_token이 없는 모델(예: LLaMA 계열)을 위해 처리
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Model loaded successfully.")
    return model, tokenizer


def process_all_results_to_dataset(all_results, tokenizer, step_tag_id=128256):
    dataset_dict = {
        'input_ids': [],
        'attention_mask': [],
        'labels': [],
        'values': []
    }
    error_count = 0
    for entry in all_results:
        question_id = entry.get("question_id", "unknown")
        solution_index = entry.get("solution_index", "unknown")
        
        if not isinstance(entry, dict):
            print(f"[ERROR] Skipping non-dict entry for question_id: {question_id}, solution_index: {solution_index}")
            error_count += 1
            continue
        
        raw_text = entry.get('query', "")
        raw_text = raw_text.replace(" ки\n", " ки")
        
        encode = tokenizer(raw_text, add_special_tokens=True, truncation=True)
        new_encode_id = encode['input_ids'].copy()
        new_encode_id.append(tokenizer.pad_token_id)
        
        attention_mask = encode['attention_mask'].copy()
        attention_mask.append(0)
        
        raw_label = raw_text.replace(" ки", " +")
        reference_labels = tokenizer(raw_label, add_special_tokens=True)['input_ids']
        reference_labels.insert(0, tokenizer.pad_token_id)
        reference_labels[0] = -100
        
        value_list = []
        counter = 0
        ann = entry.get('annotation', [])
        
        # 오류 조건 1: 토큰 길이가 맞지 않는 경우
        if len(reference_labels) != len(new_encode_id):
            print(f"[ERROR] Mismatched lengths for question_id: {question_id}, solution_index: {solution_index}. "
                  f"Labels length: {len(reference_labels)}, Input_ids length: {len(new_encode_id)}. Skipping entry.")
            error_count += 1
            continue
        
        skip_entry = False
        for j in range(len(reference_labels)):
            if j == 0:
                value_list.append(0)
                continue
            
            # 만약 해당 위치 토큰이 step_tag_id와 일치하면 annotation을 할당
            if new_encode_id[j - 1] == step_tag_id:
                if counter < len(ann):
                    assigned_value = ann[counter]
                    value_list.append(assigned_value)
                    counter += 1
                else:
                    print(f"[ERROR] No annotation available at position {j} for question_id: {question_id}, solution_index: {solution_index}. Skipping entry.")
                    error_count += 1
                    skip_entry = True
                    break
            else:
                value_list.append(0)
                reference_labels[j] = -100
        
        if skip_entry:
            continue
        
        dataset_dict['input_ids'].append(new_encode_id)
        dataset_dict['attention_mask'].append(attention_mask)
        dataset_dict['labels'].append(reference_labels)
        dataset_dict['values'].append(value_list)
    
    print(f"총 {error_count}개의 solution에서 오류가 발생하였습니다.")
    return dataset_dict


def compute_er_label(p, mu):
    """
    0 <= p <= 1 일 때,
    er_label = (1 / mu) * ln[(1 - p) + p * e^mu]
    """
    return (1.0 / mu) * math.log((1.0 - p) + p * math.exp(mu))

def process_gemini_label(label_list):
    """
    gemini_label 처리 함수:
    0이 처음 등장한 이후의 모든 값을 0으로 변환
    예: [1, 1, 0, 1, 0] -> [1, 1, 0, 0, 0]
    """
    processed_label = []
    found_zero = False
    
    for val in label_list:
        if found_zero:
            processed_label.append(0)
        else:
            processed_label.append(val)
            if val == 0:
                found_zero = True
    
    return processed_label

def format_question_with_options(item):
    """
    입력 JSON의 "question"과 "options" 필드를 재구성하여,
    "질문 본문 (A) 옵션1 (B) 옵션2 ... (N) 옵션N" 형식의 문자열을 반환합니다.
    """
    question_text = item.get("question", "")
    options = item.get("options", [])
    if not options:
        return question_text
    formatted_options = ""
    for idx, opt in enumerate(options):
        # 알파벳 A, B, C, ... 로 옵션을 표기
        letter = chr(ord('A') + idx)
        formatted_options += f" ({letter}) {opt}"
    return question_text + formatted_options
    
def load_json_and_create_dataset_from_data(data, tokenizer, step_tag_id=128256, args=None):
    SYSTEM_PROMPT = (
        "You are an evaluator assessing the quality of reasoning in each step of an given explanation. "
        "If the reasoning in a step is logical and valid, output + after that step. "
        "If the reasoning contains errors, output - after that step."
    )
    
    all_results = []
    for idx, item in enumerate(data):
        # question_id가 없으면 인덱스를 기본값으로 사용
        question_id = item.get("question_id", f"question_{idx}")
        formatted_question = format_question_with_options(item)
        solutions = item.get("solutions", [])
        
        if not solutions:
            continue
        
        for sol_idx, sol in enumerate(solutions):
            # 선택한 label 타입에 따라 새로운 label 생성
            if args.train_label == "er_label":
                original_soft = sol.get("prm_soft_label", [])
                if not isinstance(original_soft, list):
                    original_soft = [original_soft]
                new_label = []
                for p in original_soft:
                    er_val = compute_er_label(p, args.risk_param)
                    new_label.append(er_val)
            elif args.train_label == "gemini_label":
                original_gemini = sol.get("prm_gemini_label", [])
                if not isinstance(original_gemini, list):
                    original_gemini = [original_gemini]
                new_label = process_gemini_label(original_gemini)
            else:
                new_label = sol.get(args.train_label, [])
                if not isinstance(new_label, list):
                    new_label = [new_label]
            
            solution_text = sol.get("prm_processed_solution", "")
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": formatted_question + solution_text}
            ]
            
            raw_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # question_id와 solution_index를 함께 저장
            new_entry = {
                "query": raw_text,
                "annotation": new_label,
                "question_id": question_id,
                "solution_index": sol_idx
            }
            all_results.append(new_entry)
    
    dataset_dict = process_all_results_to_dataset(
        all_results,
        tokenizer,
        step_tag_id=step_tag_id
    )
    return dataset_dict


class AutoRegressiveTrainer(Trainer):
    """
    +, - 두 토큰에 대해 Soft Label Cross Entropy를 적용하는 Trainer 예시
    """
    def __init__(self, my_tokenizer=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.my_tokenizer = my_tokenizer  # 커스텀 필드에 저장
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs["logits"]
        
        # shift
        logits = logits[..., :-1, :].contiguous().to(torch.bfloat16)
        labels = labels[..., 1:].contiguous()
        values = inputs['values'][..., 1:].contiguous().to(torch.bfloat16)
        
        # + / - 토큰만 추출
        plus_id = self.my_tokenizer.encode(' +')[-1]
        minus_id = self.my_tokenizer.encode(' -')[-1]
        plus_logits = logits[:, :, plus_id]
        minus_logits = logits[:, :, minus_id]
        
        # 차원 맞추기
        if labels.dim() == plus_logits.dim() - 1:
            labels = labels.unsqueeze(-1)
            values = values.unsqueeze(-1)
        
        chosen = (labels != -100)
        pred_plus_values = plus_logits[chosen]
        pred_minus_values = minus_logits[chosen]
        gt_values = values[chosen]  # 실제 라벨(0~1 사이 값 가능)
        
        # pred_combined: [batch_size*num_steps, 2]
        pred_combined = torch.stack((pred_plus_values, pred_minus_values), dim=1)
        gt_negative = 1 - gt_values
        gt_combined = torch.stack((gt_values, gt_negative), dim=1)
        
        # soft label cross entropy
        loss = torch.nn.functional.cross_entropy(
            pred_combined, 
            gt_combined, 
            reduction="mean"
        )
        loss = loss.to(torch.bfloat16)
        return (loss, outputs) if return_outputs else loss


def main():
    args = parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # 1. 로그인
    login(args.hf_token)
    wandb.login(key=args.wandb_token)
    print("1: Login Success")

    # 2. 모델 불러오기
    model, tokenizer = load_model(model_name=args.model_path, dtype=args.dtype)
    print("2: Model Loading Success")

    # (원한다면) 스페셜 토큰 추가 예시
    special_tokens = {"additional_special_tokens": [" ки"]}
    tokenizer.add_special_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))

    step_tag = 'ки'
    step_tag_id = tokenizer.encode(f" {step_tag}")[-1]
    plus_tag_id = tokenizer.encode(' +')[-1]
    minus_tag_id = tokenizer.encode(' -')[-1]
    print("Step tag ID:", step_tag_id)
    print("Plus tag ID:", plus_tag_id)
    print("Minus tag ID:", minus_tag_id)

    # 3. 데이터 준비 (전체 데이터를 train에 사용)
    with open(args.train_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    random.shuffle(data)
    train_data = data  # 전체 데이터를 train_data로 사용
    train_dict = load_json_and_create_dataset_from_data(train_data, tokenizer, step_tag_id=128256, args=args)
    train_dataset = Dataset.from_dict(train_dict)

    # 길이 고정 padding
    def pad_sequences(example, max_length=1023, pad_value=0):
        input_ids = example["input_ids"]
        attention_mask = example["attention_mask"]
        labels = example["labels"]
        values = example["values"]
        
        if len(input_ids) < max_length:
            padding_length = max_length - len(input_ids)
            input_ids += [pad_value] * padding_length
            attention_mask += [0] * padding_length
            values += [0] * padding_length
            labels += [-100] * padding_length
        else:
            input_ids = input_ids[:max_length]
            attention_mask = attention_mask[:max_length]
            values = values[:max_length]
            labels = labels[:max_length]
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "values": values,
        }

    train_dataset = train_dataset.map(lambda x: pad_sequences(x, max_length=1023))

    print("3: Data Pre-processing Success")

    # 4. Trainer
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 5. TrainingArguments (평가 관련 항목 제거)
    model_name = args.model_path.split("/")[-1]
    specific_output_dir = f"{args.output_dir}"
    os.makedirs(specific_output_dir, exist_ok=True)
    
    if not args.run_name:
        args.run_name = f"finetune_{model_name}_{args.train_label}_{args.learning_rate}"

    training_args = TrainingArguments(
        output_dir=specific_output_dir,
        # 평가 관련 항목 제거
        per_device_train_batch_size=args.per_device_train_batch_size,
        num_train_epochs=args.num_train_epochs,
        save_strategy="epoch",
        logging_steps=args.logging_steps,
        save_total_limit=10,
        run_name=args.run_name,
        bf16=args.bf16,  # BF16 활성화
        remove_unused_columns=False,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        report_to=["wandb"],
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=0.05,
        ddp_find_unused_parameters=False,
    )

    # eval_dataset 인자 없이 Trainer 초기화
    finetuner = AutoRegressiveTrainer(
        model=model,
        train_dataset=train_dataset,
        my_tokenizer=tokenizer,
        args=training_args,
        data_collator=data_collator,
    )

    print("4: Finetuning Ready")

    # 6. Fine-tuning
    print(f"Fine-tuning 시작... (learning rate: {args.learning_rate}, risk_param(mu): {args.risk_param})")
    finetuner.train()
    print("Fine-tuning 완료!")

    finetuner.save_model(specific_output_dir)
    tokenizer.save_pretrained(specific_output_dir)
    print(f"모델이 {specific_output_dir}에 저장되었습니다.")

    # 8. Hugging Face Hub 업로드(선택)
    date_str = datetime.now().strftime("%Y%m%d")
    repo_name = f"jhyun0414/{date_str}-{model_name}-{args.train_label}-{args.risk_param}"
    create_repo(repo_name, exist_ok=True, private=False)
    finetuner.model.push_to_hub(repo_name)
    tokenizer.push_to_hub(repo_name)
    print(f"🚀 모델 업로드 완료! 🔗 확인: https://huggingface.co/{repo_name}")


if __name__ == "__main__":
    # Accelerate로 실행할 때: accelerate launch --config_file accelerate_config.yaml train.py --args...
    main()
