import argparse
import torch
import re
import json
import random
import os
import math
import wandb
from datetime import datetime
import shutil
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
    parser.add_argument("--model_path", type=str, default="meta-llama/Llama-3.1-8B",
                        help="Path or name of the base LLM model (default: Llama-3.1-8B).")
    parser.add_argument("--device", type=str, help="CUDA visible devices (e.g. '0,1')")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Data type (default: bfloat16).")
    
    # 데이터 관련
    parser.add_argument("--train_json", type=str, help="Path to the training JSON file.")
    parser.add_argument("--train_ratio", type=float, default=0.98, help="Ratio for train/valid split.")
    
    # 학습 파라미터
    parser.add_argument("--output_dir", type=str, help="Output directory.")
    parser.add_argument("--logging_steps", type=int, help="Logging steps.")
    parser.add_argument("--num_train_epochs", type=int, help="Number of epochs.")
    parser.add_argument("--learning_rate", type=float, help="Learning rate.")
    parser.add_argument("--gradient_accumulation_steps", type=int, help="Gradient accumulation steps.")
    parser.add_argument("--lr_scheduler_type", type=str, default="linear", help="Learning rate scheduler type.")
    parser.add_argument("--per_device_train_batch_size", type=int, help="Training batch size per device.")
    parser.add_argument("--bf16", type=bool, default=True, help="Whether to use bf16 training.")
    parser.add_argument("--run_name", type=str, help="Name for the run in W&B or logs.")
    parser.add_argument("--save_steps", type=int, help="Interval steps for saving checkpoints.")
    
    parser.add_argument("--train_label", type=str, 
                       choices=["prm_soft_label", "prm_hard_label", "er_label", "gemini_label", "llama_label"],
                       help="Which training label to use. 'er_label' will be computed from soft label with risk_param.")
    parser.add_argument("--risk_param", type=float,
                       help="Entropic risk parameter (mu) for converting soft labels to ER labels.")
    
    # ✨ 필터링 여부 추가 ✨
    parser.add_argument("--do_filtering", type=str, default="yes",
                       help="Whether to perform filtering for orm_label=0/1. 'yes' or 'no'")
    
    return parser.parse_args()


def load_model(model_name="meta-llama/Llama-3.1-8B", dtype="bfloat16"):
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if dtype == "bfloat16" else torch.float16,
        attn_implementation="flash_attention_2",
        device_map="auto"
    )
    model.gradient_checkpointing_enable()
    
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
        
        # 라벨 텍스트
        raw_label = raw_text.replace(" ки", "<good_step>")
        reference_labels = tokenizer(raw_label, add_special_tokens=True)['input_ids']
        reference_labels.insert(0, tokenizer.pad_token_id)
        reference_labels[0] = -100
        
        ann = entry.get('annotation', [])
        value_list = []
        counter = 0
        
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
    return (1.0 / mu) * math.log((1.0 - p) + p * math.exp(mu))


def process_gemini_label(label_list):
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
    question_text = item.get("question", "")
    options = item.get("options", [])
    if not options:
        return question_text
    formatted_options = ""
    for idx, opt in enumerate(options):
        letter = chr(ord('A') + idx)
        formatted_options += f" ({letter}) {opt}"
    return question_text + formatted_options


def load_json_and_create_dataset_from_data(data, tokenizer, step_tag_id=128256, args=None):
    all_results = []
    for idx, item in enumerate(data):
        question_id = item.get("question_id", f"question_{idx}")
        formatted_question = format_question_with_options(item)
        solutions = item.get("solutions", [])
        
        if not solutions:
            continue
        
        for sol_idx, sol in enumerate(solutions):
            # (1) label 타입별 처리
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
            elif args.train_label == "llama_label":
                original_llama = sol.get("prm_llama_label", [])
                if not isinstance(original_llama, list):
                    original_llama = [original_llama]
                new_label = process_gemini_label(original_llama)
            else:
                new_label = sol.get(args.train_label, [])
                if not isinstance(new_label, list):
                    new_label = [new_label]
            
            solution_text = sol.get("prm_processed_solution", "")
            raw_text = f"{formatted_question}\n{solution_text}"
            
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


# ✨ 변경된 split_json_data: do_filtering 인자를 받아 필터링 수행 ✨
def split_json_data(json_path, train_ratio=0.98, args=None):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    random.shuffle(data)
    split_idx = int(len(data) * train_ratio)
    train_data = data[:split_idx]
    valid_data = data[split_idx:]
    
    # do_filtering == "yes"일 때만 필터링 로직 수행
    if args and args.do_filtering.lower() == "yes":
        for item in train_data:
            if "solutions" not in item:
                continue
            
            solutions = item["solutions"]
            orm_0_solutions = []
            orm_1_solutions = []
            
            for sol in solutions:
                orm_label_val = sol.get("orm_label", 0)
                if orm_label_val == 0:
                    # ER 레이블이면 0/1 체크 제외
                    if args.train_label == "er_label":
                        orm_0_solutions.append(sol)
                        continue
                    
                    # gemini/hard/soft/llama 구분
                    if args.train_label == "gemini_label":
                        arr = sol.get("prm_gemini_label", [])
                        if not isinstance(arr, list):
                            arr = [arr]
                        arr = process_gemini_label(arr)
                    elif args.train_label == "llama_label":
                        arr = sol.get("prm_llama_label", [])
                        if not isinstance(arr, list):
                            arr = [arr]
                        arr = process_gemini_label(arr)
                    else:
                        arr = sol.get(args.train_label, [])
                        if not isinstance(arr, list):
                            arr = [arr]
                    
                    # 0이 하나라도 있으면 남김
                    if any(x == 0 for x in arr):
                        orm_0_solutions.append(sol)
                    else:
                        # 전부 1이면 제거
                        pass
                else:
                    orm_1_solutions.append(sol)
            
            # 0 솔루션 수만큼 1 솔루션을 남기는데, 최소 2개는 남김
            remain_0_count = len(orm_0_solutions)
            need_1_count = max(remain_0_count, 2)
            keep_1_solutions = orm_1_solutions[:need_1_count]
            
            item["solutions"] = orm_0_solutions + keep_1_solutions
    
    return train_data, valid_data


class AutoRegressiveTrainer(Trainer):
    def __init__(self, my_tokenizer=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.my_tokenizer = my_tokenizer
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs["logits"]
        
        logits = logits[..., :-1, :].contiguous().to(torch.bfloat16)
        labels = labels[..., 1:].contiguous()
        values = inputs['values'][..., 1:].contiguous().to(torch.bfloat16)
        
        plus_id = self.my_tokenizer.encode('<good_step>')[-1]
        minus_id = self.my_tokenizer.encode('<bad_step>')[-1]
        
        plus_logits = logits[:, :, plus_id]
        minus_logits = logits[:, :, minus_id]
        
        if labels.dim() == plus_logits.dim() - 1:
            labels = labels.unsqueeze(-1)
            values = values.unsqueeze(-1)
        
        chosen = (labels != -100)
        pred_plus_values = plus_logits[chosen]
        pred_minus_values = minus_logits[chosen]
        gt_values = values[chosen]
        
        pred_combined = torch.stack((pred_plus_values, pred_minus_values), dim=1)
        gt_negative = 1 - gt_values
        gt_combined = torch.stack((gt_values, gt_negative), dim=1)
        
        loss = torch.nn.functional.cross_entropy(
            pred_combined,
            gt_combined,
            reduction="mean"
        )
        loss = loss.to(torch.bfloat16)
        return (loss, outputs) if return_outputs else loss


def main():
    args = parse_args()
    
    # GPU 설정
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # 1. 로그인
    login(args.hf_token)
    wandb.login(key=args.wandb_token)
    print("1: Login Success")

    # 2. 모델 불러오기
    model, tokenizer = load_model(model_name=args.model_path, dtype=args.dtype)
    print("2: Model Loading Success")

    # 스페셜 토큰 추가
    special_tokens = {"additional_special_tokens": [" ки", "<good_step>", "<bad_step>"]}
    tokenizer.add_special_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))

    step_tag = 'ки'
    step_tag_id = tokenizer.encode(f" {step_tag}")[-1]
    plus_tag_id = tokenizer.encode('<good_step>')[-1]
    minus_tag_id = tokenizer.encode('<bad_step>')[-1]

    print("Step tag ID:", step_tag_id)
    print("Good step ID:", plus_tag_id)
    print("Bad step  ID:", minus_tag_id)

    # 3. 데이터 준비 (필터링 포함)
    train_data, valid_data = split_json_data(args.train_json, train_ratio=args.train_ratio, args=args)
    train_dict = load_json_and_create_dataset_from_data(train_data, tokenizer, step_tag_id=step_tag_id, args=args)
    valid_dict = load_json_and_create_dataset_from_data(valid_data, tokenizer, step_tag_id=step_tag_id, args=args)

    train_dataset = Dataset.from_dict(train_dict)
    valid_dataset = Dataset.from_dict(valid_dict)

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
    valid_dataset = valid_dataset.map(lambda x: pad_sequences(x, max_length=1023))

    print("3: Data Pre-processing Success")

    # 4. Trainer 설정
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model_name = args.model_path.split("/")[-1]
    specific_output_dir = f"{args.output_dir}"
    os.makedirs(specific_output_dir, exist_ok=True)
    
    if not args.run_name:
        args.run_name = f"finetune_{model_name}_{args.train_label}_{args.learning_rate}"

    training_args = TrainingArguments(
        output_dir=specific_output_dir,
        eval_strategy="epoch",
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=1,
        num_train_epochs=args.num_train_epochs,
        save_strategy="epoch",
        logging_steps=args.logging_steps,
        save_total_limit=10,
        load_best_model_at_end=False,
        metric_for_best_model="loss",
        greater_is_better=False,
        run_name=args.run_name,
        bf16=args.bf16,
        remove_unused_columns=False,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        report_to=["wandb"],
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=0.05,
        ddp_find_unused_parameters=False,
    )

    finetuner = AutoRegressiveTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        my_tokenizer=tokenizer,
        args=training_args,
        data_collator=data_collator,
    )

    print("4: Finetuning Ready")

    # 5. Fine-tuning
    print(f"Fine-tuning 시작... (learning rate: {args.learning_rate}, risk_param(mu): {args.risk_param}, do_filtering: {args.do_filtering})")
    finetuner.train()
    print("Fine-tuning 완료!")

    finetuner.save_model(specific_output_dir)
    tokenizer.save_pretrained(specific_output_dir)
    print(f"모델이 {specific_output_dir}에 저장되었습니다.")

    # 6. Hugging Face Hub 업로드
    date_str = datetime.now().strftime("%Y%m%d")
    base_model_name = args.model_path.split("/")[-1]
    
    # filtering 문자열
    filter_str = "filter" if args.do_filtering.lower() == "yes" else "nofilter"
    # repo 이름: "user/날짜-base_model-라벨-filtering-에폭-lr"
    repo_name = f"jhyun0414/{date_str}-{base_model_name}-{args.train_label}-{filter_str}-e{args.num_train_epochs}-lr{args.learning_rate}"
    
    create_repo(repo_name, exist_ok=True, private=False)
    finetuner.model.push_to_hub(repo_name)
    tokenizer.push_to_hub(repo_name)
    print(f"🚀 모델 업로드 완료! 🔗 확인: https://huggingface.co/{repo_name}")

    # 7. 업로드 후 로컬 디렉터리 삭제(용량 절약)
    try:
        shutil.rmtree(specific_output_dir)
        print(f"로컬 디렉터리({specific_output_dir})가 성공적으로 삭제되었습니다.")
    except Exception as e:
        print(f"디렉터리 삭제 중 오류 발생: {e}")


if __name__ == "__main__":
    main()
