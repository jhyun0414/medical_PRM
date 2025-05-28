#!/usr/bin/env python
# coding: utf-8
"""
Fine-tune LLM on PRM dataset (MedQA only) with optional RAG support.
"""

# =====================================================================
# 1. 라이브러리 임포트
# =====================================================================
import argparse
import torch
import json
import random
import os
import math
from datetime import datetime
import socket

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from datasets import Dataset

# --- wandb/huggingface_hub 온라인 기능용 패키지 (optional import) ----
try:
    import wandb
    from huggingface_hub import HfApi, create_repo
except ImportError:
    wandb = None
    HfApi = None

# =====================================================================
# 2. 인자 파서
# =====================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune LLM model on PRM dataset (MedQA only)."
    )

    # 모델 관련
    parser.add_argument("--model_path", type=str, help="Path or name of the LLM model.")
    parser.add_argument("--device", type=str, help="CUDA visible devices (e.g. '0,1')")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Data type (default: bfloat16).")
    parser.add_argument("--max_token_len", type=int, default=1024, help="최대 토큰 길이 (기본값: 1023).")

    # 데이터 관련
    parser.add_argument("--train_json", type=str, help="Path to the training JSON file.")
    parser.add_argument("--train_ratio", type=float, default=1.0, help="Ratio for train/valid split (default: 1.0).")

    # 학습 하이퍼파라미터
    parser.add_argument("--output_dir", type=str, help="Output directory.")
    parser.add_argument("--logging_steps", type=int, help="Logging steps.")
    parser.add_argument("--num_train_epochs", type=int, help="Number of epochs.")
    parser.add_argument("--learning_rate", type=float, help="Learning rate.")
    parser.add_argument("--gradient_accumulation_steps", type=int, help="Gradient accumulation steps.")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", help="Learning rate scheduler type.")
    parser.add_argument("--per_device_train_batch_size", type=int, help="Training batch size per device.")
    parser.add_argument("--bf16", type=bool, default=True, help="Whether to use bf16 training.")
    parser.add_argument("--run_name", type=str, help="Name for the run in logs.")
    parser.add_argument("--save_steps", type=int, help="Interval steps for saving checkpoints.")

    # 라벨 종류
    parser.add_argument("--train_label", type=str, 
                        choices=["prm_soft_label", "prm_hard_label", "gemini_label", "llama_label", "orm_label"],
                        help="Which training label to use.")

    # 필터링
    parser.add_argument("--do_filtering", type=str, default="yes",
                        help="Whether to perform filtering for orm_label=0/1. 'yes' or 'no'")

    # RAG 사용 여부 (10_training_code.py 고유 옵션)
    parser.add_argument("--use_rag", type=str, default="no",
                        choices=["yes", "no"],
                        help="Include related docs (RAG) if 'yes'.")

    # 온라인 관련
    parser.add_argument("--online", type=bool, default=False, help="온라인 모드 활성화 여부 (True/False)")
    parser.add_argument("--wandb_token", type=str, help="Wandb API token for online mode.")
    parser.add_argument("--wandb_project", type=str, help="Wandb project name for online mode.")
    parser.add_argument("--hf_token", type=str, help="HuggingFace Hub token for online mode.")

    return parser.parse_args()

# =====================================================================
# 3. 모델·토크나이저 로더
# =====================================================================
def load_model(model_name="meta-llama/Llama-3.1-8B-Instruct", dtype="bfloat16"):
    print("Loading model...")
    print(f"Model being loaded: {model_name}")  # 디버깅용 출력 추가
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"Tokenizer loaded: {tokenizer.__class__.__name__}")  # 토크나이저 정보 출력
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if dtype == "bfloat16" else torch.float16,
        attn_implementation="flash_attention_2",
        device_map="auto"
    )
    print(f"Model loaded: {model.__class__.__name__}, Parameters: {model.num_parameters():,}")  # 모델 정보 출력
    model.gradient_checkpointing_enable()
    
    # pad_token이 없는 모델(예: LLaMA 계열)을 위해 처리
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Set pad_token to eos_token: {tokenizer.pad_token}")  # pad_token 설정 정보
    
    print("Model loaded successfully.")
    return model, tokenizer

# =====================================================================
# 4. 유틸리티 함수
# =====================================================================
def process_gemini_label(label_list):
    processed, found_zero = [], False
    for v in label_list:
        processed.append(0 if found_zero else v)
        if v == 0:  # 이후 모두 0
            found_zero = True
    return processed

# --- 추가: 관련 문서 토큰 예산 내에서 자르기 --------------------------
def truncate_related_docs(docs, tokenizer,
                          max_total_len: int,
                          reserve_for_prompt: int = 1024):
    """
    docs: list[str] (raw docs)
    max_total_len: 전체 토큰 상한
    reserve_for_prompt: 질문/해설/시스템 프롬프트를 위해 남겨둘 토큰 수
    """
    kept, used = [], 0
    budget = max_total_len - reserve_for_prompt
    for d in docs:
        dtok = len(tokenizer(d, add_special_tokens=False)["input_ids"])
        if used + dtok + 1 > budget:
            break
        kept.append(d)
        used += dtok + 1
    return kept

def format_question_with_options(item):
    q = item.get("question", "")
    opts = item.get("options", [])
    if not opts:
        return q
    formatted = "".join(f" ({chr(ord('A')+i)}) {o}" for i, o in enumerate(opts))
    return q + formatted

# =====================================================================
# 5. JSON → Dataset 변환
# =====================================================================
def process_all_results_to_dataset(all_results, tokenizer, step_tag_id=128256):

    data = {"input_ids": [], "attention_mask": [], "labels": [], "values": []}
    errors = 0

    for entry in all_results:
        raw = entry["query"].replace(" ки\n", " ки")  # 줄바꿈 정리
        enc = tokenizer(raw, add_special_tokens=True, truncation=True)

        ids = enc["input_ids"] + [tokenizer.pad_token_id]
        attn = enc["attention_mask"] + [0]

        raw_label = raw.replace(" ки", " +")
        ref = tokenizer(raw_label, add_special_tokens=True)["input_ids"]
        ref.insert(0, tokenizer.pad_token_id)
        ref[0] = -100

        vals, cnt, ann = [], 0, entry["annotation"]
        if len(ref) != len(ids):
            errors += 1
            continue

        skip = False
        for j in range(len(ref)):
            if j == 0:
                vals.append(0)
                continue
            if ids[j - 1] == step_tag_id:
                if cnt < len(ann):
                    vals.append(ann[cnt]); cnt += 1
                else:
                    errors += 1; skip = True; break
            else:
                vals.append(0); ref[j] = -100
        if skip:  # 주석 길이 부족
            continue

        data["input_ids"].append(ids)
        data["attention_mask"].append(attn)
        data["labels"].append(ref)
        data["values"].append(vals)

    print(f"⚠️  오류로 건너뜀: {errors} solution(s)")
    return data



def build_dataset_from_json(data, tokenizer, step_tag_id: int, args):
    """
    MedQA 샘플을 LLM 입력으로 변환
    """
    RAG_SYSTEM_PROMPT = (
    "You are an evaluator assessing the logicality and validity of the reasoning in each step of the given explanation. "
    "In order to support the evaluation, the relevant documents, the question, and the explanation are provided sequentially. "
    "If the reasoning contains errors, output - after that step. If the reasoning in a step is logical and valid, output + after that step."
            )
    ORM_SYSTEM_PROMPT = (
        "You are an evaluator assessing the overall quality and correctness of the final answer in the given explanation. "
        "In order to support the evaluation, the question and the explanation are provided. "
        "If the final answer is incorrect or not well-supported, output -. If the final answer is correct and well-supported, output +."
            )
    PRM_SYSTEM_PROMPT = (
        "You are an evaluator assessing the logicality and validity of the reasoning in each step of the given explanation. "
        "In order to support the evaluation, the question and the explanation are provided. "
        "If the reasoning contains errors, output - after that step. If the reasoning in a step is logical and valid, output + after that step."
            )
    if args.use_rag == "yes":
        SYSTEM_PROMPT = RAG_SYSTEM_PROMPT
    elif args.train_label == "orm_label":
        SYSTEM_PROMPT = ORM_SYSTEM_PROMPT
    else:
        SYSTEM_PROMPT = PRM_SYSTEM_PROMPT

    all_results = []

    for idx, item in enumerate(data):
        qid = item.get("question_id", f"q_{idx}")
        q_formatted = format_question_with_options(item)
        sols = item.get("solutions", [])
        if not sols:
            continue

        # --- 수정: RAG 문서 처리 --------------------------------------
        if args.use_rag == "yes":
            docs_kept = truncate_related_docs(
                item.get("related_docs", []),
                tokenizer,
                max_total_len=args.max_token_len,
                reserve_for_prompt=1024,
            )
            doc_block = "".join(
                f"Document {i+1}: {d}\n\n" for i, d in enumerate(docs_kept)
            )
        else:
            doc_block = ""

        for sidx, sol in enumerate(sols):

            # 라벨 추출 ------------------------------------------------
            if args.train_label == "gemini_label":
                arr = sol.get("prm_gemini_label", [])
                arr = arr if isinstance(arr, list) else [arr]
                label = process_gemini_label(arr)
            elif args.train_label == "llama_label":
                arr = sol.get("prm_llama_label", [])
                arr = arr if isinstance(arr, list) else [arr]
                label = process_gemini_label(arr)
            elif args.train_label == "orm_label":
                label = sol.get("orm_label", [])
                label = label if isinstance(label, list) else [label]
            else:
                label = sol.get(args.train_label, [])
                label = label if isinstance(label, list) else [label]

            # 솔루션 텍스트 선택: ORM 라벨인 경우와 그 외의 경우
            if args.train_label == "orm_label":
                sol_text = sol.get("orm_processed_solution", "")
            else:
                sol_text = sol.get("prm_processed_solution", "")

            # 입력 템플릿 --------------------------------------------
            content = (
                f"{doc_block}Question: {q_formatted}\n\nExplanation: {sol_text}"
            )
            msgs = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ]
            raw_text = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )

            all_results.append(
                {
                    "query": raw_text,
                    "annotation": label,
                    "question_id": qid,
                    "solution_index": sidx,
                }
            )

    return process_all_results_to_dataset(
        all_results, tokenizer, step_tag_id=step_tag_id
    )

# =====================================================================
# 6. JSON 분할 (MedQA 필터 유지)
# =====================================================================
def split_json_data(json_path, args):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 원래 코드: MedQA만 필터링
    # medqa = [d for d in data if d.get("data_source") == "med_qa"]
    # print(f"전체: {len(data)} / MedQA: {len(medqa)}")
    
    # 전체 데이터 사용
    print(f"전체 데이터 수: {len(data)}")

    # do_filtering == "yes"일 때만 필터링 로직 수행
    if args.do_filtering.lower() == "yes":
        for item in data:  # medqa 대신 data 사용
            if "solutions" not in item:
                continue
            
            solutions = item["solutions"]
            
            orm_0_solutions = []
            orm_1_solutions = []
            
            for sol in solutions:
                orm_label_val = sol.get("orm_label", 0)
                if orm_label_val == 0:
                    # er_label이면 0/1 체크 안함
                    if args.train_label == "er_label":
                        orm_0_solutions.append(sol)
                        continue
                    
                    # gemini/hard/soft/llama
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
                        # soft/hard
                        arr = sol.get(args.train_label, [])
                        if not isinstance(arr, list):
                            arr = [arr]
                    
                    if any(x == 0 for x in arr):
                        orm_0_solutions.append(sol)
                    else:
                        # 전부 1이면 제거
                        pass
                else:
                    orm_1_solutions.append(sol)
            
            remain_0_count = len(orm_0_solutions)
            need_1_count = max(remain_0_count, 2)
            
            keep_1_solutions = orm_1_solutions[:need_1_count]
            
            item["solutions"] = orm_0_solutions + keep_1_solutions

    # train_ratio 무시하고 전체 데이터를 train으로 사용
    return data, []  # medqa 대신 data 반환

# =====================================================================
# 7. 커스텀 Trainer (loss 동일)
# =====================================================================
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
        
        plus_id = self.my_tokenizer.encode(' +')[-1]
        minus_id = self.my_tokenizer.encode(' -')[-1]
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

# =====================================================================
# 8. 메인
# =====================================================================
def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # ---- 온라인 여부 판단 -----------------------------------------
    # args.online이 bool 타입이 아닐 경우 문자열로 처리
    if isinstance(args.online, str):
        online = args.online.lower() == 'true'
    else:
        online = bool(args.online)
    
    print(f"🌐 Online mode: {online}")

    # ---- W&B 초기화 (온라인 시) ------------------------------------
    if online:
        os.environ["WANDB_API_KEY"] = args.wandb_token or ""
        wandb.login(key=args.wandb_token, relogin=True)
        wandb.init(project=getattr(args, "wandb_project", "prm-finetune"),
                   name=args.run_name or "prm-finetune",
                   config=vars(args))

    # 1) 모델 로드 ---------------------------------------------------
    print(f"DEBUG: 모델 로드 시작 - 모델 경로: {args.model_path}")  # 디버깅 출력 추가
    model, tokenizer = load_model(
        model_name=args.model_path or "meta-llama/Llama-3.1-8B-Instruct",  # args.model_path 사용
        dtype=args.dtype
    )
    print(f"DEBUG: 모델 로드 완료 - 모델 타입: {type(model).__name__}")  # 디버깅 출력 추가

    # 2) 추가 스페셜 토큰 -------------------------------------------
    special_tokens = {"additional_special_tokens": [" ки"]}
    tokenizer.add_special_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))

    step_tag = "ки"
    step_tag_id = tokenizer.encode(f" {step_tag}")[-1]
    print("Step tag id:", step_tag_id)

    # 3) 데이터 로딩 -------------------------------------------------
    train_raw, valid_raw = split_json_data(args.train_json, args=args)
    train_dict = build_dataset_from_json(train_raw, tokenizer, step_tag_id, args)
    train_ds = Dataset.from_dict(train_dict)

    # 유효성 검증을 사용하지 않을 경우
    if len(valid_raw) == 0:
        valid_ds = None
    else:
        valid_dict = build_dataset_from_json(valid_raw, tokenizer, step_tag_id, args)
        valid_ds = Dataset.from_dict(valid_dict)

    # 4) 패딩/컷 ----------------------------------------------------
    def pad_fn(ex, max_len=args.max_token_len, pad_val=0):
        iid, attn, lab, val = ex["input_ids"], ex["attention_mask"], ex["labels"], ex["values"]
        if len(iid) < max_len:
            pad = max_len - len(iid)
            iid += [pad_val] * pad
            attn += [0] * pad
            lab += [-100] * pad
            val += [0] * pad
        else:
            iid, attn, lab, val = iid[:max_len], attn[:max_len], lab[:max_len], val[:max_len]
        return {
            "input_ids": iid,
            "attention_mask": attn,
            "labels": lab,  
            "values": val,
        }

    train_ds = train_ds.map(pad_fn)
    if valid_ds is not None:
        valid_ds = valid_ds.map(pad_fn)

    # 5) Trainer ----------------------------------------------------
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    run_name = (
        args.run_name
        or f"{os.path.basename(args.model_path)}_{args.train_label}_{args.learning_rate}"
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="no",  # --- 수정: MedQA only / no eval ----
        per_device_train_batch_size=args.per_device_train_batch_size,
        num_train_epochs=args.num_train_epochs,
        save_strategy="epoch",
        logging_steps=args.logging_steps,
        save_total_limit=1,
        run_name=run_name,
        bf16=args.bf16,
        remove_unused_columns=False,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        report_to=["wandb"] if online else ["none"],
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=0.05,
    )

    trainer = AutoRegressiveTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,  # None 이면 자동으로 평가 생략
        data_collator=data_collator,
        my_tokenizer=tokenizer,
    )

    # 6) 학습 -------------------------------------------------------
    print(
        f"● Fine-tuning start | lr={args.learning_rate} | filter={args.do_filtering} "
        f"| RAG={args.use_rag} | μ={args.risk_param}"
    )
    trainer.train()
    print("✅ Fine-tuning finished")

    # 7) 저장 -------------------------------------------------------
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    date_str = datetime.now().strftime("%Y%m%d")
    filter_str = "filter" if args.do_filtering == "yes" else "nofilter"
    rag_str = "rag" if args.use_rag == "yes" else "norag"
    save_tag = f"{date_str}-{os.path.basename(args.model_path)}-{args.train_label}-{filter_str}-{rag_str}-e{args.num_train_epochs}"
    print(f"🚀 Saved: {args.output_dir}/{save_tag}")

    # ---- 8) 온라인 시 HF Hub 업로드 -------------------------------
    if online:
        os.environ["HF_HUB_TOKEN"] = args.hf_token
        api = HfApi(token=args.hf_token)
        repo_name = f"{api.whoami()['name']}/{save_tag}"
        try:
            create_repo(repo_name, exist_ok=True, private=False)
            trainer.model.push_to_hub(repo_name)
            tokenizer.push_to_hub(repo_name)
            print(f"🚀 Pushed to HF Hub: https://huggingface.co/{repo_name}")
        except Exception as e:
            print(f"HF 업로드 실패: {e}")

    if online and wandb.run is not None:
        wandb.finish()

    print("🎉 작업 완료!")

if __name__ == "__main__":
    main()