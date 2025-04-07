#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import json
import re
import torch
import numpy as np
from huggingface_hub import login
from vllm import LLM, SamplingParams

##############################################################################
# (1) 공통 함수 (스텝 추출 및 partial solution 생성 등)
##############################################################################

def extract_steps_from_text(generated_text):
    """
    'Step N:' 형태로 분할한 스텝들을 리스트로 반환
    """
    step_pattern = r'(?:## )?Step \d+:'
    matches = list(re.finditer(step_pattern, generated_text))

    if not matches:
        return [generated_text.strip()] if generated_text.strip() else []

    steps = []
    for i in range(len(matches)):
        start_idx = matches[i].start()
        end_idx = matches[i + 1].start() if i + 1 < len(matches) else len(generated_text)
        step_text = generated_text[start_idx:end_idx].strip().replace("## ", "")
        steps.append(step_text)
    return steps

def extract_answer_from_text(text):
    """
    최종 '정답 알파벳' 추출
    (solution_sampling.py와 동일)
    """
    text = text.strip().lower()
    if "\\boxed{" in text:
        match = re.findall(r'\\boxed\{\(?([a-z])\)?\}', text)
        if match:
            return match[-1].upper()

    pattern = re.finditer(
        r'(?:the final answer is|the answer is|final answer is|answer is)\s*:?\s*\(?([a-z])\)?',
        text, re.IGNORECASE
    )
    matches = list(pattern)
    if matches:
        return matches[-1].group(1).upper()

    pattern2 = re.finditer(
        r'swer is\s*:?\s*\(?([a-z])\)?',
        text, re.IGNORECASE
    )
    matches2 = list(pattern2)
    if matches2:
        return matches2[-1].group(1).upper()

    return None

##############################################################################
# (2) PRM 레이블 계산 함수들
##############################################################################

def calculate_prm_soft_label(annotations):
    """
    annotations: 2차원 리스트 (각 스텝별 [score1, score2, ...])
    -> soft label = 평균 스코어
    """
    if not annotations:
        return []
    prm_soft_label = []
    for step_scores in annotations:
        avg_score = np.mean(step_scores)
        prm_soft_label.append(avg_score)
    return prm_soft_label

def calculate_prm_hard_label(annotations):
    """
    annotations: 2차원 리스트 (각 스텝별 [score1, score2, ...])
    -> hard label = 1 if any score>0 else 0 (스텝별)
    """
    if not annotations:
        return []
    prm_hard_label = []
    for step_scores in annotations:
        hard_label = 1 if any(s > 0 for s in step_scores) else 0
        prm_hard_label.append(hard_label)
    return prm_hard_label

##############################################################################
# (3) Partial solution 평가 (중간 스텝별 정답) + PRM 레이블 추가
##############################################################################

def annotate_partial_solutions(all_data, completion_number, llm, sampling_params, output_file):
    """
    1) 'all_data'는 solution_sampling.py에서 만든 리스트 구조
    2) 각 question -> 여러 solutions -> 중간 스텝별 partial solution
       -> completion_number번씩 chat 생성 -> 정답률 계산
    3) 'prm_soft_label', 'prm_hard_label'를 각 solution에 붙여 저장
    """
    # system 메시지 (중간 스텝 평가 시에도 동일한 지침)
    SYSTEM_PROMPT = (
        "Solve the following question step-by-step. "
        "Do not analyze individual options in a single step. "
        "Each step must start with 'Step {number}:'. "
        "Include the final answer as 'the answer is (option alphabet)'."
    )

    # (A) partial 솔루션 평가에 필요한 프롬프트 수집
    partial_prompts = []
    partial_meta = []

    for q_idx, qitem in enumerate(all_data):
        question_id = qitem["question_id"]
        question_text = qitem["question"]
        correct_answer = qitem["correct_answer"]
        solutions = qitem["solutions"]

        for sol_idx, sol in enumerate(solutions):
            sol_text = sol["solution"]
            steps = extract_steps_from_text(sol_text)
            # 중간 스텝(마지막 제외)이 1개 이상
            if len(steps) > 1:
                # 마지막 전까지 반복
                for step_i in range(len(steps) - 1):
                    partial_text = " ".join(steps[:(step_i+1)])  # 해당 스텝까지 누적
                    messages = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": question_text + "\n\n" + partial_text}
                    ]
                    partial_prompts.append(messages)
                    partial_meta.append({
                        "question_idx": q_idx,
                        "solution_idx": sol_idx,
                        "step_idx": step_i,
                        "correct_answer": correct_answer
                    })

    # (B) partial_prompts를 completion_number번씩 반복 -> LLM 호출
    scores_map = {}  # (q_idx, sol_idx, step_idx) -> list of scores

    if partial_prompts:
        repeated_prompts = []
        extended_meta = []
        for meta, msg in zip(partial_meta, partial_prompts):
            for _ in range(completion_number):
                repeated_prompts.append(msg)
            extended_meta.extend([meta]*completion_number)

        outputs = llm.chat(repeated_prompts, sampling_params)

        # (C) 생성된 최종 답안 -> 정답 여부 (0/1)
        for i, out in enumerate(outputs):
            generated_text = out.outputs[0].text
            meta = extended_meta[i]
            ans = extract_answer_from_text(generated_text)
            score = 1 if (ans == meta["correct_answer"]) else 0

            key = (meta["question_idx"], meta["solution_idx"], meta["step_idx"])
            if key not in scores_map:
                scores_map[key] = []
            scores_map[key].append(score)

    # (D) 이제 각 솔루션별로 annotations를 만들어 PRM 레이블 추가
    for q_idx, qitem in enumerate(all_data):
        for sol_idx, sol in enumerate(qitem["solutions"]):
            sol_text = sol["solution"]
            steps = extract_steps_from_text(sol_text)
            if len(steps) <= 1:
                # 중간 스텝 없으므로 전부 0
                sol["prm_soft_label"] = []
                sol["prm_hard_label"] = []
                continue

            # 각 스텝별 점수 목록
            # 마지막 스텝은 이미 solution_sampling.py에서 'answer'로 저장,  
            # 그건 ORM(최종) 이었으므로, 여기서는 중간 스텝+마지막 스텝을 합쳐서 봄
            step_scores_list = []
            for step_i in range(len(steps) - 1):
                key = (q_idx, sol_idx, step_i)
                sc_list = scores_map.get(key, [])
                step_scores_list.append(sc_list)

            # 마지막 스텝은 이미 sol["answer"]와 correct_answer 비교
            final_ans = sol.get("answer", "N/A")
            final_score = 1 if final_ans == qitem["correct_answer"] else 0
            # 여러 번 샘플링하지 않았으므로 completion_number개로 동일하게 복제
            final_scores = [final_score]*completion_number
            step_scores_list.append(final_scores)

            # PRM soft/hard
            soft_label = calculate_prm_soft_label(step_scores_list)
            hard_label = calculate_prm_hard_label(step_scores_list)

            sol["prm_soft_label"] = soft_label
            sol["prm_hard_label"] = hard_label

    # (E) 최종 저장
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_data, f, indent=4, ensure_ascii=False)
        print(f"[INFO] Saved final annotation to: {output_file}")
    except Exception as e:
        print(f"[ERROR] Failed to save JSON: {e}")

    return all_data


##############################################################################
# (4) 메인 함수
##############################################################################

def parse_args():
    parser = argparse.ArgumentParser(description="(Step 2) Annotate partial solutions (PRM).")
    parser.add_argument("--hf_token", type=str, required=True, help="Hugging Face access token.")
    parser.add_argument("--model_path", type=str, default="meta-llama/Llama-3.2-3B-Instruct",
                        help="Path or name of the LLM model.")
    parser.add_argument("--completion_number", type=int, default=8,
                        help="Number of times to sample partial solutions.")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature.")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k value.")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-P value.")
    parser.add_argument("--max_tokens", type=int, default=512, help="Max tokens for generation.")
    parser.add_argument("--input_file", type=str, required=True,
                        help="JSON file from 'solution_sampling.py'.")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Output JSON with PRM annotation.")
    parser.add_argument("--gpu_ids", type=str, default="0", help="GPU device IDs.")
    return parser.parse_args()

def main():
    args = parse_args()
    login(args.hf_token)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    device_count = torch.cuda.device_count()
    device = torch.device("cuda" if device_count > 0 else "cpu")
    print(f"[INFO] Device: {device} (GPUs={device_count}), GPU IDs={args.gpu_ids}")

    from vllm import LLM, SamplingParams
    llm = LLM(
        model=args.model_path,
        device="cuda",
        dtype="bfloat16",
        max_model_len=args.max_tokens,
        gpu_memory_utilization=0.8,
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        max_tokens=args.max_tokens
    )

    # 1) solution_sampling.py에서 만든 JSON 읽기
    with open(args.input_file, "r", encoding="utf-8") as f:
        all_data = json.load(f)
    # all_data: [ {question_id, question, correct_answer, accuracy, solutions=[...]} , ... ]

    # 2) partial solutions 평가 + PRM 레이블 추가
    final_data = annotate_partial_solutions(
        all_data=all_data,
        completion_number=args.completion_number,
        llm=llm,
        sampling_params=sampling_params,
        output_file=args.output_file
    )

    print("[INFO] Done. (solution_annotation)")

if __name__ == "__main__":
    main()
