#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import json
import re
import torch
import string
from huggingface_hub import login
from vllm import LLM, SamplingParams

##############################################################################
# (1) 공통 유틸 함수 (문항 포맷팅 / 정답 추출 / 스텝 분할 / 후처리)
##############################################################################

def format_question(qdata):
    """
    question_id, full_question, correct_letter 형태로 변환.
    - ground_truth -> 올바른 알파벳(correct_letter)으로 매핑
    """
    question_id = int(qdata.get("question_id", "0"))
    question_str = qdata["question"]
    options = qdata["options"]

    # (A), (B), (C), ...
    option_labels = [f"({letter})" for letter in string.ascii_uppercase[:len(options)]]
    formatted_options = "\n".join(
        f"{label} {option}" for label, option in zip(option_labels, options)
    )
    full_question = f"{question_str}\n\n{formatted_options}"

    # ground_truth(옵션 텍스트)에 해당하는 알파벳 찾기
    gt_text = qdata["ground_truth"].strip().lower()
    correct_letter = None
    for i, option in enumerate(options):
        if option.strip().lower() == gt_text:
            correct_letter = string.ascii_uppercase[i]  # 0->A, 1->B...
            break
    if correct_letter is None:
        correct_letter = "unknown"

    return question_id, full_question, correct_letter

def extract_answer_from_text(text):
    """
    생성된 솔루션(문자열)에서 최종 '정답 알파벳'을 추출하는 함수.
    """
    text = text.strip().lower()

    # 1) LaTeX 수식 \boxed{} 내부 알파벳
    if "\\boxed{" in text:
        match = re.findall(r'\\boxed\{\(?([a-z])\)?\}', text)
        if match:
            return match[-1].upper()

    # 2) "the answer is x", "final answer is x" ...
    pattern = re.finditer(
        r'(?:the final answer is|the answer is|final answer is|answer is)\s*:?\s*\(?([a-z])\)?',
        text, re.IGNORECASE
    )
    matches = list(pattern)
    if matches:
        return matches[-1].group(1).upper()

    # 3) "swer is x" (타이포 대응)
    pattern2 = re.finditer(
        r'swer is\s*:?\s*\(?([a-z])\)?',
        text, re.IGNORECASE
    )
    matches2 = list(pattern2)
    if matches2:
        return matches2[-1].group(1).upper()

    return None  # 찾지 못한 경우

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

def prm_process_solution(input_text):
    """
    스텝 분할 후 각 스텝 뒤에 ' ки'를 덧붙여주는 처리(데이터셋 후처리에 사용).
    """
    text_no_newlines = input_text.replace("\n", " ")
    step_pattern = r'(?:## )?Step \d+:'
    matches = list(re.finditer(step_pattern, text_no_newlines))

    if not matches:
        return text_no_newlines.strip() + " ки" if text_no_newlines.strip() else ""

    problem_part = text_no_newlines[:matches[0].start()].strip()
    steps = []
    for i in range(len(matches)):
        start_idx = matches[i].start()
        end_idx = matches[i + 1].start() if i + 1 < len(matches) else len(text_no_newlines)
        step_text = text_no_newlines[start_idx:end_idx].strip().replace("## ", "")
        steps.append(step_text + " ки")

    if problem_part:
        steps.insert(0, problem_part)

    return " ".join(steps)

def orm_process_solution(solution_text):
    """
    최종 스텝 뒤에 ' ки'를 덧붙이는 처리(ORM 용).
    """
    processed_text = solution_text.replace('\n', ' ')
    processed_text += ' ки'
    return processed_text

##############################################################################
# (2) 메인 로직: 각 문항별 솔루션 다중 생성 -> 정답/ORM 레이블 -> JSON 저장
##############################################################################

def generate_solutions_with_labels(questions_data, repeat_count, llm, sampling_params):
    """
    1) 각 문항별 repeat_count번 솔루션 생성
    2) 최종 답안(알파벳) 추출 -> 정답과 비교하여 orm_label 계산(1/0)
    3) "answer", "orm_label", "prm_processed_solution", "orm_processed_solution" 등 추가
    4) question 단위로 묶어서 반환 (list 형태)
    """
    # 공통 system 메시지
    SYSTEM_PROMPT = (
        "Solve the following question step-by-step. "
        "Do not analyze individual options in a single step. "
        "Each step of your explanation must start with 'Step {number}:' format. "
        "You must provide the answer using the phrase 'the answer is (option alphabet)' at the end of your step."
    )

    # (A) 전체 prompt 생성
    chat_prompts = []
    prompt_metadata = []
    for qdata in questions_data:
        qid, full_question, correct_letter = format_question(qdata)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": full_question}
        ]

        for _ in range(repeat_count):
            chat_prompts.append(messages)
            prompt_metadata.append({
                "question_id": qid,
                "ground_truth_letter": correct_letter
            })

    # (B) 모델 호출
    outputs = llm.chat(chat_prompts, sampling_params)

    # (C) 각 question_id별 생성된 텍스트/정답 여부 집계
    #     - "answers"에 해당하는 알파벳, "orm_label" (1/0)
    result_map = {}
    for qdata in questions_data:
        qid, _, correct_letter = format_question(qdata)
        result_map[qid] = {
            "question_id": qid,
            "question": qdata["question"],
            "correct_answer": correct_letter,
            "solutions_raw": [],    # 임시 저장용
            "correct_count": 0,     # accuracy 계산용
        }

    for i, output_data in enumerate(outputs):
        generated_text = output_data.outputs[0].text
        meta = prompt_metadata[i]
        qid = meta["question_id"]
        gt_letter = meta["ground_truth_letter"]

        # 최종 추출한 답
        model_ans = extract_answer_from_text(generated_text)
        is_correct = (model_ans == gt_letter)
        if is_correct:
            result_map[qid]["correct_count"] += 1

        # 저장
        result_map[qid]["solutions_raw"].append(generated_text)

    # (D) question_id별 결과를 최종 구조화
    final_list = []
    for qdata in questions_data:
        qid, _, correct_letter = format_question(qdata)
        item = result_map[qid]
        total_solutions = len(item["solutions_raw"])
        accuracy = 0.0
        if total_solutions > 0:
            accuracy = item["correct_count"] / total_solutions

        # 각 솔루션 구조화: answer, orm_label, prm_processed, orm_processed
        solutions_data = []
        for sol_text in item["solutions_raw"]:
            ans_letter = extract_answer_from_text(sol_text)
            orm_label = 1 if ans_letter == correct_letter else 0

            solutions_data.append({
                "solution": sol_text,
                "answer": ans_letter if ans_letter else "N/A",
                "prm_processed_solution": prm_process_solution(sol_text),
                "orm_processed_solution": orm_process_solution(sol_text),
                "orm_label": orm_label,
            })

        final_list.append({
            "question_id": qid,
            "question": qdata["question"],
            "correct_answer": correct_letter,
            "accuracy": accuracy,
            "solutions": solutions_data
        })

    return final_list

##############################################################################
# (3) 메인 함수
##############################################################################

def parse_args():
    parser = argparse.ArgumentParser(description="(Step 1) Generate multiple solutions with basic ORM labeling.")
    parser.add_argument("--hf_token", type=str, required=True, help="Hugging Face access token.")
    parser.add_argument("--model_path", type=str, default="meta-llama/Llama-3.2-3B-Instruct",
                        help="Path or name of the LLM model.")
    parser.add_argument("--repeat_count", type=int, default=8,
                        help="Number of times to generate for each prompt.")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature.")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k value.")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-P value.")
    parser.add_argument("--max_tokens", type=int, default=512, help="Max tokens for generation.")
    parser.add_argument("--input_file", type=str, required=True,
                        help="Input JSON file (contains question info).")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Output JSON file (structured data with solutions).")
    parser.add_argument("--start_number", type=int, default=0,
                        help="Start index (inclusive) of questions to process.")
    parser.add_argument("--end_number", type=int, default=10,
                        help="End index (exclusive) of questions to process.")
    parser.add_argument("--gpu_ids", type=str, default="0",
                        help="Comma separated list of GPU device indices to use.")
    return parser.parse_args()

def main():
    args = parse_args()

    # 1) HF 로그인
    login(args.hf_token)

    # 2) GPU 환경 설정
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    device_count = torch.cuda.device_count()
    device = torch.device("cuda" if device_count > 0 else "cpu")
    print(f"[INFO] Device: {device} (GPUs={device_count}), GPU IDs={args.gpu_ids}")

    # 3) 모델 초기화
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

    # 4) 입력 JSON 읽기
    with open(args.input_file, "r", encoding="utf-8") as f:
        all_questions = json.load(f)

    questions_data = all_questions[args.start_number: args.end_number]
    print(f"[INFO] Loaded total={len(all_questions)} questions. Using slice=[{args.start_number}:{args.end_number}] -> {len(questions_data)} questions.")

    # 5) 솔루션 생성 + ORM 채점
    structured_data = generate_solutions_with_labels(
        questions_data=questions_data,
        repeat_count=args.repeat_count,
        llm=llm,
        sampling_params=sampling_params
    )

    # 6) JSON 저장
    try:
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(structured_data, f, indent=4, ensure_ascii=False)
        print(f"[INFO] Saved output file: {args.output_file}")
    except Exception as e:
        print(f"[ERROR] Failed to save JSON: {e}")

    print("[INFO] Done. (solution_sampling)")

if __name__ == "__main__":
    main()
