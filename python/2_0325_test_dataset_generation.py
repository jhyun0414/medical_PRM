#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import json
import re
import numpy as np
import torch
import string

from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from collections import defaultdict

############################
# 함수 정의부
############################

def format_question(qdata):
    """
    기존 json:
    {
      "question_id": "6616",
      "question": "A 75-year-old woman ... ...",
      "options": [
          "Mycobacterium tuberculosis",
          "Staphylococcus aureus",
          "Legionella pneumoniae",
          "Klebsiella pneumoniae",
          "Streptococcus agalactiae"
      ],
      "ground_truth": "Staphylococcus aureus",
      "type": "medqa"
    }

    이를 다음 형태로 변환:
    Question text + \n\n(A) ...
                     (B) ...
                     (C) ...
                     ...
    """
    question_id = int(qdata.get("question_id", "0"))
    question_str = qdata["question"]
    options = qdata["options"]

    # 옵션 라벨 생성 (예: (A), (B), (C), ...)
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
            correct_letter = string.ascii_uppercase[i]  # 예: 0 -> "A", 1 -> "B"
            break
    if correct_letter is None:
        correct_letter = "unknown"

    return question_id, full_question, correct_letter

def extract_steps_from_text(generated_text):
    # "## Step N:" 또는 "Step N:" 패턴 찾기
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
    text = text.strip().lower()

    # 1) LaTeX 수식 \boxed{} 내부 알파벳 추출
    if "\\boxed{" in text:
        match = re.findall(r'\\boxed\{\(?([a-z])\)?\}', text)
        if match:
            return match[-1].upper()

    # 2) "the answer is x" / "final answer is x" 등에서 알파벳 추출
    # 마지막에 등장하는 패턴을 우선시하기 위해 모든 매치를 찾고 마지막 것을 사용
    answer_pattern = re.finditer(
        r'(?:the final answer is|the answer is|final answer is|answer is)\s*:?\s*\(?([a-z])\)?',
        text, re.IGNORECASE
    )
    matches = list(answer_pattern)
    if matches:
        return matches[-1].group(1).upper()

    # 3) "swer is x" 패턴 추가 (answer is의 변형을 모두 포함)
    answer_pattern2 = re.finditer(
        r'swer is\s*:?\s*\(?([a-z])\)?',
        text, re.IGNORECASE
    )
    matches = list(answer_pattern2)
    if matches:
        return matches[-1].group(1).upper()

    return None  # 정답을 찾지 못한 경우

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

def calculate_prm_soft_label(annotations):
    """
    annotations: 2차원 리스트 (각 스텝별 [score1, score2, ...])
    """
    if not annotations:
        return []

    num_steps = len(annotations)
    completion_number = len(annotations[0])
    prm_soft_label = []
    for step_idx in range(num_steps):
        step_scores = [annotations[step_idx][i] for i in range(completion_number)]
        avg_score = np.mean(step_scores)
        prm_soft_label.append(avg_score)
    return prm_soft_label

def calculate_prm_hard_label(annotations):
    """
    annotations: 2차원 리스트
    각 스텝에서 1회라도 정답이면(>0) hard_label=1, 아니면 0
    """
    if not annotations:
        return []

    num_steps = len(annotations)
    completion_number = len(annotations[0])
    prm_hard_label = []
    for step_idx in range(num_steps):
        step_scores = [annotations[step_idx][i] for i in range(completion_number)]
        hard_label = 1 if any(score > 0 for score in step_scores) else 0
        prm_hard_label.append(hard_label)
    return prm_hard_label

def calculate_orm_label(annotations):
    """
    annotations: 2차원 리스트
    최종 스텝의 점수 중 하나라도 정답이면 ORM=1, 아니면 0
    """
    if not annotations:
        return 0

    final_step_scores = annotations[-1]
    orm_label = 1 if any(score > 0 for score in final_step_scores) else 0
    return orm_label

def generate_outputs_for_multiple_questions(
    questions_data,
    repeat_count,
    llm,
    sampling_params
):
    """
    (1) questions_data를 (A),(B) 옵션 형식으로 변환
    (2) repeat_count번씩 생성
    (3) accuracy & 스텝수 필터
    """
    chat_prompts = []
    prompt_metadata = []

    # 공통 system 프롬프트
    SYSTEM_PROMPT = (
        "Solve the following question step-by-step. "
        "Do not analyze individual options in a single step. "
        "Each step of your explanation must start with 'Step {number}:' format. "
        "You must provide the answer using the phrase 'the answer is (option alphabet)' at the end of your step."
    )

    # 1) Chat Prompt & 메타데이터 생성
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
                "ground_truth_letter": correct_letter,
            })

    # 2) vLLM 호출 (chat 모드)
    outputs = llm.chat(chat_prompts, sampling_params)

    # 3) question_id별 결과 묶기
    results_by_question = defaultdict(lambda: {
        "generated_texts": [],
        "correct_count": 0
    })

    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        meta = prompt_metadata[i]
        question_id = meta["question_id"]
        ground_truth_letter = meta["ground_truth_letter"]

        model_answer = extract_answer_from_text(generated_text)
        if model_answer == ground_truth_letter:
            results_by_question[question_id]["correct_count"] += 1

        results_by_question[question_id]["generated_texts"].append(generated_text)

    # 4) 문제별 accuracy & step 길이 필터링
    filtered_results = {}
    for qdata in questions_data:
        qid, full_question, correct_letter = format_question(qdata)  # 다시 추출
        data_dict = results_by_question[qid]
        correct_count = data_dict["correct_count"]
        generated_texts = data_dict["generated_texts"]

        accuracy = correct_count / repeat_count if repeat_count > 0 else 0.0

        print(f"\n{'='*80}")
        print(f"[Question {qid}] Accuracy: {accuracy:.2f} (Correct: {correct_count}/{repeat_count})")
        print(f"Ground Truth: {correct_letter}")
        print("Generated Solutions:")
        
        # 스텝 수 필터 (2 < step < 8)
        step_filtered_texts = []
        for idx, txt in enumerate(generated_texts, 1):
            cleaned_txt = txt.replace("\n", " ")
            steps = extract_steps_from_text(cleaned_txt)
            model_answer = extract_answer_from_text(cleaned_txt)
            
            print(f"\nSolution {idx}:")
            print(f"Model Answer: {model_answer}")
            print(f"Number of steps: {len(steps)}")
            print(f"Text: {cleaned_txt}\n")
            
            if 2 < len(steps) < 10:
                step_filtered_texts.append(cleaned_txt)

        filtered_results[qid] = {
            "solutions": step_filtered_texts,
            "accuracy": accuracy,
            "full_question": full_question  # full_question 추가
        }

        print(f"Filtered Solutions Count: {len(filtered_results[qid]['solutions'])}")
        print(f"{'='*80}\n")

    return filtered_results

def create_labeled_data_for_multiple_questions(
    filtered_results,
    questions_data,
    completion_number,
    llm,
    sampling_params,
    output_file_path
):
    """
    (1) 필터링된 솔루션들을 대상으로 중간 스텝별 partial solution 평가(정답 여부)
    (2) Soft/Hard label( PRM ), ORM label 생성
    (3) 결과를 JSON으로 저장
    """
    # question_id -> {question_text, correct_letter}
    question_meta = {}
    for q in questions_data:
        qid, full_question, correct_letter = format_question(q)
        question_meta[qid] = {
            "question": q["question"],  # 원본문
            "ground_truth_letter": correct_letter,
            "full_question": full_question  # full_question 추가
        }

    partial_solutions_list = []
    partial_solutions_metadata = []

    SYSTEM_PROMPT = (
        "Solve the following question step-by-step. "
        "Do not analyze individual options in a single step. "
        "Each step of your explanation must start with 'Step {number}:' format. "
        "You must provide the answer using the phrase 'the answer is (option alphabet)' at the end of your step."
    )

    # 결과 구조
    results_for_all = defaultdict(lambda: {
        "question_id": None,
        "question": "",
        "correct_answer": "",
        "accuracy": 0.0,
        "solutions": []
    })

    # (A) 각 qid별 솔루션 및 중간 step 파싱
    for q in questions_data:
        qid, _, _ = format_question(q)
        if qid not in filtered_results:
            continue
        # 문제 텍스트/정답
        question_text = q["question"]
        gt_letter = question_meta[qid]["ground_truth_letter"]
        full_question = question_meta[qid]["full_question"]  # full_question 가져오기

        results_for_all[qid]["question_id"] = qid
        results_for_all[qid]["question"] = question_text
        results_for_all[qid]["correct_answer"] = gt_letter
        results_for_all[qid]["accuracy"] = filtered_results[qid]["accuracy"]

        sols = filtered_results[qid]["solutions"]
        for sol_index, sol_text in enumerate(sols):
            steps = extract_steps_from_text(sol_text)
            num_mid_steps = len(steps) - 1

            prm_processed_sol_text = prm_process_solution(sol_text)
            orm_processed_sol_text = orm_process_solution(sol_text)

            results_for_all[qid]["solutions"].append({
                "solution": sol_text,
                "answer": None,
                "prm_processed_solution": prm_processed_sol_text,
                "orm_processed_solution": orm_processed_sol_text,
                "prm_soft_label": None,
                "prm_hard_label": None,
                "orm_label": None
            })

            # 중간 스텝(최종 제외)이 1개 이상 있어야 partial solution 채점 가능
            if num_mid_steps <= 0:
                continue

            # 중간 스텝별 partial solution 메시지 생성
            cumulative_steps = ""
            for step_i in range(num_mid_steps):
                step_text = steps[step_i]
                cumulative_steps = (cumulative_steps + " " + step_text).strip()

                # Chat 형식 메시지
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": full_question + "\n\n" + cumulative_steps}
                ]

                partial_solutions_list.append(messages)
                partial_solutions_metadata.append({
                    "question_id": qid,
                    "solution_index": sol_index,
                    "step_index": step_i,
                    "total_mid_steps": num_mid_steps
                })

    # (B) partial solution 스코어링
    scores_for_partial_solutions = defaultdict(list)

    if partial_solutions_list:
        repeated_prompts = []
        extended_metadata = []
        # partial solution마다 completion_number번씩 추가
        for meta, messages in zip(partial_solutions_metadata, partial_solutions_list):
            for _ in range(completion_number):
                repeated_prompts.append(messages)
            extended_metadata.extend([meta]*completion_number)

        # Chat 호출
        outputs = llm.chat(repeated_prompts, sampling_params)

        # 생성된 답안으로부터 정답 여부 스코어링
        for i, output_data in enumerate(outputs):
            generated_text = output_data.outputs[0].text
            meta = extended_metadata[i]

            qid = meta["question_id"]
            sol_idx = meta["solution_index"]
            step_idx = meta["step_index"]

            # 모델이 낸 answer
            model_answer = extract_answer_from_text(generated_text)
            gt_letter = question_meta[qid]["ground_truth_letter"]

            score = 1 if (model_answer == gt_letter) else 0
            scores_for_partial_solutions[(qid, sol_idx, step_idx)].append(score)
    else:
        print("No partial solutions found. Possibly all solutions had <= 1 step.")

    # (C) 마지막 스텝도 정답 여부 판별
    for q in questions_data:
        qid, _, _ = format_question(q)
        if qid not in filtered_results:
            continue

        gt_letter = question_meta[qid]["ground_truth_letter"]
        sols = filtered_results[qid]["solutions"]

        for sol_idx, sol_text in enumerate(sols):
            steps = extract_steps_from_text(sol_text)
            if not steps:
                continue

            final_step = steps[-1]
            final_ans_letter = extract_answer_from_text(final_step)
            is_correct = 1 if (final_ans_letter == gt_letter) else 0

            # 중간 step + 최종 step score를 하나로 묶어 annotations 구성
            num_mid = max(0, len(steps) - 1)
            mid_step_scores = []
            for step_i in range(num_mid):
                sc_list = scores_for_partial_solutions.get((qid, sol_idx, step_i), [])
                mid_step_scores.append(sc_list)

            # final step 스코어를 completion_number개로 (동일)
            final_step_scores = [is_correct]*completion_number
            mid_step_scores.append(final_step_scores)

            annotations = mid_step_scores

            prm_soft_label = calculate_prm_soft_label(annotations)
            prm_hard_label = calculate_prm_hard_label(annotations)
            orm_label = calculate_orm_label(annotations)

            # 결과 저장
            results_for_all[qid]["solutions"][sol_idx]["answer"] = final_ans_letter
            results_for_all[qid]["solutions"][sol_idx]["prm_soft_label"] = prm_soft_label
            results_for_all[qid]["solutions"][sol_idx]["prm_hard_label"] = prm_hard_label
            results_for_all[qid]["solutions"][sol_idx]["orm_label"] = orm_label

    # (D) 결과 저장
    try:
        final_results_list = list(results_for_all.values())
        with open(output_file_path, "w", encoding="utf-8") as f:
            json.dump(final_results_list, f, indent=4, ensure_ascii=False)
        print(f"[INFO] 결과가 {output_file_path}에 저장되었습니다.")
    except Exception as e:
        print(f"[ERROR] JSON 저장 중 오류 발생: {e}")

    return final_results_list

############################
# 인자 파싱 및 main 함수
############################
def parse_args():
    parser = argparse.ArgumentParser(description="Generate and label datasets with vLLM (New JSON Structure).")
    parser.add_argument("--hf_token", type=str, required=True, help="Hugging Face access token.")
    parser.add_argument("--model_path", type=str, default="meta-llama/Llama-3.2-3B-Instruct",
                        help="Path or name of the LLM model.")
    parser.add_argument("--repeat_count", type=int, default=16,
                        help="Number of times to generate for each prompt.")
    parser.add_argument("--completion_number", type=int, default=16,
                        help="Number of times to sample partial solutions.")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature.")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k value.")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-P value.")
    parser.add_argument("--max_tokens", type=int, default=1024, help="Max tokens for generation.")
    parser.add_argument("--input_file", type=str, required=True, help="Input JSON file with questions.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output JSON files.")
    parser.add_argument("--start_number", type=int, required=True, help="Start index (inclusive) of questions to process.")
    parser.add_argument("--end_number", type=int, required=True, help="End index (exclusive) of questions to process.")
    parser.add_argument("--gpu_ids", type=str, default="0", help="Comma separated list of GPU device indices to use.")
    return parser.parse_args()

def main():
    args = parse_args()

    # 1) Hugging Face 로그인
    login(args.hf_token)

    # 2) GPU 환경 설정
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    available_gpu_count = torch.cuda.device_count()
    device = torch.device("cuda" if available_gpu_count > 0 else "cpu")
    print(f"✅ 실행 기기: {device} (총 {available_gpu_count}개 GPU 사용), 선택된 GPU IDs: {args.gpu_ids}")

    # 3) LLM 초기화
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
#        top_p=args.top_p,
        max_tokens=args.max_tokens
    )

    # 4) 입력 JSON 파일 읽기
    try:
        with open(args.input_file, "r", encoding="utf-8") as f:
            all_questions_data = json.load(f)
    except FileNotFoundError:
        print(f"❌ 파일을 찾을 수 없음: {args.input_file}")
        return

    # 문제 번호 범위 (인덱스 슬라이싱)
    questions_data = all_questions_data[args.start_number: args.end_number]
    print(f"📂 총 {len(all_questions_data)}문제 중, {args.start_number}부터 {args.end_number}까지 ({len(questions_data)}문제) 처리합니다.")

    # (A) 1차 생성 (accuracy/스텝수 필터)
    filtered_outputs = generate_outputs_for_multiple_questions(
        questions_data=questions_data,
        repeat_count=args.repeat_count,
        llm=llm,
        sampling_params=sampling_params
    )

    # (B) 2차 레이블 (중간 스텝 정답 판별)
    output_file = f"{args.output_dir}/train_dataset_{args.start_number}_{args.end_number}.json"
    results = create_labeled_data_for_multiple_questions(
        filtered_results=filtered_outputs,
        questions_data=questions_data,
        completion_number=args.completion_number,
        llm=llm,
        sampling_params=sampling_params,
        output_file_path=output_file
    )

    print(f"✅ 최종 결과 저장 완료: {output_file}")

if __name__ == "__main__":
    main()
