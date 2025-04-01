import argparse
import os
import json
import re
import math
import string  # <-- (A)(B)(C) 등 라벨링용
import torch
from collections import defaultdict

from huggingface_hub import login
from vllm import LLM, SamplingParams

def parse_args():
    parser = argparse.ArgumentParser(description="vLLM chatting 기반 문제 풀이 생성")
    parser.add_argument("--hf_token", type=str, required=True, help="Hugging Face Access Token")
    parser.add_argument("--model_path", type=str, required=True, help="사용할 모델 경로")
    parser.add_argument("--gpu_id", type=str, required=True, help="사용할 GPU id (예: 0)")
    parser.add_argument("--input_file", type=str, required=True, help="입력 JSON 파일 경로")
    parser.add_argument("--output_dir", type=str, required=True, help="출력 JSON 파일을 저장할 디렉토리")
    parser.add_argument("--repeat_count", type=int, default=64, help="각 문제별로 생성할 솔루션 개수")
    parser.add_argument("--temperature", type=float, default=0.7, help="샘플링 온도")
    parser.add_argument("--top_k", type=int, default=50, help="샘플링 top_k")
    parser.add_argument("--top_p", type=float, default=0.9, help="샘플링 top_p")
    parser.add_argument("--max_tokens", type=int, default=1024, help="최대 토큰 수")
    return parser.parse_args()

# --------------------------
# (1) 질문 전처리 함수
# --------------------------
def format_question(qdata):
    """
    새 json 예시:
    {
      "question_id": "776",
      "question": "Which of the following is ...?",
      "options": [
          "Prostaglandins",
          "Inositol triphosphate",
          "Cyclic AMP",
          "Calmodulin"
      ],
      "ground_truth": "Inositol triphosphate"
    }
    
    이 정보를 아래처럼 변환:
    full_question = "Which of the following ...?\n\n(A) Prostaglandins\n(B) Inositol triphosphate\n(C) Cyclic AMP\n(D) Calmodulin"
    그리고 ground_truth(텍스트)에 해당하는 (B) → 알파벳 "B" 찾기
    """
    question_id = qdata.get("question_id", "unknown")
    question_str = qdata["question"]
    options = qdata["options"]
    gt_text = qdata["ground_truth"].strip().lower()

    # (A), (B), ... 라벨링
    option_labels = [f"({letter})" for letter in string.ascii_uppercase[:len(options)]]
    options_text = "\n".join(f"{lab} {opt}" for lab, opt in zip(option_labels, options))

    # 최종 user에게 보여줄 질문
    full_question = f"{question_str}\n\n{options_text}"

    # ground_truth로부터 정답 알파벳 찾기
    correct_letter = "unknown"
    for i, opt in enumerate(options):
        if opt.strip().lower() == gt_text:
            correct_letter = string.ascii_uppercase[i]  # 예: 0 → A, 1 → B, ...
            break

    return question_id, full_question, correct_letter

# --------------------------
# (2) 후처리 및 JSON 입출력
# --------------------------
def extract_steps_from_text(generated_text):
    """Step 패턴(## Step N: or Step N:) 추출"""
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
    """결과물에서 (option alphabet) 형태를 인식"""
    text = text.strip().lower()

    # 1) LaTeX 수식 \boxed{} 내부 알파벳 추출
    if "\\boxed{" in text:
        match = re.findall(r'\\boxed\{\(?([a-z])\)?\}', text)
        if match:
            return match[-1].upper()

    # 2) "answer is X" 패턴
    answer_pattern = re.finditer(
        r'(?:the final answer is|the answer is|final answer is|answer is)\s*:?\s*\(?([a-z])\)?',
        text, re.IGNORECASE
    )
    matches = list(answer_pattern)
    if matches:
        return matches[-1].group(1).upper()

    return None

def prm_process_solution(input_text):
    """
    예시 처리: Step 마다 ' ки' 라는 단어를 덧붙이는 형태
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
        end_idx = matches[i+1].start() if i+1 < len(matches) else len(text_no_newlines)
        step_text = text_no_newlines[start_idx:end_idx].strip().replace("## ", "")
        steps.append(step_text + " ки")
    if problem_part:
        steps.insert(0, problem_part)
    return " ".join(steps)

def load_json_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json_file(data, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"✅ Output saved to: {file_path}")

# --------------------------
# (3) Chat 프롬프트/응답 수집
# --------------------------
SYSTEM_PROMPT = (
    "Solve the following question step-by-step. "
    "Do not analyze individual options in a single step. "
    "Each step of your explanation must start with 'Step {number}:' format. "
    "You must provide the answer using the phrase 'the answer is (option alphabet)' at the end of your step."
)

def _collect_generation_requests(questions_data, n_samples):
    """
    주어진 questions_data에는 이미
      {
        "question_id": ...,
        "question":  (options을 포함하여 (A)..(B).. 형태로 변환된 질문),
        "ground_truth":  (정답 알파벳)
      }
    형태가 들어 있음.

    각 질문마다 n_samples개씩 대화 프롬프트 생성 → conversations, prompt_metadata 반환
    """
    conversations = []
    prompt_metadata = []
    
    for qdata in questions_data:
        qid = qdata["question_id"]
        question_str = qdata["question"]      # (A), (B)... 까지 포함된 전체 텍스트
        ground_truth = qdata["ground_truth"]  # 알파벳

        conversation_template = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question_str}
        ]
        for _ in range(n_samples):
            conversations.append(conversation_template)
            prompt_metadata.append({
                "question_id": qid,
                "ground_truth": ground_truth,
            })
    return conversations, prompt_metadata

def _run_llm_and_filter(llm, sampling_params, conversations, prompt_metadata):
    """
    llm.chat() 호출 후, question_id별로 결과를 묶어서 반환
    """
    outputs = llm.chat(conversations, sampling_params)
    results_by_question = defaultdict(lambda: {"generated_texts": []})

    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        meta = prompt_metadata[i]
        question_id = meta["question_id"]
        results_by_question[question_id]["generated_texts"].append(generated_text)

    return results_by_question

# --------------------------
# (4) 1차/2차 샘플링 & 필터링
# --------------------------
def generate_outputs_for_multiple_questions(questions_data, repeat_count, llm, sampling_params):
    """
    1) 1차: 각 문제마다 ceil(repeat_count * 1.25)개 생성
    2) 필터링 후 부족분 계산 → 부족분 * 3만큼 2차 생성
    3) 최종 필터링 후 상위 repeat_count개 확정
    """
    # 1차 생성 개수
    first_pass_count = math.ceil(repeat_count * 1.25)

    # (1) 1차 생성
    conversations, prompt_metadata = _collect_generation_requests(questions_data, first_pass_count)
    pass1_results = _run_llm_and_filter(llm, sampling_params, conversations, prompt_metadata)
    
    # (1차) 필터링
    valid_solutions_pass1 = defaultdict(list)
    for qdata in questions_data:
        qid = qdata["question_id"]
        generated_texts = pass1_results[qid]["generated_texts"]

        # ex) 필터 조건: "Step"이 3개 이상 10개 미만
        # (원하시는 기준으로 변경 가능)
        for txt in generated_texts:
            steps = extract_steps_from_text(txt)
            if 2 < len(steps) < 10:
                valid_solutions_pass1[qid].append(txt)

    # (2) 부족분 계산
    questions_need_pass2 = []
    for qdata in questions_data:
        qid = qdata["question_id"]
        curr_valid_count = len(valid_solutions_pass1[qid])
        if curr_valid_count < repeat_count:
            # 부족분만큼 추가 생성
            lacking = repeat_count - curr_valid_count
            qdata["__needed_pass2"] = lacking * 3
            questions_need_pass2.append(qdata)

    # (3) 2차 생성
    if questions_need_pass2:
        conversations_pass2 = []
        prompt_metadata_pass2 = []
        for qdata in questions_need_pass2:
            needed_count = qdata["__needed_pass2"]
            qid = qdata["question_id"]
            question_str = qdata["question"]
            ground_truth = qdata["ground_truth"]

            conversation_template = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question_str}
            ]
            for _ in range(needed_count):
                conversations_pass2.append(conversation_template)
                prompt_metadata_pass2.append({
                    "question_id": qid,
                    "ground_truth": ground_truth,
                })

        pass2_results = _run_llm_and_filter(llm, sampling_params, conversations_pass2, prompt_metadata_pass2)
        
        # 2차 결과도 필터링 후 1차 결과와 합침
        for qdata in questions_need_pass2:
            qid = qdata["question_id"]
            generated_texts = pass2_results[qid]["generated_texts"]
            for txt in generated_texts:
                steps = extract_steps_from_text(txt)
                if 2 < len(steps) < 10:
                    valid_solutions_pass1[qid].append(txt)

    # --------------------------
    # (4) 최종 output_dataset 만들기
    # --------------------------
    output_dataset = []
    for qdata in questions_data:
        qid = qdata["question_id"]
        question_str = qdata["question"]
        ground_truth_letter = qdata["ground_truth"]  # 알파벳 (ex: "B")

        # 상위 repeat_count개의 솔루션만 최종 사용
        valid_solutions_all = valid_solutions_pass1[qid]
        final_solutions = valid_solutions_all[:repeat_count]

        solutions_list = []
        for solution_text in final_solutions:
            processed_solution = prm_process_solution(solution_text)
            extracted_answer = extract_answer_from_text(solution_text)  # 모델이 예측한 알파벳

            solutions_list.append({
                "solution": solution_text,
                "prm_processed_solution": processed_solution,
                "answer": extracted_answer
            })

        # output용
        output_dataset.append({
            "question_id": qid,
            # 여기서는 모델에게 준 최종 question(옵션 포함) 저장
            # 필요하다면 original question, options 등을 추가 필드로 붙여도 됨
            "question": question_str,
            "correct_answer": ground_truth_letter,  # 실제 정답 알파벳
            "solutions": solutions_list
        })

    return output_dataset

# --------------------------
# (5) 메인 실행부
# --------------------------
def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    # HF 로그인
    login(args.hf_token)
    print("login 완료")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"현재 활성화된 디바이스: {device}")
    print(f"현재 선택된 GPU 번호: {torch.cuda.current_device()}")

    # vLLM 생성
    llm = LLM(
        model=args.model_path,
        device="cuda",
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=0.8,
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    # (1) JSON 로드
    raw_data = load_json_file(args.input_file)
    print(f"📂 총 {len(raw_data)}개 문제")

    # (2) format_question을 적용하여 데이터 변환
    transformed_data = []
    for entry in raw_data:
        qid, full_q, correct_letter = format_question(entry)
        transformed_data.append({
            "question_id": qid,
            "question": full_q,            # (A), (B), ... 형태로 변환된 질문
            "ground_truth": correct_letter # 정답 알파벳
        })

    # (3) 최종 생성
    output_dataset = generate_outputs_for_multiple_questions(
        transformed_data,
        args.repeat_count,
        llm,
        sampling_params
    )

    # (4) 저장 - input 파일명 기반으로 output 파일명 생성
    input_filename = os.path.basename(args.input_file)
    input_name = os.path.splitext(input_filename)[0]  # 확장자 제외한 파일명
#    output_filename = f"{input_name}_solutions_{args.repeat_count}.json"
    output_filename = f"test_dataset.json"
    output_file_path = os.path.join(args.output_dir, output_filename)
    
    save_json_file(output_dataset, output_file_path)
    print(f"✅ 완료: {output_file_path}")

if __name__ == "__main__":
    main()
