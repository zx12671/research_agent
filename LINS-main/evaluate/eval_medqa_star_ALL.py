"""
MedQA 全地区（US, Mainland, Taiwan）评测脚本
参照 eval_pubmedqa_star_ALL.py 的框架，支持三个地区数据集
"""
from arguments import get_medlinker_args
import json
from tqdm import tqdm
import argparse
import logging
import concurrent.futures
from functools import partial
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.model_LINS import LINS as model_LINS

# 数据集路径映射（相对于脚本所在目录的 evaluate_data）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATHS = {
    "medqa_us": os.path.join(SCRIPT_DIR, "evaluate_data", "medqa_us", "data", "medqa_us_test.json"),
    "medqa_mainland": os.path.join(SCRIPT_DIR, "evaluate_data", "medqa_mainland", "data", "medqa_mainland_test.json"),
    "medqa_taiwan": os.path.join(SCRIPT_DIR, "evaluate_data", "medqa_taiwan", "data", "medqa_taiwan_test.json"),
}

# Chat 模式的 System Prompt（英文数据集用英文 prompt，中文数据集用中文 prompt）
SYSTEM_PROMPTS = {
    "medqa_us": """
You are a helpful assistant specialized in single-choice medical question answering. Provide only the option index ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H') as the answer to single-choice questions rather than the specific content of the options. Do not include any additional text or explanations. For example, don't say: "Here are the answer".

""",
    "medqa_mainland": """
你是一个专业的医学单项选择题解答助手。请只输出选项的字母索引（'A'、'B'、'C'、'D'、'E'、'F'、'G'、'H'）作为答案，不要输出选项的具体内容。不要包含任何额外的文字或解释。

""",
    "medqa_taiwan": """
你是一个专业的医学单项选择题解答助手。请只输出选项的字母索引（'A'、'B'、'C'、'D'、'E'、'F'、'G'、'H'）作为答案，不要输出选项的具体内容。不要包含任何额外的文字或解释。

""",
}

# 结果保存根路径
RESULT_ROOTS = {
    "medqa_us": os.path.join(SCRIPT_DIR, "evaluate_results", "medqa_us"),
    "medqa_mainland": os.path.join(SCRIPT_DIR, "evaluate_results", "medqa_mainland"),
    "medqa_taiwan": os.path.join(SCRIPT_DIR, "evaluate_results", "medqa_taiwan"),
}


def run_batch_jobs(run_task, tasks, max_thread):
    """
    Run a batch of tasks with cache.
    - run_task: the function to be called
    - tasks: the list of inputs for the function
    - max_thread: the number of threads to use
    """
    results = [None] * len(tasks)
    max_failures = 10
    observed_failures = 0

    with concurrent.futures.ThreadPoolExecutor(max_thread) as executor, tqdm(total=len(tasks)) as pbar:
        if tasks[0].get("context"):
            future_to_index = {executor.submit(run_task, 
                                               question=task['prompt'], 
                                               search_results=task['search_result'],
                                               retrieval_passages=task['retrieval_passage'],
                                               contexts=task['context']): idx for idx, task in enumerate(tasks)}
        else:
            future_to_index = {executor.submit(run_task, 
                                               question=task['prompt'], 
                                               search_results=task['search_result'],
                                               retrieval_passages=task['retrieval_passage']): idx for idx, task in enumerate(tasks)}

        for future in concurrent.futures.as_completed(future_to_index):
            pbar.update(1)
            idx = future_to_index[future]
            try:
                result = future.result()
                results[idx] = result  # Store the result at the correct index
            except Exception as e:
                logging.exception("Error occurred during run_batch_jobs.")
                observed_failures += 1
                if observed_failures > max_failures:
                    raise
    return results


def load_task_data(task_path, num_lines, medlinker_args, search_results=None, retrieval_passages=None):
    """
    加载任务数据，返回 batch 数组
    """
    batch_prompt = []
    batch_question = []
    batch_contexts = []
    batch_answer = []
    batch_search_results = []
    batch_retrieval_passages = []

    with open(task_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        i = -1
        for qid, value in data.items():
            i += 1
            if i < num_lines:
                continue
            if i < medlinker_args.num_begin:
                continue
            if medlinker_args.num_end != -1 and i >= medlinker_args.num_end:
                break
            
            question = value["QUESTION"]
            options_str = value["options_str"]
            answer = value["answer"]  # 直接是 "A", "B", "C" 等字母索引

            prompt = f"**Question**: {question}"
            prompt += options_str
            prompt += f"\n**Answer**:\n"

            batch_prompt.append(prompt)
            batch_question.append(question)
            batch_answer.append(answer)
            batch_contexts.append([])  # MedQA 没有 context 字段
            if search_results:
                batch_search_results.append(search_results[i])
            if retrieval_passages:
                batch_retrieval_passages.append(retrieval_passages[i])

    return batch_prompt, batch_question, batch_answer, batch_contexts, batch_search_results, batch_retrieval_passages


def main():
    medlinker_args = get_medlinker_args()
    num_batch = medlinker_args.num_batch
    model_name_list = medlinker_args.model_name_list
    task_list = medlinker_args.task_list
    local_data_name = medlinker_args.local_data_name

    medlinker_args.llm_keys = os.environ.get("OPEN_API_KEY")
    medlinker_args.assistant_keys = os.environ.get("OPEN_API_KEY")
    
    for model_name in model_name_list:
        medlinker_args.llm_model_name = model_name
        LINS = model_LINS(LLM_name=model_name)

        for method in medlinker_args.method_list:
            for task_id, task_name in enumerate(task_list):
                # 检查任务名是否支持
                assert task_name in DATASET_PATHS, f"Task {task_name} not supported. Supported: {list(DATASET_PATHS.keys())}"
                task_path = DATASET_PATHS[task_name]
                result_root = RESULT_ROOTS[task_name]

                # 准备结果保存路径
                if not os.path.exists(result_root):
                    os.makedirs(result_root)
                result_save_path = os.path.join(result_root, f"{task_name}_")
                result_save_path = result_save_path + model_name.replace(".","") + f"_{method}_{local_data_name}_{medlinker_args.save_name}"
                
                # 断点续跑
                if os.path.exists(result_save_path):
                    with open(result_save_path, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                        num_lines = len(lines)
                        if num_lines:
                            line = json.loads(lines[-1])
                            total_number = line["total_number"]
                            correct_number = line["correct_number"]
                        else:
                            num_lines = 0
                            total_number = 0
                            correct_number = 0
                else:
                    num_lines = 0
                    total_number = 0
                    correct_number = 0

                # 加载搜索结果（如果有）
                if medlinker_args.search_results_path_list == []:
                    search_results = None
                else:
                    search_path = medlinker_args.search_results_path_list[task_id]
                    with open(search_path, "r", encoding="utf-8") as f:
                        search_results = f.readlines()
                
                # 加载检索段落（如果有）
                if medlinker_args.retrieval_passages_path_list == []:
                    retrieval_passages = None
                else:
                    retrieval_path = medlinker_args.retrieval_passages_path_list[task_id]
                    with open(retrieval_path, "r", encoding="utf-8") as f:
                        retrieval_passages = f.readlines()

                # 加载任务数据
                batch_prompt, batch_question, batch_answer, batch_contexts, batch_search_results, batch_retrieval_passages = \
                    load_task_data(task_path, num_lines, medlinker_args, search_results, retrieval_passages)

                # 批量推理并保存
                with open(result_save_path, "a", encoding="utf-8") as sf:
                    for i in range(0, len(batch_prompt), num_batch):
                        print(f"Processing {i//num_batch + 1}/{(len(batch_prompt) + num_batch - 1)//num_batch}...")
                        prompts = batch_prompt[i:i+num_batch]
                        questions = batch_question[i:i+num_batch]
                        answers = batch_answer[i:i+num_batch]  
                        contexts = batch_contexts[i:i+num_batch]

                        if batch_search_results:
                            search_results_batch = batch_search_results[i:i+num_batch]
                            search_results_batch = [json.loads(result) for result in search_results_batch]
                            search_questions = [result["QUESTION"] for result in search_results_batch]
                            assert search_questions == questions, "Search results order mismatch"
                        else:
                            search_results_batch = None

                        if batch_retrieval_passages:
                            retrieval_passages_batch = batch_retrieval_passages[i:i+num_batch]
                            retrieval_passages_batch = [json.loads(result) for result in retrieval_passages_batch]
                            retrieval_questions = [result["question"] for result in retrieval_passages_batch]
                            retrieval_passages_batch = [result["retrieved_passages"] for result in retrieval_passages_batch]
                            assert retrieval_questions == questions, "Retrieval passages order mismatch"
                        else:
                            retrieval_passages_batch = None       

                        # 选择方法
                        if method == "chat":
                            run_task = LINS.chat_for_evaluation
                            # 根据任务名选择对应的 system prompt
                            sys_prompt = SYSTEM_PROMPTS.get(task_name, SYSTEM_PROMPTS["medqa_us"])
                            prompts = [sys_prompt + prompt for prompt in prompts]
                        elif method == "Original_RAG":
                            run_task = partial(LINS.Original_RAG_option, topk=5, local_data_name="", yuzhi=0, if_pubmed=True, embedding_model="text-embedding-ada-002")
                        elif method == "MAIRAG":
                            run_task = partial(LINS.MAIRAG_options, topk=5, local_data_name="", yuzhi=0, if_pubmed=True)
                        else:
                            print(f"method {method} is not supported.")
                            exit()
                            
                        # 构建任务列表
                        task_dict_list = []
                        for j, prompt in enumerate(prompts):
                            if search_results_batch:
                                search_result = search_results_batch[j]
                            else:
                                search_result = None
                            if retrieval_passages_batch:
                                retrieval_passage = retrieval_passages_batch[j]
                            else:
                                retrieval_passage = None
                            task_dict = {"prompt": prompt, "search_result": search_result, "retrieval_passage": retrieval_passage}
                            task_dict_list.append(task_dict)

                        true_correct_number = correct_number
                        true_total_number = total_number

                        for num_try in range(5):
                            correct_number = true_correct_number  # 错误之后要回溯
                            total_number = true_total_number
                            try:
                                results = run_batch_jobs(run_task=run_task, tasks=task_dict_list, max_thread=num_batch)
                                batch_save_results = []
                                for value in results:
                                    assert value is not None and (len(value) == 3 if method == "chat" else len(value) == 5)
                                for j, value in enumerate(results):
                                    if method == "chat":
                                        response, history, _ = value
                                        if "Final Answer" in response:
                                            response2 = response.split("Final Answer: ")[1]
                                        else:
                                            response2 = response
                                    else:
                                        response, urls, retrieved_passages, history, question_list = value
                                        if "Final Answer" in response:
                                            response2 = response.split("Final Answer: ")[1]
                                        else:
                                            response2 = response
                                    # 提取答案字母
                                    if response2 in ["A", "B", "C", "D", "E", "F", "G", "H"]:
                                        response_idx = response2
                                    else:
                                        response_idx = 'A'  # 默认
                                        for ch in response2:
                                            if ch in ["A", "B", "C", "D", "E", "F", "G", "H"]:
                                                response_idx = ch
                                                break
                                    model_pred_idx = response_idx
                                    question = questions[j]
                                    answer = answers[j]
                                    total_number += 1
                                    if model_pred_idx == answer:
                                        correct_number += 1
                                    acc = correct_number / total_number
                                    if method == "chat":
                                        model_results = {"acc": acc, "model_pred_idx": model_pred_idx,
                                                        "response": response, 
                                                        "correct_number": correct_number, 
                                                        "total_number": total_number, 
                                                        "question": question, "history": history}
                                    else:
                                        model_results = {"acc": acc, "model_pred_idx": model_pred_idx,
                                                       "response": response, 
                                                        "correct_number": correct_number, 
                                                        "total_number": total_number, 
                                                        "question_list": question_list,
                                                        "urls": urls, "retrieved_passages": retrieved_passages,
                                                        "question": question, "history": history}
                                    batch_save_results.append(model_results)
                                break
                            except Exception as e:
                                print(f"Error in {num_try} try: {e}")
                        if len(batch_save_results) == num_batch:
                            for save_line in batch_save_results:
                                sf.write(json.dumps(save_line, ensure_ascii=False) + "\n")
                                sf.flush()
                        else:
                            for num in range(num_batch - len(batch_save_results)):
                                model_results['acc'] = -1
                                model_results['retrieved_passages'] = []
                                sf.write(json.dumps(model_results, ensure_ascii=False) + "\n")
                                sf.flush()
                            

if __name__ == "__main__":
    main()
