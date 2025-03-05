# train_grpo.py
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
import wandb
import os
from peft import LoraConfig, PeftModel,PeftConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    pipeline,
    logging,
    DataCollatorForLanguageModeling
)
import torch
import math
from langdetect import detect_langs
import sys
from sentence_transformers.util import cos_sim
from sentence_transformers import SentenceTransformer as SBert

os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2"

if torch.cuda.is_available():
    print("Using CUDA GPU")
    device = torch.device("cuda")
else:
    print("Using CPU")
    device = torch.device("cpu")


# start a new wandb run to track this script
# wandb.init(
#     # set the wandb project where this run will be logged
#     project="GRPO",
# )


def extract_final_answer(text):
    """提取 Final Answer: 后面的内容"""
    if '</think>' in text:
        text = text.split('</think>')[-1].strip()
        return text
    else:
        return text[-30:] 

local_path="/"
similarity_model = SBert("model/paraphrase-MiniLM-L12-v2")


def setup_model():
    """Setup the DeepSeek model with memory-efficient configuration for Apple Silicon."""
    base_model_name = "/model/Qwen2.5-7B-Instruct"
    adapter_model_name = "/model/Qwen-2_5-7B-SFT100"
    config = PeftConfig.from_pretrained(adapter_model_name)
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path, 
        load_in_4bit=True, 
        device_map="auto",
        torch_dtype=torch.bfloat16,
        use_cache=False,  # Disable KV-cache to save memory,
        ).eval()
    tokenizer = AutoTokenizer.from_pretrained(adapter_model_name)

    model = PeftModel.from_pretrained(model, adapter_model_name, adapter_name="SFT100")

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Apply memory optimizations
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    return model, tokenizer


model, tokenizer = setup_model()


# 奖励函数为中文的占比
def reward_chinese_ratio(completions, **kwargs):
    zh_probs = []
    for completion in completions:
        try:
            # 检测语言
            detected_langs = detect_langs(completion)
            # 获取中文 (zh) 的概率，如果没有检测到中文则返回 0
            zh_prob = next(
                (lang.prob for lang in detected_langs if lang.lang == 'zh-cn'), 0.01)
            zh_probs.append(zh_prob)
        except Exception as e:
            # 如果检测失败（例如文本太短），默认返回 0
            print(
                f"Error detecting language for completion: {completion}. Error: {e}")
            zh_probs.append(0)
    return zh_probs


def reward_think_ratio(completions, **kwargs):
    scores = []
    for completion in completions:
        think_count = completion.count("<think>")
        think_end_count = completion.count("</think>")
        # 根据 <think> 和 </think> 的数量计算分值
        score =  - abs(think_count - think_end_count)*0.5 -abs(think_count - 1)-abs(think_end_count - 1)
        
        # 检查是否有其他标签
        other_tags_count = sum(1 for tag in completion.split("<") if tag and tag.split(">")[0] not in ["think", "/think"])
        score -= other_tags_count  # 如果有其他标签，减分

        scores.append(score)
    return scores


def similarity_sentence_score(completions, target,  ** kwargs):
    # Compute embedding for both lists
    scores = []
    for c, t in zip(completions, target):
        c = extract_final_answer(c)
        t = extract_final_answer(t)
        embeddings1 = similarity_model.encode(c)
        embeddings2 = similarity_model.encode(t)
        cosine_scores = cos_sim(embeddings1, embeddings2)
        scores.append(cosine_scores)
    return scores


def correct_score(completions, target, type, clear_answer, ** kwargs):
    # Compute embedding for both lists
    scores = []
    for c, t ,ty, a in zip(completions, target, type, clear_answer):
        print(c)
        c = extract_final_answer(c)
        score=0.0
        if a in c:
            score+=1.0
        else:
            score+=0.0
        if "最终答案"in c:
            score+=.5
        else:
            score+=0.0
        scores.append(score)
    return scores

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=4,  # Reduced rank for faster training
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"]
)


def prepare_dataset(dataset):
    print(f"Dataset loaded with {len(dataset['train'])} training examples")

    def format_instruction(sample):
        # Following DeepSeek's recommended prompt format
        return f"""<|im_start|>system\n您是一位专业的医生。现在需要您根据患者的情况和问题仔细回答问题。
问题如下，请你一步步用中文回答，不要提问<|im_end|>\n \
<|im_start|>user\n{sample['question']}<|im_end|>\n \
<|im_start|>assistant
"""
    # """您是一位专业的医生。现在需要您根据患者的情况和问题仔细回答问题。
    # 问题如下，请你一步步用中文回答，不要提问。同时你需要在<think>和</think>标签对中输出你的思考步骤，并在</think>后输出你的最终总结答案。
    # {sample['question']}
    # 现在，让我们一步步来解决这个问题。
    # """
# """<|im_start|>system\n您是一位专业的医生。现在需要您根据患者的情况和问题仔细回答问题。
# 问题如下，请你一步步用中文回答，不要提问<|im_end|>\n \
# <|im_start|>user\n{sample['question']}<|im_end|>\n \ 
# <|im_start|>assistant\n<think>现在，让我们一步步来解决这个问题。{sample['cot']}</think>
# {sample['originla_answer']}<|im_end|>\n"""

# The code you provided seems to be a Python script for training a model using the GRPO (Generative
# Retrieval Pretraining with Oracles) framework. Here's a brief overview of the key components and
# functionalities in the script:

# 同时你需要在<think>和</think>标签对中输出你的思考步骤，并在</think>后输出你的最终总结答案。
    def format_answer(sample):
        return f"""<think>现在，让我们一步步来解决这个问题。{sample['cot']}</think>
{sample['originla_answer']}<|im_end|>"""

    # Prepare training dataset
    train_dataset = dataset["train"].map(
        lambda x: {
            "prompt": format_instruction(x),
            "target": format_answer(x),
            "type": x['question_type'],
            "clear_answer": x["answer"]},
        remove_columns=dataset["test"].column_names,
        num_proc=os.cpu_count()
    )

    print(f"\nUsing {len(train_dataset)} examples for training")
    print("\nSample formatted data:")
    print(format_instruction(dataset["train"][0]))
    # Prepare testing dataset
    test_dataset = dataset["test"].map(
        lambda x: {
            "prompt": format_instruction(x),
            "target": format_answer(x),
            "type": x['question_type'],
            "clear_answer": x["answer"]},
        remove_columns=dataset["test"].column_names,
        num_proc=os.cpu_count()
    )

    return train_dataset, test_dataset


dataset = load_dataset(
    "json", data_files="/clean_zh_results.json")

dataset = dataset["train"].select(range(100))
dataset = dataset.train_test_split(train_size=0.9, test_size=0.1, seed=42)
train_dataset, test_dataset = prepare_dataset(dataset=dataset)

training_args = GRPOConfig(
    output_dir="tmp",
    run_name="grpo",
    beta=0.01,
    logging_steps=1,
    learning_rate=1e-5,
    max_completion_length=512,
    report_to="wandb",
    num_train_epochs=2,
    optim="adamw_torch_fused",
    weight_decay=0.1,
    warmup_ratio=0.1,
    save_total_limit=1,
    lr_scheduler_type='cosine',
    save_steps=100,
    bf16=True,
    group_by_length=True,
    max_grad_norm=0.3,
    dataloader_num_workers=0,
    temperature=0.7,
    log_completions=True,
    num_generations=2,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    use_vllm=True,
    vllm_device="cuda:0",
    vllm_gpu_memory_utilization=0.7,
    vllm_max_model_len=1024
)

trainer = GRPOTrainer(
    model=model,
    reward_funcs=[
        reward_think_ratio,
        similarity_sentence_score,
        correct_score],
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    peft_config=peft_config,

)
trainer.train()
save_model_path = "/drpo/model"
print("\nSaving model...")
trainer.model.save_pretrained(save_model_path)
tokenizer.save_pretrained(save_model_path)
