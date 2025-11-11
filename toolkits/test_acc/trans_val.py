from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from tqdm import tqdm
import string
import os


model_path = '/root/aicloud-data/models/Qwen3-8B-hf-16000its'

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto",
    attn_implementation="flash_attention_2"
)
correct = 0
file_path = "/root/aicloud-fs/emo-contest-split/valid/valid-classfication.jsonl"
save_path = "/root/aicloud-fs/emo-contest-split/valid/valid-classfication_with_gen_16000its.jsonl"

dataset = []
results = []

with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():  # 跳过空行
            dataset.append(json.loads(line))

for data in tqdm(dataset):
    messages = [
        {"role": "user", "content": data['instruction'] + data['input'] },
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False # Switches between thinking and non-thinking modes. Default is True.
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=32768
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

    try:
        # rindex finding 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    data['generated'] = thinking_content + content
    results.append(data)
    # print("thinking content:", thinking_content)
    # print("content:", content)
    ans = content.strip(string.punctuation).lower()
    if ans == data['output'].strip(string.punctuation).lower():
        correct += 1


print(f"Accuracy: {correct/len(dataset)}")
os.makedirs(os.path.dirname(save_path), exist_ok=True)
with open(save_path, "w", encoding="utf-8") as f:
    for item in results:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

