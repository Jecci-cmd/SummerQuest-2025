from transformers import AutoTokenizer
import json
import time
import vllm
from vllm import SamplingParams

MODEL_PATH = "/data-mnt/data/downloaded_ckpts/Qwen3-8B"
INPUT_JSON = "/data-mnt/data/camp-2025/jxli/SummerQuest-2025/handout/day-3/query_only.json"
OUTPUT_JSON = "hw3_2.json"
TOKENIZER_PATH = "./tokenizer_with_special_tokens"

# 初始化分词器
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)

# 初始化vllm引擎
print("初始化vLLM引擎...")
llm = vllm.LLM(
    model=MODEL_PATH,
    trust_remote_code=True,
    gpu_memory_utilization=0.85,
    max_model_len=4096,
    enforce_eager=True
)
print("vLLM引擎初始化完成")

# 加载查询数据
with open(INPUT_JSON, "r", encoding="utf-8") as f:
    queries = json.load(f)

# 配置采样参数
sampling_params = SamplingParams(
    temperature=0.6,
    top_p=0.75,
    max_tokens=1500,
    skip_special_tokens=False
)

# 生成提示词函数
def generate_prompt(query):
    system_content = (
        "你是代码辅助系统核心模块，你需根据用户请求选择工作模式，一共有以下两种模式：\n"
        "1. 代理模式（<|AGENT|>）：用户仅提代码异常/报错等模糊问题时，步骤为：\n"
        "   - 先调用python工具，传入待分析代码获取错误信息\n"
        "   - 再调用editor工具，传入原始代码与修正代码\n"
        "2. 编辑模式（<|EDIT|>）：用户明确指出具体问题（如缩进/语法错误）时，直接调用editor工具\n"
        "必须包含：\n"
        "- 用</think>和<|FunctionCallEnd|>包裹的思考过程（体现判断逻辑）\n"
        "- 严格格式的模式标记及函数调用，示例：\n"
        "代理模式示例：\n"
        "<|FunctionCallBegin|>用户说代码结果不对，先运行def add(a,b):return a-b看看，发现是减法逻辑错了，再改成加法<RichMediaReference><|AGENT|>\n{\"name\": \"python\", \"arguments\": {\"code\": \"def add(a, b):\\n    return a - b\"}}\n<|AGENT|>\n{\"name\": \"editor\", \"arguments\": {\"original_code\": \"def add(a, b):\\n    return a - b\", \"modified_code\": \"def add(a, b):\\n    return a + b\"}}\n"
        "编辑模式示例：\n"
        "<think>用户说缩进错了，直接修正def check_positive(num):if num>0:return True的缩进<RichMediaReference><|EDIT|>\n{\"name\": \"editor\", \"arguments\": {\"original_code\": \"def check_positive(num):\\nif num > 0:\\nreturn True\", \"modified_code\": \"def check_positive(num):\\n    if num > 0:\\n        return True\"}}\n"
    )
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": query}
    ]
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    return tokenizer.decode(text)

# 批量生成提示词
print("生成提示词...")
prompts = [generate_prompt(item["Query"]) for item in queries]

# 批量推理
print("开始批量推理...")
start_time = time.time()
outputs = llm.generate(prompts, sampling_params)
end_time = time.time()
print(f"推理完成，耗时{end_time - start_time:.2f}秒")

# 整理结果
results = []
for item, output in zip(queries, outputs):
    results.append({
        "Query": item["Query"],
        "Output": output.outputs[0].text
    })

# 保存结果
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"结果已保存至{OUTPUT_JSON}")