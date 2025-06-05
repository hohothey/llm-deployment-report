from transformers import TextStreamer, AutoTokenizer, AutoModelForCausalLM
import sys

if len(sys.argv) > 1:
    question = sys.argv[1]
else:
    question = "请解释量子计算的基本原理"

model_name = "/mnt/data/Qwen-7B-Chat"

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype="auto"
).eval()

inputs = tokenizer(question, return_tensors="pt").input_ids

streamer = TextStreamer(tokenizer)
outputs = model.generate(inputs, streamer=streamer, max_new_tokens=300)
