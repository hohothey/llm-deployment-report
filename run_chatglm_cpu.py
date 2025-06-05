from transformers import AutoModel, AutoTokenizer
import sys

if len(sys.argv) > 1:
    question = sys.argv[1]
else:
    question = "请解释量子计算的基本原理"

model_name = "/mnt/data/chatglm3-6b"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True).float().eval()

response, history = model.chat(tokenizer, question, history=[])
print(response)
