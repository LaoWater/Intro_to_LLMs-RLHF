from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("Zyphra/Zamba2-2.7B")
model = AutoModelForCausalLM.from_pretrained("Zyphra/Zamba2-2.7B", device_map="auto",
                                             torch_dtype=torch.bfloat16, use_mamba_kernels=False)

input_text = "What factors contributed to the fall of the Roman Empire?"
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

outputs = model.generate(**input_ids, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))
