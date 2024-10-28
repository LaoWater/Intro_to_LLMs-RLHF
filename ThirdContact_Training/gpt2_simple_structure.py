from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import textwrap

# Load tokenizer and model, then move model to GPU
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2").to("cuda")


prompt = "Imagine a perfect afternoon of simple joys and reflections."

input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
gen_tokens = model.generate(
    input_ids,
    do_sample=True,
    temperature=0.9,
    max_length=100,
)
gen_text = tokenizer.batch_decode(gen_tokens)[0]

print(gen_text)
