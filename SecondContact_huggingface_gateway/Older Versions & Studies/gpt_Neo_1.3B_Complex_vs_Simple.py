import textwrap

from transformers import GPTNeoForCausalLM, GPT2Tokenizer

model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B").to("cuda")
tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")

prompt = (
    "Imagine a perfect afternoon of simple joys and reflections."
)

input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")

gen_tokens = model.generate(
    input_ids,
    do_sample=True,
    temperature=0.9,
    max_length=100,
)
gen_text = tokenizer.batch_decode(gen_tokens)[0]

print(gen_text)


# Using Top-K seems to reaaally mess up the model output - making it strongly hallucinating off-subject.

def complex_model_call(input_text, max_new_tokens=150, temperature=0.9, top_k=50, repetition_penalty=1.2):
    # Tokenize input text with padding and attention mask
    encoding = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=512,
        padding="max_length",  # Pad to max_length
        truncation=True,
        return_attention_mask=True
    )
    input_ids_f = encoding.input_ids.to("cuda")
    attention_mask = encoding.attention_mask.to("cuda")

    # Generate output with specified parameters
    outputs = model.generate(
        input_ids=input_ids_f,
        attention_mask=attention_mask,  # Pass attention mask
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        # top_k=top_k,
        # repetition_penalty=repetition_penalty,
        do_sample=True,  # Enable sampling for diversity
        pad_token_id=model.config.pad_token_id,  # Ensure proper padding
        # no_repeat_ngram_size=5
    )

    # Decode and format the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    formatted_text = textwrap.fill(generated_text, width=120)
    print(formatted_text)


print("\nCalling with complicated parameters:")
complex_model_call(prompt)
