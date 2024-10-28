from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import textwrap

print(torch.version.cuda)
print(torch.cuda.is_available())


def run_gpt2_cpu(input_text, max_length=150, temperature=0.7, top_k=50, repetition_penalty=1.2):
    """
    Runs GPT-2 model on CPU with specified generation parameters.
    """
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")

    # Explicitly set the pad_token_id to eos_token_id
    model.config.pad_token_id = model.config.eos_token_id

    # Tokenize input
    input_ids = tokenizer(input_text, return_tensors="pt", max_length=500, truncation=True).input_ids

    # Generate output with improved control over generation
    outputs = model.generate(
        input_ids,
        max_new_tokens=max_length,  # Limit output length
        temperature=temperature,  # Control randomness
        top_k=top_k,  # Top-K sampling
        repetition_penalty=repetition_penalty  # Penalize repetition
    )

    # Decode the output and skip special tokens for cleaner output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    formatted_text = textwrap.fill(generated_text, width=120)
    print(formatted_text)


def run_gpt2_gpu(input_text, max_length=150, temperature=0.7, top_k=50, repetition_penalty=1.2):
    """
    Runs GPT-2 model on GPU with specified generation parameters.
    """
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2").to("cuda")

    # Explicitly set the pad_token_id to eos_token_id
    model.config.pad_token_id = model.config.eos_token_id

    # Tokenize input and move to GPU
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")

    # Generate output with improved control over generation
    outputs = model.generate(
        input_ids,
        max_new_tokens=max_length,  # Limit output length
        temperature=temperature,  # Control randomness
        top_k=top_k,  # Top-K sampling
        repetition_penalty=repetition_penalty  # Penalize repetition
    )

    # Decode the output and skip special tokens for cleaner output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return textwrap.fill(generated_text, width=120)


#################
## Main Script ##
#################

if __name__ == "__main__":
    input_text_cpu = "to love an onion means to"
    print("Running GPT-2 on CPU:")
    run_gpt2_cpu(input_text=input_text_cpu)

    if torch.cuda.is_available():
        input_text_gpu = "to suck an onion means to"
        model_answer = run_gpt2_gpu(input_text=input_text_gpu)
        print("\nRunning GPT-2 on GPU:")
        print(model_answer)
    else:
        print("\nCUDA is not available. Skipping GPU execution.")
