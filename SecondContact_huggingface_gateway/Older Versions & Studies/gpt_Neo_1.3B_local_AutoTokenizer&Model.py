from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import textwrap

print("CUDA Version:", torch.version.cuda)
print("Is CUDA Available?", torch.cuda.is_available())


def run_gptneo_cpu(input_text, max_length=150, temperature=0.7, top_k=50, repetition_penalty=1.2):
    """
    Runs GPT-Neo model on CPU with specified generation parameters.
    """
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")

    # Set pad_token_id to eos_token_id to avoid warnings
    model.config.pad_token_id = model.config.eos_token_id

    # Tokenize input
    input_ids = tokenizer(input_text, return_tensors="pt", max_length=500, truncation=True).input_ids

    # Generate output with controlled parameters
    outputs = model.generate(
        input_ids,
        max_new_tokens=max_length,  # Limit output length
        temperature=temperature,  # Control randomness
        top_k=top_k,  # Top-K sampling
        repetition_penalty=repetition_penalty,  # Penalize repetition
        do_sample=True,  # Enable sampling
        pad_token_id=model.config.pad_token_id  # Avoid errors with padding
    )

    # Decode the output and format it
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    formatted_text = textwrap.fill(generated_text, width=120)
    print(formatted_text)


def run_gptneo_gpu(input_text, max_length=150, temperature=0.7, top_k=50, repetition_penalty=1.2):
    """
    Runs GPT-Neo model on GPU with specified generation parameters.
    """
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B").to("cuda")

    # Set pad_token_id to eos_token_id to avoid warnings
    model.config.pad_token_id = model.config.eos_token_id

    # Tokenize input and move to GPU
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")

    # Generate output with controlled parameters
    outputs = model.generate(
        input_ids,
        max_new_tokens=max_length,  # Limit output length
        temperature=temperature,  # Control randomness
        top_k=top_k,  # Top-K sampling
        repetition_penalty=repetition_penalty,  # Penalize repetition
        do_sample=True,  # Enable sampling
        pad_token_id=model.config.pad_token_id  # Avoid errors with padding
    )

    # Decode the output and format it
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    formatted_text = textwrap.fill(generated_text, width=120)
    return formatted_text


#################
## Main Script ##
#################

if __name__ == "__main__":
    input_text_main = "The future of AI is"

    print("Running GPT-Neo on CPU:")
    run_gptneo_cpu(input_text=input_text_main)

    if torch.cuda.is_available():
        print("\nRunning GPT-Neo on GPU:")
        model_answer = run_gptneo_gpu(input_text=input_text_main)
        print(model_answer)
    else:
        print("\nCUDA is not available. Skipping GPU execution.")
