from transformers import GPTNeoForCausalLM, GPT2Tokenizer
import torch
import textwrap




def run_gptneo_cpu(input_text, max_new_tokens=50, temperature=0.7, top_k=50, repetition_penalty=1.2):
    """
    Runs the GPT-Neo model on CPU with specified generation parameters.

    Args:
        input_text (str): The input text prompt for the model.
        max_new_tokens (int): Maximum number of tokens to generate.
        temperature (float): Sampling temperature; higher values mean more randomness.
        top_k (int): Number of highest probability vocabulary tokens to keep for top-k-filtering.
        repetition_penalty (float): Penalty for repeated tokens.

    Returns:
        None: Prints the generated text output.
    """
    # Load tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
    model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")

    # GPT-Neo uses the same pad_token_id as eos_token_id by default. Ensure proper handling.
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    else:
        model.config.pad_token_id = tokenizer.pad_token_id

    # Tokenize input text with padding and attention mask
    encoding = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=512,
        padding="max_length",  # Pad to max_length
        truncation=True,
        return_attention_mask=True
    )
    input_ids = encoding.input_ids
    attention_mask = encoding.attention_mask

    # Generate output with specified parameters
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,  # Pass attention mask
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        do_sample=True,  # Enable sampling for diversity
        pad_token_id=model.config.pad_token_id,  # Ensure proper
        no_repeat_ngram_size=2
    )

    # Decode and format the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    formatted_text = textwrap.fill(generated_text, width=120)
    print("CPU Generated Output:")
    print(formatted_text)


def run_gptneo_gpu(input_text, max_new_tokens=150, temperature=0.9, top_k=150, repetition_penalty=1.2):
    """
    Runs the GPT-Neo model on GPU with specified generation parameters.

    Args:
        input_text (str): The input text prompt for the model.
        max_new_tokens (int): Maximum number of tokens to generate.
        temperature (float): Sampling temperature; higher values mean more randomness.
        top_k (int): Number of highest probability vocabulary tokens to keep for top-k-filtering.
        repetition_penalty (float): Penalty for repeated tokens.

    Returns:
        str: The generated text output.
    """
    # Load tokenizer and model, then move model to GPU
    tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
    model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B").to("cuda")

    # GPT-Neo uses the same pad_token_id as eos_token_id by default. Ensure proper handling.
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    else:
        model.config.pad_token_id = tokenizer.pad_token_id

    # Tokenize input text with padding and attention mask
    encoding = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=512,
        padding="max_length",  # Pad to max_length
        truncation=True,
        return_attention_mask=True
    )
    input_ids = encoding.input_ids.to("cuda")
    attention_mask = encoding.attention_mask.to("cuda")

    # Generate output with specified parameters
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,  # Pass attention mask
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        do_sample=True,  # Enable sampling for diversity
        pad_token_id=model.config.pad_token_id,  # Ensure proper padding
        no_repeat_ngram_size=5
    )

    # Decode and format the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    formatted_text = textwrap.fill(generated_text, width=120)
    return formatted_text


#################
## Main Script ##
#################

if __name__ == "__main__":

    # Print CUDA version and availability
    print("CUDA Version:", torch.version.cuda)
    print("Is CUDA Available?", torch.cuda.is_available())

    # Define the input prompt
    input_prompt = "to love an onion means to"

    print("\nRunning GPT-Neo on CPU:")
    run_gptneo_cpu(input_text=input_prompt)

    # Check if CUDA (GPU) is available before attempting GPU execution
    if torch.cuda.is_available():
        print("\nRunning GPT-Neo on GPU:")
        gpu_generated_output = run_gptneo_gpu(input_text=input_prompt)
        print(gpu_generated_output)
    else:
        print("\nCUDA is not available. Skipping GPU execution.")
