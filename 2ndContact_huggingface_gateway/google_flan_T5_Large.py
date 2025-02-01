from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import textwrap


def run_flant5_large_cpu(input_text, max_new_tokens=50, temperature=0.7, top_k=50, repetition_penalty=1.2):
    """
    Runs the Flan-T5 Large model on CPU with specified generation parameters.

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
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")

    # Ensure pad_token_id is set correctly (usually pad_token_id is already set in T5 models)
    # Only set pad_token_id to eos_token_id if a distinct pad token does not exist
    if tokenizer.pad_token_id is None:
        print("Pad token not found. Setting pad_token_id to eos_token_id.")
        model.config.pad_token_id = model.config.eos_token_id
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
        do_sample=True  # Enable sampling for diversity
    )

    # Decode and format the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    formatted_text = textwrap.fill(generated_text, width=120)
    print("CPU Generated Output:")
    print(formatted_text)


def run_flant5_large_gpu(input_text, max_new_tokens=150, temperature=0.7, top_k=50, repetition_penalty=1.2):
    """
    Runs the Flan-T5 Large model on GPU with specified generation parameters.

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
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large").to("cuda")

    # Ensure pad_token_id is set correctly (usually pad_token_id is already set in T5 models)
    if tokenizer.pad_token_id is None:
        print("Pad token not found. Setting pad_token_id to eos_token_id.")
        model.config.pad_token_id = model.config.eos_token_id
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
        do_sample=True  # Enable sampling for diversity
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
    input_prompt = "to suck an onion means to"

    print("\nRunning Flan-T5 Large on CPU:")
    run_flant5_large_cpu(input_text=input_prompt)

    # Check if CUDA (GPU) is available before attempting GPU execution
    if torch.cuda.is_available():
        print("\nRunning Flan-T5 Large on GPU:")
        gpu_generated_output = run_flant5_large_gpu(input_text=input_prompt)
        print(gpu_generated_output)
    else:
        print("\nCUDA is not available. Skipping GPU execution.")
