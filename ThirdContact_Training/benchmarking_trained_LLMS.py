# Load the fine-tuned model
from transformers import GPT2Tokenizer, GPT2LMHeadModel


def gpt2_base(prompt):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to("cuda")

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    gen_tokens = model.generate(
        input_ids,
        do_sample=True,
        temperature=0.9,
        max_length=100,
    )
    gen_text = tokenizer.batch_decode(gen_tokens)[0]
    return gen_text


def lao_alpha_v2(prompt):
    # Load tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained('released_trained_models/fine_tuned_gpt2_2nd')
    model = GPT2LMHeadModel.from_pretrained('released_trained_models/fine_tuned_gpt2_2nd').to('cuda')

    # Tokenize the prompt
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to('cuda')

    # Generate text with the fine-tuned model
    output = model.generate(input_ids, max_length=100, temperature=0.9, top_p=0.95, do_sample=True)

    # Decode the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    return generated_text


def lao_alpha_rlhf(prompt):
    # Load tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained('released_trained_models/fine_tuned_gpt2_after_ppo')
    model = GPT2LMHeadModel.from_pretrained('released_trained_models/fine_tuned_gpt2_after_ppo').to('cuda')

    # Tokenize the prompt
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to('cuda')

    # Generate text with the fine-tuned model
    output = model.generate(input_ids, max_length=100, temperature=0.1, top_p=0.9, do_sample=True)

    # Decode the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    return generated_text


def main():
    prompt = "Walking Alone at Night"

    # Call gpt2_base model
    base_output = gpt2_base(prompt)
    print(f"GPT-2 Base Output:\n{base_output}\n")

    # Call lao_alpha_v2 model
    alpha_v2_output = lao_alpha_v2(prompt)
    print(f"LAO Alpha V2 Output:\n{alpha_v2_output}\n")

    # Call lao_alpha_rlhf model
    alpha_rlhf_output = lao_alpha_rlhf(prompt)
    print(f"LAO Alpha RLHF Output:\n{alpha_rlhf_output}\n")


if __name__ == "__main__":
    main()
