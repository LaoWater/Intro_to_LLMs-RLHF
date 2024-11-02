# Load the fine-tuned model
from transformers import GPT2Tokenizer, GPT2LMHeadModel


def lao_alpha_rlhf(prompt):
    # Load tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained('./fine_tuned_lao_after_ppo_v2')
    model = GPT2LMHeadModel.from_pretrained('./fine_tuned_lao_after_ppo_v2').to('cuda')

    # Prompt for text generation
    prompt = "Imagine a perfect afternoon of simple joys and reflections."

    # Tokenize the prompt
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to('cuda')

    # Generate text with the fine-tuned model
    output = model.generate(input_ids, max_length=133, temperature=0.9, top_p=0.9, do_sample=True)

    # Decode the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    return generated_text


input_text = 'Walking alone at night'
model_response = lao_alpha_rlhf(input_text)
print(model_response)