from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import json

# Guideline for Human Rating Scale (1-5):
# Use a simple numerical scale to rank the responses based on how well they meet the criteria:
# 5: Excellent – The response is highly coherent, creative, relevant, and fluent.
# 4: Good – The response is mostly coherent, creative, and relevant, with minor issues.
# 3: Average – The response makes sense but lacks depth or creativity.
# 2: Below Average – The response is somewhat incoherent or irrelevant.
# 1: Poor – The response is nonsensical or completely off-topic.


# Load the fine-tuned GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(
    r'C:\Users\baciu\Desktop\Neo Training\G2I\ThirdContact_Training\notebook\fine_tuned_gpt2')
model = GPT2LMHeadModel.from_pretrained(
    r'C:\Users\baciu\Desktop\Neo Training\G2I\ThirdContact_Training\notebook\fine_tuned_gpt2').to('cuda')

# List of prompts
prompts = [
    "Imagine a perfect afternoon of simple joys and reflections.",
    "What makes a morning truly beautiful?",
    "Describe a healing meal that comforts the body and soul.",
    "What is the meaning of life?",
    "What is consciousness?",
    "The Natural Truths of the Mind are...",
    "The Natural Truths of the Body are...",
    "Is there Free Will?",
    "To live a life full-hearted is to...",
    "To bake or to cut an onion"
]

# Number of responses to generate per prompt
num_responses_per_prompt = 3

# Initialize an empty dictionary to hold the human feedback structure
human_feedback = {}

# Generate multiple responses for each prompt and add them to the human_feedback dictionary
for prompt in prompts:
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to('cuda')

    # Initialize prompt entry if not already in the dictionary
    if prompt not in human_feedback:
        human_feedback[prompt] = []

    # Generate multiple responses per prompt
    for i in range(num_responses_per_prompt):
        output = model.generate(input_ids, max_length=80, temperature=0.9, top_p=0.95, do_sample=True)
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

        # Add the generated response with a placeholder for reward
        human_feedback[prompt].append({
            "response": generated_text,
            "reward": None  # Placeholder for human feedback to input a reward later
        })

        # Print the generated response to show progress
        print(f"Prompt: {prompt}\nGenerated Response {i + 1}: {generated_text}\n")

# Save the human feedback structure to a JSON file
feedback_file_path = r'human_feedback.json'
with open(feedback_file_path, 'w', encoding='utf-8') as f:
    json.dump(human_feedback, f, indent=4)

print(f"Human feedback structure saved to: {feedback_file_path}")

# Rating Scale (1-5):
# Use a simple numerical scale to rank the responses based on how well they meet the criteria above:
# 5: Excellent – The response is highly coherent, creative, relevant, and fluent.
# 4: Good – The response is mostly coherent, creative, and relevant, with minor issues.
# 3: Average – The response makes sense but lacks depth or creativity.
# 2: Below Average – The response is somewhat incoherent or irrelevant.
# 1: Poor – The response is nonsensical or completely off-topic.
