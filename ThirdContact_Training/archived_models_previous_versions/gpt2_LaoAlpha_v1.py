# Lao alpha v1 was trained on the following Args[]:
#
# training_args = TrainingArguments(
#     output_dir='./results',
#     num_train_epochs=3,               # Start with 3 epochs
#     per_device_train_batch_size=1,    # Use small batch size due to memory constraints
#     gradient_accumulation_steps=8,    # Accumulate gradients to simulate batch size of 8
#     learning_rate=5e-5,               # Standard learning rate for fine-tuning GPT-2
#     evaluation_strategy="steps",      # Evaluate the model every few steps
#     eval_steps=500,                   # Evaluate every 500 steps
#     save_steps=500,                   # Save model checkpoint every 500 steps
#     logging_steps=100,                # Log training metrics every 100 steps
#     fp16=True,                        # Use mixed-precision training for efficiency
#     logging_dir='./logs',             # Save logs
#     save_total_limit=2                # Keep only the last 2 checkpoints
# )


# Load the fine-tuned model
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('released_trained_models/fine_tuned_gpt2')
model = GPT2LMHeadModel.from_pretrained('released_trained_models/fine_tuned_gpt2').to('cuda')

# Prompt for text generation
prompt = "Imagine a perfect afternoon of simple joys and reflections."

# Tokenize the prompt
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to('cuda')

# Generate text with the fine-tuned model
output = model.generate(input_ids, max_length=133, temperature=0.9, top_p=0.95, do_sample=True)

# Decode the generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
