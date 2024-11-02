# Fallback to known models

import textwrap
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch

# Enabling cuda & matching pytorch version
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
print(torch.version.cuda)
print(torch.cuda.is_available())

# Step 1: Load the pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121model and tokenizer for text generation
model_id = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Step 2: Setup the pipeline
text_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device="cuda")

# Step 3: Pass input data to the pipeline
text_input = "The future of AI is"
result = text_pipe(text_input, max_length=444, truncation=True)

transcription = result[0]["generated_text"]

# Format the transcription
formatted_transcription = textwrap.fill(transcription, width=120)

# Print the formatted transcription
print(f"Transcription\n{formatted_transcription}")
print("-" * 50)  # Separator for readability
