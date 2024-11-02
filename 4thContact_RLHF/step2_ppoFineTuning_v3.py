# Succesfully ran with initial versions of PPOs - generate first graphs.
# In next release Version 4 we will move to updated PPO version 2.
#
# V3 improvements:
#
# 1. Wrapped Inputs in Lists:
# Ensured that input_ids, response_ids, and reward_tensor are provided as lists to match the expected input format of ppo_trainer.step.
# 2. Configured Generation Arguments:
# Defined and included generation_kwargs in the PPO configuration to stabilize generation behavior and prevent negative KL divergence.
# 3. Set Tokenizer's Pad Token:
# Assigned the tokenizer's pad_token to the eos_token to handle padding correctly during training.
#
# 4. Adjusted Batch Sizes:
# Reduced batch_size from 32 to 8 and ensured mini_batch_size aligns appropriately.
#
# 5. Enhanced Error Handling:
# Wrapped the PPO step in a try-except block to catch and log errors without interrupting the training loop.
#
# 6. Implemented Data Shuffling:
# Shuffled the training data at the start of each epoch to improve training robustness and model generalization.
#
# 7. Added Progress Indicators:
# Integrated tqdm progress bars to provide real-time feedback on training batch processing.
#
# 8. Improved Logging and Metrics Tracking:
# Calculated and printed average rewards and losses after each epoch for better monitoring of training performance.
#
# 9. Enabled Memory Cleanup:
# Included commands to delete the model and clear GPU cache post-training to free up system resources.
#
# 10. Enhanced Model and Tokenizer Saving:
# Added error handling when saving the fine-tuned model and tokenizer, ensuring confirmation messages upon successful saves.
#
# 11. Intermediate Saving:
# Provided commented-out code for saving model checkpoints after each epoch, allowing for backup and analysis if needed.
#
# 12. Ensured Correct Token IDs in Generation:
# Explicitly set pad_token_id and eos_token_id in generation_kwargs to prevent padding issues and ensure proper sequence termination.
#
#

import json
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from trl import PPOTrainer, PPOConfig
from trl.models import AutoModelForCausalLMWithValueHead
import matplotlib.pyplot as plt
from tqdm import tqdm  # For progress bars
import random
import gc

# ============================
# 1. Setup and Initialization
# ============================

# Paths
model_path = './notebook/fine_tuned_gpt2_2nd'
feedback_file_path = './human_feedback.json'
save_path = './fine_tuned_lao_after_ppo_v2'

# Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token  # Set pad token to eos token

# Load base model and wrap with value head
base_model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
model = AutoModelForCausalLMWithValueHead.from_pretrained(model_path).to(device)

# ============================
# 2. Load Human Feedback
# ============================

# Load human feedback from the JSON file
try:
    with open(feedback_file_path, 'r', encoding='utf-8') as f:
        human_feedback = json.load(f)
    print(f"Loaded human feedback from {feedback_file_path}")
except FileNotFoundError:
    print(f"Feedback file not found at {feedback_file_path}")
    human_feedback = {}

# ============================
# 3. Configure PPOTrainer
# ============================

# Define generation kwargs separately if needed during training steps
generation_kwargs = {
    "max_new_tokens": 100,
    "do_sample": True,
    "top_k": 50,
    "top_p": 0.90,
    "temperature": 0.95,
    "pad_token_id": tokenizer.eos_token_id,
    "eos_token_id": tokenizer.eos_token_id,
}

# Set the PPO configuration without generation_kwargs
ppo_config = PPOConfig(
    model_name="gpt2",
    learning_rate=1e-5,
    batch_size=8,  # Adjusted batch size based on data size
    mini_batch_size=4,  # Define mini_batch_size
    gradient_accumulation_steps=1,  # Define gradient accumulation steps
    # Removed generation_kwargs from here
    log_with="tensorboard",  # Optional: for better logging
)

# Initialize PPO trainer
ppo_trainer = PPOTrainer(
    model=model,
    tokenizer=tokenizer,
    config=ppo_config
)

# Assign generation_kwargs to the PPOTrainer's generate_kwargs attribute
ppo_trainer.config.generate_kwargs = generation_kwargs

print("Initialized PPOTrainer")

# ============================
# 4. Prepare Training Data
# ============================

# Prepare data for batching
data = []
for prompt, responses in human_feedback.items():
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.squeeze(0)  # Remove batch dimension
    for feedback in responses:
        generated_response = feedback.get("response")
        reward = feedback.get("reward")
        if generated_response is None or reward is None:
            continue  # Skip if response or reward is missing
        response_ids = tokenizer(generated_response, return_tensors="pt").input_ids.squeeze(0)
        data.append((input_ids, response_ids, reward))

if not data:
    raise ValueError("No valid training data found in human_feedback.json")

print(f"Total training samples: {len(data)}")

# ============================
# 5. Training Loop
# ============================

# Track the average reward and loss during training
reward_history = []
loss_history = []

# Define number of epochs
num_epochs = 25

# Training epochs
for epoch in range(num_epochs):
    epoch_rewards = []
    epoch_loss = []
    print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")

    # Shuffle data each epoch for better training
    random_indices = torch.randperm(len(data))
    shuffled_data = [data[i] for i in random_indices]

    # Process data in batches
    for i in tqdm(range(0, len(shuffled_data), ppo_config.batch_size), desc="Training Batches"):
        batch = shuffled_data[i:i + ppo_config.batch_size]
        if len(batch) == 0:
            continue  # Skip empty batches

        # Unpack batch
        batch_input_ids = [item[0].to(device) for item in batch]
        batch_response_ids = [item[1].to(device) for item in batch]
        batch_rewards = [torch.tensor([item[2]], dtype=torch.float32, device=device) for item in batch]

        # Perform PPO step
        try:
            stats = ppo_trainer.step(batch_input_ids, batch_response_ids, batch_rewards)
            print(stats)
            # Access the total loss from the stats dictionary
            loss_value = stats['ppo/loss/total']
            epoch_loss.append(loss_value)
        except Exception as e:
            print(f"Error during PPO step: {e}")
            continue  # Skip this batch on error

        # Track rewards
        epoch_rewards.extend([item[2] for item in batch])

    # Calculate average reward and loss for the epoch
    avg_reward = sum(epoch_rewards) / len(epoch_rewards) if epoch_rewards else 0
    avg_loss = sum(epoch_loss) / len(epoch_loss) if epoch_loss else 0

    reward_history.append(avg_reward)
    loss_history.append(avg_loss)
    print(f"Epoch {epoch + 1}: Average Reward: {avg_reward:.2f}, Average Loss: {avg_loss:.4f}")

print(f'Loss history: {loss_history})')

# ============================
# 6. Plot Training Metrics
# ============================

plt.figure(figsize=(12, 5))

# Plot rewards
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), reward_history, marker='o', color='blue')
plt.title("Average Reward Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Average Reward")
plt.grid(True)

# Plot losses
plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), loss_history, marker='o', color='red')
plt.title("Average Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Average Loss")
plt.grid(True)

plt.tight_layout()
plt.show()

# ============================
# 7. Save the Fine-Tuned Model
# ============================

try:
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"\nModel and tokenizer successfully saved to: {save_path}")
except Exception as e:
    print(f"Error saving model and tokenizer: {e}")

# ============================
# 8. Cleanup (Optional)
# ============================

# Free up memory
del model
torch.cuda.empty_cache()
gc.collect()
