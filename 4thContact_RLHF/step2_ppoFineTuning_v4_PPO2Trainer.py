import copy
import json
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from trl import PPOv2Trainer, PPOv2Config  # Updated import
from trl.models import AutoModelForCausalLMWithValueHead
import matplotlib.pyplot as plt
from tqdm import tqdm  # For progress bars
import random
import gc

# ============================
# 1. Setup and Initialization
# ============================

# Paths
model_path = r'C:\Users\baciu\Desktop\Neo Training\G2I\ThirdContact_Training\notebook\fine_tuned_gpt2'
feedback_file_path = './human_feedback.json'
save_path = './fine_tuned_gpt2_after_ppo'

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
# 3. Configure PPOv2Trainer
# ============================

# Define generation kwargs
generation_kwargs = {
    "max_length": 50,
    "do_sample": True,
    "top_k": 50,
    "top_p": 0.95,
    "temperature": 1.0,
    "pad_token_id": tokenizer.eos_token_id,
    "eos_token_id": tokenizer.eos_token_id,
}

# Set the PPOv2 configuration
ppo_config = PPOv2Config(
    exp_name="ppo_experiment",          # Name of the experiment
    reward_model_path="EleutherAI/pythia-160m",  # Path to reward model
    num_ppo_epochs=4,                   # Number of PPO epochs
    whiten_rewards=False,               # Whether to whiten rewards
    kl_coef=0.05,                       # KL coefficient
    cliprange=0.2,                      # Clip range
    vf_coef=0.1,                        # Value function coefficient
    cliprange_value=0.2,                # Clip range for value function
    gamma=1.0,                          # Discount factor
    lam=0.95,                              # Lambda for GAE
    output_dir='./fine_tuned_gpt2_after_ppo'
    # Additional parameters can be added here if needed
)

# Create a deep copy of base_model
base_model_ref = copy.deepcopy(base_model)

# Initialize PPOv2 trainer
ppo_trainer = PPOv2Trainer(
    config=ppo_config,
    tokenizer=tokenizer,
    policy=base_model,                     # Use the base_model as policy
    ref_policy=base_model_ref,                 # Reference policy, ideally a copy of the base_model
    reward_model=model,                    # The reward model could be the value-head model or a separate reward model
    train_dataset=human_feedback,      # The training dataset for PPO
    # generation_kwargs=generation_kwargs    # Pass generation_kwargs here
)
print("Initialized PPOv2Trainer")

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
num_epochs = 5

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
            loss = ppo_trainer.step(batch_input_ids, batch_response_ids, batch_rewards)
        except Exception as e:
            print(f"Error during PPO step: {e}")
            continue  # Skip this batch on error

        # Track rewards and loss
        epoch_rewards.extend([item[2] for item in batch])
        if hasattr(loss, 'item'):
            epoch_loss.append(loss.item())
        else:
            loss_value = loss.get("loss", None) if isinstance(loss, dict) else loss
            if loss_value is not None:
                print(f"Loss current epoch: {loss_value}")
                epoch_loss.append(float(loss_value))

    # Calculate average reward and loss for the epoch
    avg_reward = sum(epoch_rewards) / len(epoch_rewards) if epoch_rewards else 0
    avg_loss = sum(epoch_loss) / len(epoch_loss) if epoch_loss else 0

    reward_history.append(avg_reward)
    loss_history.append(avg_loss)
    print(f"Epoch {epoch + 1}: Average Reward: {avg_reward:.2f}, Average Loss: {avg_loss:.4f}")

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
