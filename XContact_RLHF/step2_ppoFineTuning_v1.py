import json
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from trl import PPOTrainer, PPOConfig
from trl.models import AutoModelForCausalLMWithValueHead
import matplotlib.pyplot as plt

# Load fine-tuned GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(r'C:\Users\baciu\Desktop\Neo Training\G2I\ThirdContact_Training\notebook\fine_tuned_gpt2')
base_model = GPT2LMHeadModel.from_pretrained(r'C:\Users\baciu\Desktop\Neo Training\G2I\ThirdContact_Training\notebook\fine_tuned_gpt2').to('cuda')

# Wrap the model with a value head using the correct class from trl
model = AutoModelForCausalLMWithValueHead.from_pretrained(r'C:\Users\baciu\Desktop\Neo Training\G2I\ThirdContact_Training\notebook\fine_tuned_gpt2').to('cuda')

# Load human feedback from the JSON file
feedback_file_path = './human_feedback.json'
with open(feedback_file_path, 'r', encoding='utf-8') as f:
    human_feedback = json.load(f)

# Set the PPO configuration
ppo_config = PPOConfig(
    model_name="gpt2",
    learning_rate=1e-5,
    batch_size=32,                    # Make sure batch_size is a multiple of mini_batch_size * gradient_accumulation_steps
    mini_batch_size=4,                 # Define mini_batch_size
    gradient_accumulation_steps=1,     # Define gradient accumulation steps
)

# Initialize PPO trainer
ppo_trainer = PPOTrainer(
    model=model,
    tokenizer=tokenizer,
    config=ppo_config
)

# Track the average reward during training
reward_history = []
loss_history = []

# Run fine-tuning for multiple epochs
num_epochs = 5
for epoch in range(num_epochs):
    epoch_rewards = []
    epoch_loss = []

    for prompt, responses in human_feedback.items():
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to('cuda')

        for feedback in responses:
            generated_response = feedback["response"]
            reward = feedback["reward"]

            # Skip if the reward is missing
            if reward is None:
                continue

            response_ids = tokenizer(generated_response, return_tensors="pt").input_ids.to('cuda')
            # Convert reward to a float tensor directly
            reward_tensor = torch.tensor([reward], dtype=torch.float32, device='cuda')
            # Perform PPO step with the correctly shaped reward tensor
            loss = ppo_trainer.step(input_ids, response_ids, [reward_tensor])

            # Track rewards and loss for each step
            epoch_rewards.append(reward)
            # Check if the returned loss is a tensor and has an item() method
            if hasattr(loss, 'item'):
                epoch_loss.append(loss.item())
            else:
                epoch_loss.append(loss)  # Append the loss directly if it's not a tensor

    # Log the average reward and loss for each epoch
    avg_reward = sum(epoch_rewards) / len(epoch_rewards) if epoch_rewards else 0
    avg_loss = sum(epoch_loss) / len(epoch_loss) if epoch_loss else 0

    reward_history.append(avg_reward)
    loss_history.append(avg_loss)
    print(f"Epoch {epoch + 1}: Average Reward: {reward_history[-1]:.2f}, Average Loss: {loss_history[-1]:.4f}")

# Plot the training rewards and losses
plt.figure(figsize=(10, 4))

# Plot rewards
plt.subplot(1, 2, 1)
plt.plot(reward_history)
plt.title("Average Reward Over Time")
plt.xlabel("Epoch")
plt.ylabel("Reward")

# Plot losses
plt.subplot(1, 2, 2)
plt.plot(loss_history)
plt.title("Training Loss Over Time")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.show()

# Save the fine-tuned model and tokenizer
save_path = './fine_tuned_gpt2_after_ppo'
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print(f"Model and tokenizer saved to: {save_path}")
