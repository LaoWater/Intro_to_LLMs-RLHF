from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load tokenizer and model, then move model to GPU
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2").to("cuda")

# Set padding token as EOS token
tokenizer.pad_token = tokenizer.eos_token

# Example: Padding two sequences
sequences = ["Hello, how are you?", "I'm fine."]

# Tokenize and pad sequences
encoded = tokenizer(sequences, padding=True, truncation=True, return_tensors='pt')

# Check the padded output
print(encoded['input_ids'])
print(encoded['attention_mask'])



# Output Explanation
#
# 1. Tokenization
# The sequences list contains two strings:
#
# "Hello, how are you?" (5 tokens)
# "I'm fine." (3 tokens)
# The tokenizer converts these strings into token IDs, which are numerical representations of the tokens that the model can understand.
#
# When we run:
# encoded = tokenizer(sequences, padding=True, truncation=True, return_tensors='pt')
# It performs the following steps:
#
# Tokenization: Each word and punctuation mark is mapped to a corresponding token ID from GPT-2’s vocabulary.
#
# For "Hello, how are you?", the tokens might look like:
#
# "Hello" → token ID 15496
# "," → token ID 11
# "how" → token ID 2322
# "are" → token ID 318
# "you?" → token ID 257
# For "I'm fine.", the tokens might be:
#
# "I'm" → token ID 40
# "fine" → token ID 922
# "." → token ID 13
# Padding: The sequences are padded to ensure they are the same length. The first sequence has 5 tokens, and the second sequence has 3 tokens. To create uniform-length sequences, padding tokens (50256, which is the EOS token) are added to the shorter sequence.
#
# Truncation: If a sequence were too long, it would be truncated to the max_length (though in this example, the sequences are short enough not to be truncated).
#
# Return as Tensors: Setting return_tensors='pt' converts the tokenized outputs into PyTorch tensors, which are required for processing by the model.
#
# 2. Understanding the Output
# The encoded['input_ids'] and encoded['attention_mask'] you get are two tensors that represent the tokenized sequences and their padding.
#
# a. input_ids Tensor
# This tensor contains the token IDs for each word in your sequences, with padding added to make the sequences the same length.
# output:
# tensor([[15496,    11,  2322,   318,   257,  50256,  50256],
#         [   40,   922,    13,  50256,  50256,  50256,  50256]])
# Explanation:
#
# First sequence ("Hello, how are you?"):
#
# "Hello" → 15496
# "," → 11
# "how" → 2322
# "are" → 318
# "you?" → 257
# The first sequence has 5 tokens, and then two 50256 tokens are added as padding.
# Second sequence ("I'm fine."):
#
# "I'm" → 40
# "fine" → 922
# "." → 13
# This sequence has 3 tokens, so four 50256 tokens (padding) are added to make it the same length as the first sequence.
# b. attention_mask Tensor
# The attention mask is a binary mask (1s and 0s) that indicates which tokens are real and which are padding. The model uses this mask to ignore the padded tokens during training or inference.
#
# Here’s the output:
# tensor([[1, 1, 1, 1, 1, 0, 0],
#         [1, 1, 1, 0, 0, 0, 0]])
# Explanation:
#
# First sequence ("Hello, how are you?"):
#
# All 5 real tokens are marked with 1s.
# The two padding tokens are marked with 0s, so the model will ignore them.
# Second sequence ("I'm fine."):
#
# The first 3 real tokens are marked with 1s.
# The remaining 4 padding tokens are marked with 0s.
# This way, during training or inference, the model focuses only on the real tokens and disregards the padding tokens.
