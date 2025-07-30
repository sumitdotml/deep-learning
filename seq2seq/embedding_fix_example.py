import torch
import torch.nn as nn

# Create an embedding layer (like in your decoder)
vocab_size = 1000
embedding_dim = 128
embedding = nn.Embedding(vocab_size, embedding_dim)

print("=== DEMONSTRATING THE ERROR AND FIX ===\n")

# 🚫 This causes the RuntimeError (DON'T RUN)
print("❌ Problem: FloatTensor input")
input_wrong = torch.tensor([1.0, 5.0, 10.0])  # FloatTensor
print(f"   Input: {input_wrong}")
print(f"   Dtype: {input_wrong.dtype}")
print("   This would cause: RuntimeError: Expected tensor for argument #1 'indices' to have one of the following scalar types: Long, Int; but got torch.FloatTensor instead\n")

# ✅ Solution 1: Convert to integers
print("✅ Solution 1: Convert FloatTensor to LongTensor")
input_fixed = input_wrong.long()  # THE KEY FIX!
print(f"   Input after .long(): {input_fixed}")
print(f"   Dtype after .long(): {input_fixed.dtype}")

# Add proper dimensions for embedding
input_unsqueezed = input_fixed.unsqueeze(0)  # [1, 3] for batch processing
embedded_output = embedding(input_unsqueezed)
print(f"   ✅ Success! Embedded shape: {embedded_output.shape}\n")

# ✅ Solution 2: Create integers directly
print("✅ Solution 2: Use integer indices from the start")
input_correct = torch.tensor([1, 5, 10])  # Already LongTensor
print(f"   Input: {input_correct}")
print(f"   Dtype: {input_correct.dtype}")

input_unsqueezed = input_correct.unsqueeze(0)
embedded_output = embedding(input_unsqueezed)
print(f"   ✅ Success! Embedded shape: {embedded_output.shape}\n")

# ✅ Solution 3: For your decoder context
print("✅ Solution 3: Decoder scenario (from your code)")
batch_size = 3
decoder_input = torch.tensor([10, 25, 7])  # Token indices for each batch item
print(f"   Decoder input: {decoder_input}")
print(f"   Dtype: {decoder_input.dtype}")

# This is what happens in decoder.py line 46
input_unsqueezed = decoder_input.unsqueeze(0)  # [1, batch_size]
print(f"   After unsqueeze(0): {input_unsqueezed.shape}")

# This is what happens in decoder.py line 51
embedded_input = embedding(input_unsqueezed)
print(f"   ✅ Success! Final embedded shape: {embedded_input.shape}")

print("\n=== DEBUGGING FUNCTION ===")

def debug_tensor_for_embedding(tensor, name="tensor"):
    """Helper function to check if tensor is ready for embedding layer"""
    print(f"\n🔍 Debugging {name}:")
    print(f"   Shape: {tensor.shape}")
    print(f"   Dtype: {tensor.dtype}")
    print(f"   Min value: {tensor.min().item()}")
    print(f"   Max value: {tensor.max().item()}")
    
    if tensor.dtype in [torch.float32, torch.float64]:
        print(f"   ⚠️  WARNING: Dtype is {tensor.dtype}, but embedding expects integers!")
        print(f"   💡 FIX: Use tensor.long() to convert to integers")
        print(f"   Fixed tensor: {tensor.long()}")
    else:
        print(f"   ✅ Dtype {tensor.dtype} is compatible with embedding layers")

# Test the debugging function
test_float = torch.tensor([1.0, 2.0, 3.0])
test_int = torch.tensor([1, 2, 3])

debug_tensor_for_embedding(test_float, "FloatTensor")
debug_tensor_for_embedding(test_int, "LongTensor")