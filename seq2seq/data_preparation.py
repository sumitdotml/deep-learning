#!/usr/bin/env python3
"""
Step 1 & 2: Dataset Loading and Small Training Subset Creation
Load WMT19 and create manageable subset for seq2seq learning
"""

from datasets import load_dataset
import json
import os
from collections import Counter


def create_training_subset(num_examples=5000):
    """Create a small subset for learning seq2seq"""

    print(f"Loading {num_examples} examples from WMT19 DE-EN...")

    # Load training and validation splits
    train_dataset = load_dataset(
        "wmt19", "de-en", split=f"train[:{num_examples}]"  # First N examples
    )

    val_dataset = load_dataset(
        "wmt19", "de-en", split="validation[:500]"  # 500 validation examples
    )

    print(f"Training examples: {len(train_dataset)}")
    print(f"Validation examples: {len(val_dataset)}")

    # Convert to simple format for easier processing
    def extract_pairs(dataset):
        pairs = []
        for example in dataset:
            german = example["translation"]["de"].strip()
            english = example["translation"]["en"].strip()

            # Filter out very long sentences (>50 words) for simplicity
            if len(german.split()) <= 50 and len(english.split()) <= 50:
                pairs.append(
                    {
                        "src": german,  # Source (German)
                        "tgt": english,  # Target (English)
                    }
                )
        return pairs

    train_pairs = extract_pairs(train_dataset)
    val_pairs = extract_pairs(val_dataset)

    print(
        f"After filtering (≤50 words): {len(train_pairs)} train, {len(val_pairs)} val"
    )

    # Save to JSON files for easy loading
    os.makedirs("data", exist_ok=True)

    with open("data/train_pairs.json", "w", encoding="utf-8") as f:
        json.dump(train_pairs, f, indent=2, ensure_ascii=False)

    with open("data/val_pairs.json", "w", encoding="utf-8") as f:
        json.dump(val_pairs, f, indent=2, ensure_ascii=False)

    print("Saved to data/train_pairs.json and data/val_pairs.json")

    # Show some examples
    print("\nSample training pairs:")
    for i, pair in enumerate(train_pairs[:3]):
        print(f"\n--- Pair {i+1} ---")
        print(f"SRC: {pair['src']}")
        print(f"TGT: {pair['tgt']}")

    # Analyzing vocabulary
    print("\nVocabulary Analysis:")

    # Collect all words
    src_words = set()
    tgt_words = set()
    src_counter = Counter()
    tgt_counter = Counter()

    for pair in train_pairs:
        # Simple word tokenization (split by space)
        src_tokens = pair["src"].lower().split()
        tgt_tokens = pair["tgt"].lower().split()

        src_words.update(src_tokens)
        tgt_words.update(tgt_tokens)
        src_counter.update(src_tokens)
        tgt_counter.update(tgt_tokens)

    print(f"German vocabulary size: {len(src_words)} unique words")
    print(f"English vocabulary size: {len(tgt_words)} unique words")
    print(f"\nMost common German words: {src_counter.most_common(10)}")
    print(f"Most common English words: {tgt_counter.most_common(10)}")

    return train_pairs, val_pairs


if __name__ == "__main__":
    # Creating training subset
    train_pairs, val_pairs = create_training_subset(num_examples=5000)

    print("\nNext Steps:")
    print("1. Small dataset created and saved")
    print("2. Vocabulary analyzed")
    print("3. Build tokenizer and vocabularies")
    print("4. Convert to tensor format")
