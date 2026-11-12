
from datasets import load_dataset
from transformers import AutoTokenizer

print("Loading CNN/DailyMail dataset...")

dataset = load_dataset(
    "abisee/cnn_dailymail",
    "3.0.0",
    streaming=True
)

# Display available dataset splits
print("\nAvailable dataset splits:")
print(dataset)

# Retrieve one training example
example = next(iter(dataset["train"]))

# Extract article and summary
article = example["article"]
highlights = example["highlights"]

print("\n" + "=" * 60)
print("ORIGINAL ARTICLE")
print("=" * 60)
print(article)

print("\n" + "=" * 60)
print("HUMAN-WRITTEN SUMMARY")
print("=" * 60)
print(highlights)


print("\nLoading T5 tokenizer...")

tokenizer = AutoTokenizer.from_pretrained("t5-small")


input_text = "summarize: " + article

inputs = tokenizer(
    input_text,
    max_length=512,
    truncation=True,
    padding="max_length",
    return_tensors="pt"
)

print("\n" + "=" * 60)
print("ARTICLE TOKENIZATION")
print("=" * 60)

print("\nInput IDs:")
print(inputs["input_ids"])

print("\nAttention Mask:")
print(inputs["attention_mask"])

print("\nTokenized words:")
print(tokenizer.convert_ids_to_tokens(
    inputs["input_ids"][0]
))

# Decode token IDs back into text
decoded_article = tokenizer.decode(
    inputs["input_ids"][0],
    skip_special_tokens=True
)

print("\nDecoded article:")
print(decoded_article)


labels = tokenizer(
    text_target=highlights,
    max_length=150,
    truncation=True,
    padding="max_length",
    return_tensors="pt"
)

print("\n" + "=" * 60)
print("SUMMARY TOKENIZATION")
print("=" * 60)

print("\nSummary Input IDs:")
print(labels["input_ids"])

print("\nDecoded summary:")
print(tokenizer.decode(
    labels["input_ids"][0],
    skip_special_tokens=True
))


print("\n" + "=" * 60)
print("TOKENIZATION SUMMARY")
print("=" * 60)

print("Article input shape:", inputs["input_ids"].shape)
print("Article attention mask shape:", inputs["attention_mask"].shape)
print("Summary label shape:", labels["input_ids"].shape)

print("\nDataset loading and tokenization completed!")
