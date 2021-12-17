from transformers import AutoTokenizer, AutoModelForTokenClassification
from datasets import load_dataset
from tqdm import tqdm
import torch

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
dataset = load_dataset("conll2003")
labels = dataset["train"].features["ner_tags"].feature.names
print(labels)

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, padding="max_length", is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples[f"ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:                            # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:              # Only label the first token of a given word.
                label_ids.append(label[word_idx])

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

dataset = dataset.map(tokenize_and_align_labels, batched=True)
dataset = dataset.remove_columns(["pos_tags", "id", "chunk_tags", "tokens", "ner_tags"])
dataset.set_format("torch")

data = torch.utils.data.DataLoader(dataset["train"], shuffle=True, batch_size=16)

model = AutoModelForTokenClassification.from_pretrained("distilbert-base-uncased", num_labels=len(labels))

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

for epoch in range(5):
    for batch in tqdm(data):
        optimizer.zero_grad()

        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
