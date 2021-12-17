from functools import partial

from transformers import AutoTokenizer, AutoModelForTokenClassification
from datasets import load_dataset
from tqdm import tqdm
import torch
import typer


def tokenize_and_align_labels(examples, tokenizer):
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

def train(pretrained_model: str = "distilbert-base-uncased", learning_rate:float=1e-5, batch_size: int=16, epochs: int=5):
    dataset = load_dataset("conll2003")
    labels = dataset["train"].features["ner_tags"].feature.names

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    tokenize_and_align_labels_partial = partial(tokenize_and_align_labels, tokenizer=tokenizer)
    dataset = dataset.map(tokenize_and_align_labels_partial, batched=True)
    dataset = dataset.remove_columns(["pos_tags", "id", "chunk_tags", "tokens", "ner_tags"])
    dataset.set_format("torch")

    data = torch.utils.data.DataLoader(dataset["train"], shuffle=True, batch_size=batch_size)

    model = AutoModelForTokenClassification.from_pretrained("distilbert-base-uncased", num_labels=len(labels))

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        batches = tqdm(data, desc=f"Epoch {epoch+1:2d}/{epochs:2d}")
        for batch in batches:
            optimizer.zero_grad()

            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            batches.set_postfix({"loss": loss.item()})

if __name__ == "__main__":
    typer.run(train)
