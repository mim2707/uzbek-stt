from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer

# Load Uzbek NER dataset from Hugging Face
dataset = load_dataset("risqaliyevds/uzbek_ner")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

# Tokenization and label alignment
def tokenize_and_align_labels(batch):
    tokenized_inputs = tokenizer(batch["tokens"], truncation=True, padding=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(batch["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        labels.append([-100 if id is None else label[id] for id in word_ids])
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)

# Load pre-trained BERT model
model = AutoModelForTokenClassification.from_pretrained("bert-base-multilingual-cased", num_labels=len(dataset["train"].features["ner_tags"].feature.names))

# Training arguments
training_args = TrainingArguments(
    output_dir="../models/bert-uzbek-ner",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="../logs",
    learning_rate=2e-5
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer
)

trainer.train()