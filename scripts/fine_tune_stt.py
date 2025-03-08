from datasets import load_dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor, TrainingArguments, Trainer

# Load pre-trained Whisper processor and model
processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="uz", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

# Load the Uzbek dataset from Common Voice
dataset = load_dataset("mozilla-foundation/common_voice_13_0", "uz", trust_remote_code=True)

# Preprocess audio
def preprocess_audio(batch):
    batch["input_features"] = processor(batch["audio"]["array"], sampling_rate=16000).input_features[0]
    batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
    return batch

dataset = dataset.map(preprocess_audio, remove_columns=["audio"])

# Training arguments
training_args = TrainingArguments(
    output_dir="../models/whisper-uzbek",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-5,
    logging_dir="../logs",
    fp16=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=processor
)

trainer.train()