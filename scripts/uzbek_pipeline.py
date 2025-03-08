import whisper
from transformers import pipeline, WhisperForConditionalGeneration, WhisperProcessor

def uzbek_stt_ner_pipeline(audio_path):
    # Load STT model
    stt_model = WhisperForConditionalGeneration.from_pretrained("../models/whisper-uzbek")
    stt_processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="uz", task="transcribe")

    # Process audio
    audio_input = whisper.load_audio(audio_path)
    input_features = whisper.log_mel_spectrogram(audio_input)

    # Transcribe
    pred_ids = stt_model.generate(input_features)
    transcript = stt_processor.batch_decode(pred_ids, skip_special_tokens=True)[0]

    # Load NER model
    ner_pipeline = pipeline("ner", model="../models/bert-uzbek-ner", tokenizer="bert-base-multilingual-cased")

    # Extract entities
    ner_results = ner_pipeline(transcript)

    return {"transcript": transcript, "entities": ner_results}

# Example usage
audio_path = "../inference/sample_uzbek_audio.wav"
result = uzbek_stt_ner_pipeline(audio_path)
print(result)