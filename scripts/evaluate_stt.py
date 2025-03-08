from transformers import WhisperForConditionalGeneration, WhisperProcessor
from jiwer import wer
import whisper

model = WhisperForConditionalGeneration.from_pretrained("../models/whisper-uzbek")
processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="uz", task="transcribe")

def compute_wer(audio_path, reference_text):
    audio_input = whisper.load_audio(audio_path)
    input_features = whisper.log_mel_spectrogram(audio_input)

    pred_ids = model.generate(input_features)
    pred_text = processor.batch_decode(pred_ids, skip_special_tokens=True)[0]

    return wer([reference_text], [pred_text])

print("WER:", compute_wer("../inference/sample_uzbek_audio.wav", "Bu misol matni."))