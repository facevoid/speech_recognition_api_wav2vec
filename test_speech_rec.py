import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import speech_recognition as sr
import io
from pydub import AudioSegment

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")


def get_audio_transcription(wav_file, sampling_rate = 16000):
  data = io.BytesIO(wav_file)
  clip = AudioSegment.from_file(data)
  x = torch.FloatTensor(clip.get_array_of_samples())
  inputs = processor(x, return_tensors='pt', sampling_rate=16000, padding = 'longest').input_values
  logits = model(inputs).logits
  tokens = torch.argmax(logits, axis = -1)
  text = processor.batch_decode(tokens)
  return 'you said ' +  str(text).lower()


if __name__ == "__main__":
  r = sr.Recognizer()
  
  with sr.Microphone(sample_rate = 16000) as source:
    print('you can start speaking now')
    while True:
        audio = r.listen(source)
        print(type(audio))
        print('you said something')
        data = io.BytesIO(audio.get_wav_data())
        clip = AudioSegment.from_file(data)
        x = torch.FloatTensor(clip.get_array_of_samples())
        inputs = processor(x, return_tensors='pt', sampling_rate=16000, padding = 'longest').input_values
        logits = model(inputs).logits
        tokens = torch.argmax(logits, axis = -1)
        text = processor.batch_decode(tokens)
        print('you said ', str(text).lower())
  
