from fastapi import FastAPI, UploadFile, File
from test_speech_rec import get_audio_transcription

app = FastAPI()


@app.post("/file")
async def upload_file(file: bytes = File(...), sampling_rate=16000):
    # Do here your stuff with the file
    #print('File recieved ', file)
    # print(dir(file))
    transcribed_text = get_audio_transcription(file, sampling_rate = sampling_rate)
    return {"filename": transcribed_text}

