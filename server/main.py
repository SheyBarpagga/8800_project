import os
import torch
import librosa
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Body
from fastapi.responses import HTMLResponse
from pydub import AudioSegment
from PIL import Image
from torchvision import transforms
from new_model import MultiInputModel
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import matplotlib.pyplot as plt
import sys
import nltk
from nltk.tokenize import word_tokenize
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from fastapi.staticfiles import StaticFiles
from typing import List
import json
import base64
from io import BytesIO
import asyncio
import time
import soundfile as sf
# import ffmpeg

nltk.download('punkt')

# Function to generate spectrogram
def audio_to_spectrogram(audio_path, save_path):
    y, sr = librosa.load(audio_path, sr=None)
    D = np.abs(librosa.stft(y))
    S_dB = librosa.amplitude_to_db(D, ref=np.max)
    plt.figure(figsize=(10, 5))
    librosa.display.specshow(S_dB, sr=sr, x_axis=None, y_axis=None)
    plt.axis("off")
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Spectrogram saved at: {save_path}")

# Function to generate MFCC
def extract_mfcc(audio_path, save_path):
    y, sr = librosa.load(audio_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, sr=sr, x_axis=None, y_axis=None)
    plt.axis("off")
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"MFCC saved at: {save_path}")
    return mfccs

# # Function to generate transcript
# def transcribe_audio(audio_path):
#     result = transcription_model.transcribe(audio_path)
#     transcript = result['text']
#     print(f"Transcript: {transcript}")
#     return transcript
def transcribe_audio(audio_path):
    # Load audio
    y, sr = librosa.load(audio_path, sr=16000)
    
    # Process audio with Wav2Vec2
    input_values = processor(y, return_tensors="pt", sampling_rate=16000, padding=True).input_values
    with torch.no_grad():
        logits = transcription_model(input_values).logits
    
    # Decode the logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcript = processor.batch_decode(predicted_ids)[0]
    
    print(f"Transcript: {transcript} \n\n")
    return transcript


# Function to preprocess the data and make a prediction
def predict(audio_path, model, vocab):
    print("Starting prediction...")
    # Generate spectrogram and MFCC
    spectrogram_path = "temp_spectrogram.png"
    mfcc_path = "temp_mfcc.png"
    audio_to_spectrogram(audio_path, spectrogram_path)
    extract_mfcc(audio_path, mfcc_path)
    transcript = transcribe_audio(audio_path)
    
    # Preprocessing
    print("Preprocessing data...")
    spectrogram = Image.open(spectrogram_path).convert('RGB')
    mfcc = Image.open(mfcc_path).convert('RGB')
    spectrogram = transforms.ToTensor()(spectrogram).unsqueeze(0)
    mfcc = transforms.ToTensor()(mfcc).unsqueeze(0)
    
    tokens = word_tokenize(transcript.lower())
    vocab_set = set(vocab.word2idx.keys())
    
    numerical_tokens = [vocab[token] for token in tokens if token in vocab_set]
    max_len = 20
    if len(numerical_tokens) < max_len:
        numerical_tokens += [vocab[vocab.pad_token]] * (max_len - len(numerical_tokens))
    else:
        numerical_tokens = numerical_tokens[:max_len]
    numerical_tokens = torch.tensor(numerical_tokens).unsqueeze(0)
    
    # Make a prediction
    print("Making prediction...")
    with torch.no_grad():
        output = model(spectrogram, mfcc, numerical_tokens)
        prediction = (output.squeeze() > 0.5).float().item()
    
    print(f"Prediction: {prediction}")
    return prediction

if __name__ == "__main__":

    app = FastAPI()

    # Load the models
    vocab = torch.load("vocab_2.pth", weights_only=False)
    vocab_size = len(vocab)
    model = MultiInputModel(vocab_size=vocab_size)
    model.load_state_dict(torch.load("multi_input_model_2.pth"))
    model.eval()

    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    transcription_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

    os.makedirs('audio_chunks', exist_ok=True)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.mount("/static", StaticFiles(directory="."), name="static")

    @app.get("/")
    async def get():
        # with open('C:\\Users\\sheyb\\Documents\\8800_project\\server\\templates\\index.html', 'r') as file:
        with open('C:\\Users\\sheyb\\OneDrive\\Documents\\8800\\8800_project\\server\\templates\\index.html', 'r') as file:            
            return HTMLResponse(file.read())
        
    @app.get("/call")
    async def get():
        # with open('C:\\Users\\sheyb\\Documents\\8800_project\\server\\templates\\client.html', 'r') as file:
        with open('C:\\Users\\sheyb\\OneDrive\\Documents\\8800\\8800_project\\server\\templates\\client.html', 'r') as file:
            return HTMLResponse(file.read())

    connected_clients = set()

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()
        connected_clients.add(websocket)

        try:
            while True:
                data = await websocket.receive_text()
                message = json.loads(data)

                for client in connected_clients:
                    if client != websocket:
                        await client.send_text(json.dumps(message))

        except WebSocketDisconnect:
            connected_clients.remove(websocket)

    @app.post("/analyze-audio")
    async def analyze_audio(file: UploadFile = File(...)):
        audio_path = f"audio_chunks/{file.filename}"
        with open(audio_path, "wb") as buffer:
            buffer.write(await file.read())
        
        prediction = predict(audio_path, model, vocab)
        os.remove(audio_path)  # Clean up the temporary file
        return {"scam": prediction}

    # @app.websocket("/audio-stream")
    # async def websocket_endpoint(websocket: WebSocket):
    #     await websocket.accept()
    #     audio_data = bytearray()
    #     try:
    #         while True:
    #             data = await websocket.receive_bytes()
    #             audio_data.extend(data)
    #             # Process every 5 seconds of audio
    #             if len(audio_data) > 16000 * 5: 
    #                 audio_path = "audio_chunks/temp_audio.wav"
    #                 with open(audio_path, "wb") as f:
    #                     f.write(audio_data)
    #                 prediction = predict(audio_path, model, vocab)
    #                 await websocket.send_json({"scam": prediction})
    #                 # Reset the buffer
    #                 audio_data = bytearray()  
    #     except WebSocketDisconnect:
    #         print("WebSocket disconnected")


    audio_queue = asyncio.Queue()

    async def process_audio_queue():
        while True:
            await asyncio.sleep(5)
            audio_data = await audio_queue.get()
            
            # Save temp file
            audio_path = "audio_chunks/temp_audio.wav"
            with open(audio_path, "wb") as f:
                f.write(audio_data)

            # def fix_wav_format(input_wav, output_wav):
            #     ffmpeg.input(input_wav).output(output_wav, format="wav", acodec="pcm_s16le", ar=44100).run(overwrite_output=True)

            # fix_wav_format("audio_chunks/temp_audio.wav", "fixed_audio.wav")
            
            # Predict
            prediction = predict(audio_path, model, vocab)
            
            # Send response
            await active_websocket.send_json({"scam": prediction})
            



    # AUDIO_CHUNK_DURATION = 5
    # audio_buffer = []

    # async def process_audio_queue():
    #     global audio_buffer

    #     while True:
    #         if not audio_queue.empty():
    #             audio_chunk = await audio_queue.get()
    #             audio_buffer.append(audio_chunk)

    #         if len(audio_buffer) >= AUDIO_CHUNK_DURATION:
    #             audio_data = np.concatenate(audio_buffer, axis=0)
                
    #             audio_path = "audio_chunks/temp_audio.wav"
    #             sf.write(audio_path, audio_data, samplerate=16000) 

    #             print("Processing 5 seconds of audio...")

    #             # Predict
    #             prediction = predict(audio_path, model, vocab)
    #             print(f"Prediction: {prediction}")

    #             for client in connected_clients:
    #                 await client.send_json({"prediction": prediction})

    #             # Clear the buffer for the next 5 seconds
    #             audio_buffer = []
            
    #         await asyncio.sleep(0.1)


    @app.websocket("/audio-stream")
    async def audio_stream(websocket: WebSocket):
        global active_websocket
        active_websocket = websocket
        await websocket.accept()
        
        asyncio.create_task(process_audio_queue())  # Start processing
        
        try:
            while True:
                data = await websocket.receive_bytes()
                await audio_queue.put(data)
        
        except WebSocketDisconnect:
            print("Audio WebSocket disconnected")



    # @app.websocket("/video-stream")
    # async def video_stream(websocket: WebSocket):
    #     await websocket.accept()
        
    #     try:
    #         while True:
    #             data = await websocket.receive_text()
    #             image_data = json.loads(data)["frame"]
                
    #             # Decode base64 image
    #             image_bytes = base64.b64decode(image_data)
    #             image = Image.open(BytesIO(image_bytes)).convert("RGB")
                
    #             # Preprocess image
    #             transform = transforms.Compose([
    #                 transforms.Resize((224, 224)),
    #                 transforms.ToTensor()
    #             ])
    #             image_tensor = transform(image).unsqueeze(0)
                
    #             # Run through CNN
    #             with torch.no_grad():
    #                 output = model(image_tensor)
    #                 prediction = (output.squeeze() > 0.5).float().item()
                
    #             # Send back result
    #             await websocket.send_json({"scam": prediction})
        
    #     except WebSocketDisconnect:
    #         print("Video WebSocket disconnected")

    uvicorn.run(app, host="0.0.0.0", port=8000)









