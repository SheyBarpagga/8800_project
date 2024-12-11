import os
import torch
import librosa
import numpy as np
from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydub import AudioSegment
from PIL import Image
from torchvision import transforms
from torchtext.data.utils import get_tokenizer
from new_model import MultiInputModel
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# Set up FastAPI
app = FastAPI()

# Load the models
vocab = torch.load("vocab.pth")
vocab_size = len(vocab)
model = MultiInputModel(vocab_size=vocab_size)
model.load_state_dict(torch.load("multi_input_model.pth"))
model.eval()

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
transcription_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# Create directories if needed
os.makedirs('audio_chunks', exist_ok=True)

# Load vocab
vocab = torch.load("vocab.pth")

@app.get("/")
async def get():
    """Serve the HTML page."""
    with open('templates/index.html', 'r') as file:
        return HTMLResponse(file.read())

@app.websocket("/audio-stream")
async def websocket_endpoint(websocket: WebSocket):
    """Receive real-time audio from the user via WebSocket and make predictions."""
    await websocket.accept()
    audio_buffer = AudioSegment.empty()  # Store audio chunks in memory
    try:
        while True:
            data = await websocket.receive_bytes()
            new_audio = AudioSegment(data)
            audio_buffer += new_audio  # Concatenate incoming audio chunks
            
            if len(audio_buffer) > 5000:  # Every 5 seconds (adjust as needed)
                audio_file_path = 'audio_chunks/temp_audio.wav'
                audio_buffer.export(audio_file_path, format='wav')  # Save the chunk as a wav file
                audio_buffer = AudioSegment.empty()  # Reset the buffer
                
                # Use your existing prediction logic
                prediction = predict(audio_file_path)
                await websocket.send_json({"scam": prediction})
    except Exception as e:
        print(f"Connection closed: {e}")
    finally:
        await websocket.close()

def predict(audio_path):
    """Runs the entire pipeline for a given audio file and returns the scam prediction."""
    print(f"Predicting on {audio_path}")
    
    # Generate spectrogram and MFCCs
    spectrogram_path = "audio_chunks/temp_spectrogram.png"
    mfcc_path = "audio_chunks/temp_mfcc.png"
    
    audio_to_spectrogram(audio_path, spectrogram_path)
    extract_mfcc(audio_path, mfcc_path)
    transcript = transcribe_audio(audio_path)
    
    # Load and prepare inputs
    spectrogram = Image.open(spectrogram_path).convert('RGB')
    mfcc = Image.open(mfcc_path).convert('RGB')
    spectrogram = transforms.ToTensor()(spectrogram).unsqueeze(0)
    mfcc = transforms.ToTensor()(mfcc).unsqueeze(0)
    
    tokenizer = get_tokenizer('basic_english')
    tokens = tokenizer(transcript)
    numerical_tokens = [vocab.get(token, vocab['<unk>']) for token in tokens]
    
    max_len = 20  # Pad or truncate to max length
    if len(numerical_tokens) < max_len:
        numerical_tokens += [vocab['<pad>']] * (max_len - len(numerical_tokens))
    else:
        numerical_tokens = numerical_tokens[:max_len]
    numerical_tokens = torch.tensor(numerical_tokens).unsqueeze(0)
    
    with torch.no_grad():
        output = model(spectrogram, mfcc, numerical_tokens)
        prediction = (output.squeeze() > 0.5).float().item()
    
    print(f"Prediction: {prediction}")
    return prediction


def audio_to_spectrogram(audio_path, save_path):
    y, sr = librosa.load(audio_path, sr=None)  
    D = np.abs(librosa.stft(y))
    S_dB = librosa.amplitude_to_db(D, ref=np.max)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    librosa.display.specshow(S_dB, sr=sr, x_axis=None, y_axis=None)
    plt.axis("off")
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def extract_mfcc(audio_path, save_path):
    y, sr = librosa.load(audio_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, sr=sr, x_axis=None, y_axis=None)
    plt.axis("off")
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def transcribe_audio(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)
    input_values = processor(y, return_tensors="pt", sampling_rate=16000, padding=True).input_values
    with torch.no_grad():
        logits = transcription_model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcript = processor.batch_decode(predicted_ids)[0]
    return transcript
