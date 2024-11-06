import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot
import torch
from PIL import Image
from torchvision import transforms
from new_model import MultiInputModel
import whisper
import torchtext
import sys

whisper_model = whisper.load_model("base")

def audio_to_spectrogram(audio_path, save_path):
    
    # time series and sampling rate
    y, sr = librosa.load(audio_path, sr=None)  
    
    # short time fourier transform
    D = np.abs(librosa.stft(y))

    S_dB = librosa.amplitude_to_db(D, ref=np.max)

    matplotlib.pyplot.figure(figsize=(10, 5))
    librosa.display.specshow(S_dB, sr=sr, x_axis=None, y_axis=None)  # no axis labels
    matplotlib.pyplot.axis("off")  # Hide axes
    # matplotlib.pyplot.colorbar(format="%+2.0f dB")
    # matplotlib.pyplot.title("Spectrogram") 
    
    matplotlib.pyplot.savefig(save_path, bbox_inches='tight', pad_inches=0)
    matplotlib.pyplot.close()  

    print(f"Spectrogram saved at: {save_path}")

def extract_mfcc(audio_path, save_path):

    y, sr = librosa.load(audio_path, sr=None)

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    matplotlib.pyplot.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, sr=sr, x_axis=None, y_axis=None)
    matplotlib.pyplot.axis("off")

    matplotlib.pyplot.savefig(save_path, bbox_inches='tight', pad_inches=0)
    matplotlib.pyplot.close()  

    print(f"transcript saved at: {save_path}")
    
    return mfccs


# Generate transcript
def transcribe_audio(audio_path):
    result = whisper_model.transcribe(audio_path)
    transcript = result['text']
    print(f"Transcript: {transcript}")
    return transcript

# Make the prediction
def predict(audio_path, model_path, vocab):

    #Generate data
    spectrogram_path = "temp_spectrogram.png"
    mfcc_path = "temp_mfcc.png"
    audio_to_spectrogram(audio_path, spectrogram_path)
    extract_mfcc(audio_path, mfcc_path)
    transcript = transcribe_audio(audio_path)
    
    # Load the model
    vocab_size = len(vocab)
    model = MultiInputModel(vocab_size=vocab_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Preprocessing
    spectrogram = Image.open(spectrogram_path).convert('RGB')
    mfcc = Image.open(mfcc_path).convert('RGB')
    spectrogram = transforms.ToTensor()(spectrogram).unsqueeze(0)
    mfcc = transforms.ToTensor()(mfcc).unsqueeze(0)
    
    tokenizer = torchtext.data.utils.get_tokenizer('basic_english')
    tokens = tokenizer(transcript)
    numerical_tokens = [vocab[token] for token in tokens if token in vocab]
    max_len = 20
    if len(numerical_tokens) < max_len:
        numerical_tokens += [vocab['<pad>']] * (max_len - len(numerical_tokens))
    else:
        numerical_tokens = numerical_tokens[:max_len]
    numerical_tokens = torch.tensor(numerical_tokens).unsqueeze(0)
    
    # Make a prediction
    with torch.no_grad():
        output = model(spectrogram, mfcc, numerical_tokens)
        prediction = (output.squeeze() > 0.5).float().item()
    
    print(f"Prediction: {prediction}")
    return prediction


def main(args):
    model_path = "multi_input_model.pth"
    vocab = torch.load("vocab.pth")
    predict(args, model_path, vocab)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py <path to testing audio>")
        sys.exit(1)
    main(sys.argv[1])