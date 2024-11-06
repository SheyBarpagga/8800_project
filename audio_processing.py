import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot
import os

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

# def batch_audio(input_folder):
    
#     os.makedirs("spectogram", exist_ok=True)
#     os.makedirs("mfcc", exist_ok=True)

#     #convert and save each file in new folders for later 
#     for filename in os.listdir(input_folder):
#         if filename.endswith('.mp3'): 
#             audio_path = os.path.join(input_folder, filename)
            
#             spectrogram_save_path = os.path.join("spectogram", f"{os.path.splitext(filename)[0]}_spectrogram.png")
#             mfcc_save_path = os.path.join("mfcc", f"{os.path.splitext(filename)[0]}_mfcc.npy")
            
#             audio_to_spectrogram(audio_path, spectrogram_save_path)
            
#             mfccs = extract_mfcc(audio_path)
#             np.save(mfcc_save_path, mfccs)


def batch_audio(input_folder, flag):

    specto_folder = "spectogram"
    mfcc_folder = "mfcc"
    
    if flag == 1:
        specto_folder = "phishing_spectogram"
        mfcc_folder = "phishing_mfcc"
        os.makedirs(specto_folder, exist_ok=True)
        os.makedirs(mfcc_folder, exist_ok=True)
    else:
        os.makedirs(specto_folder, exist_ok=True)
        os.makedirs(mfcc_folder, exist_ok=True)

    #convert and save each file in new folders for later 
    for filename in os.listdir(input_folder):
        if filename.endswith('.wav'): 
            audio_path = os.path.join(input_folder, filename)
            
            spectrogram_save_path = os.path.join(specto_folder, f"{os.path.splitext(filename)[0]}_spectrogram.png")
            mfcc_save_path = os.path.join(mfcc_folder, f"{os.path.splitext(filename)[0]}_mfcc.png")
            
            audio_to_spectrogram(audio_path, spectrogram_save_path)
            
            extract_mfcc(audio_path, mfcc_save_path)


def change_name():
    count = 41
    for filename in os.listdir("./calls"):
        os.rename(f"C:/Users/sheyb/Documents/8800_project/calls/{filename}", f"{count}.mp3")
        count +=1


batch_audio("./new/", 0)


