import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

path = './phishing/1.mp3'
y, sr = librosa.load(path, sr=None) 

mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

plt.figure(figsize=(10, 4))
librosa.display.specshow(mel_spectrogram_db, sr=sr, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel Spectrogram')
plt.tight_layout()
plt.show()

# changes in audio over time
# highlight speech rhythms and pauses
zero_crossings = librosa.feature.zero_crossing_rate(y)

#difference between peaks and valleys in the sound spectrum
#tonal shifts
spectral_contrast = librosa.feature.spectral_contrast(y=librosa.effects.harmonic(y), sr=sr)


print("zero crossing: ")
print(zero_crossings)
print("spectral_contrast: ")
print(spectral_contrast)