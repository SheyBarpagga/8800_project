import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import torch
import cv2
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# Assuming MultiInputModel is defined in new_model.py
from new_model import MultiInputModel

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate_cam(self, input_tensor, target_class):
        self.model.eval()
        output = self.model(input_tensor)
        self.model.zero_grad()
        target = output[0][target_class]
        target.backward()

        gradients = self.gradients[0].cpu().data.numpy()
        activations = self.activations[0].cpu().data.numpy()

        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (input_tensor.shape[2], input_tensor.shape[3]))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam

def visualize_cam(cam, image, alpha=0.5):
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam_image = heatmap + np.float32(image)
    cam_image = cam_image / np.max(cam_image)
    plt.imshow(cam_image)
    plt.show()

def audio_to_mfcc(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return mfcc

def transcribe_audio(audio_path, processor, model):
    y, sr = librosa.load(audio_path, sr=16000)
    input_values = processor(y, return_tensors="pt", sampling_rate=sr).input_values
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    return transcription

def main(audio_path, target_class):
    # Extract MFCC features
    mfcc = audio_to_mfcc(audio_path)
    mfcc_tensor = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, n_mfcc, time_steps)

    # Generate transcript
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    transcript = transcribe_audio(audio_path, processor, model)

    # Convert transcript to tensor (example, adjust as needed)
    transcript_tensor = torch.tensor([ord(c) for c in transcript], dtype=torch.float32).unsqueeze(0)  # Shape: (1, length_of_transcript)

    # Initialize Grad-CAM
    target_layer = model.wav2vec2.encoder.layers[-1]  # Example target layer, adjust as needed
    grad_cam = GradCAM(model, target_layer)

    # Generate and visualize CAM
    cam = grad_cam.generate_cam(mfcc_tensor, target_class)
    image = mfcc_tensor[0][0].cpu().data.numpy()  # Convert tensor to image format
    visualize_cam(cam, image)

if __name__ == "__main__":
    audio_path = "C:\\Users\\sheyb\\Documents\\8800_project\\non_phishing\\2.mp3"
    target_class = 0 
    main(audio_path, target_class)