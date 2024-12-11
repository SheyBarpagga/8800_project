import torch
import torch.nn as nn

# CHECK OUT gTTS FOR DATASET


class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        # CNN layers
        self.conv_block = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),  # Change 1 to 3
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            #get proper output size
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
    def forward(self, x):
        return self.conv_block(x)

class MultiInputModel(nn.Module):
    def __init__(self, vocab_size):
        super(MultiInputModel, self).__init__()

        embedding_dim = 50

        # combine the spectogram and MFCC spectogram
        self.cnn = CustomCNN()

        self.embedding = nn.EmbeddingBag(vocab_size, embedding_dim, sparse=True)
        self.text_fc = nn.Linear(embedding_dim, 512)

        # combine
        self.fc = nn.Sequential(
            #adjust for cnn size
            #TODO check image sizes
            nn.Linear(2 * 256 + 512, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, spectrogram, mfcc, transcript):
        # pass to same cnn
        s_features = self.cnn(spectrogram)
        mfcc_features = self.cnn(mfcc)

        #flatten dimensions
        s_features = torch.flatten(s_features, 1)
        mfcc_features = torch.flatten(mfcc_features, 1)

        embedded = self.embedding(transcript)
        transcript_features = self.text_fc(embedded)  

        combined_features = torch.cat((s_features, mfcc_features, transcript_features), dim=1)

        # fully connected layer
        output = self.fc(combined_features)
        
        return output


# import torch
# from torchviz import make_dot
# import os
# # os.environ["PATH"] += os.pathsep + 'D:/Program Files (x86)/Graphviz2.38/bin/'
# # Assume spectrogram and mfcc have shape (batch_size=1, channels=3, height=64, width=64)
# # Transcript length depends on vocab_size; assume transcript has shape (batch_size, seq_len)
# batch_size = 1
# channels, height, width = 3, 64, 64
# seq_len = 10  # Length of the transcript
# vocab_size = 10000

# # Dummy inputs
# spectrogram = torch.randn(batch_size, channels, height, width)
# mfcc = torch.randn(batch_size, channels, height, width)
# transcript = torch.randint(0, vocab_size, (batch_size, seq_len))

# # Initialize the model
# model = MultiInputModel(vocab_size)

# # Forward pass
# output = model(spectrogram, mfcc, transcript)

# # Visualize
# dot = make_dot(output, params=dict(model.named_parameters()))
# dot.render("multi_input_model_graph", format="png")