import torch
import torch.nn as nn
from torchvision import models
import torchtext; torchtext.disable_torchtext_deprecation_warning()
#mostly tutorial stuff
#TODO bigger training set
#TODO finetuning

class MultiInputModel(nn.Module):
    def __init__(self, vocab_size):
        super(MultiInputModel, self).__init__()

        embedding_dim = 50

        self.resnet1 = models.resnet50(weights="ResNet50_Weights.IMAGENET1K_V1")
        self.resnet2 = models.resnet50(weights="ResNet50_Weights.IMAGENET1K_V1")

        #remove the final layer for custom layer (combined layer)
        self.resnet1 = nn.Sequential(*list(self.resnet1.children())[:-1])
        self.resnet2 = nn.Sequential(*list(self.resnet2.children())[:-1])

        self.embedding = nn.EmbeddingBag(vocab_size, embedding_dim)
        #map embedding vector into 512 size
        self.text_fc = nn.Linear(embedding_dim, 512)

        #combine the layers
        self.fc = nn.Sequential(
            #2 images layer
            nn.Linear(2 * 2048 + 512, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            #get binary class
            nn.Sigmoid()
        )

    def forward(self, spectrogram, mfcc, transcript):

        s_features = self.resnet1(spectrogram)
        mfcc_features = self.resnet2(mfcc)

        embedded = self.embedding(transcript)
        transcript_features = self.text_fc(embedded)  

        combined_features = torch.cat((s_features, mfcc_features, transcript_features), dim=1)

        #pass through the fully connected layer
        output = self.fc(combined_features)
        
        return output
