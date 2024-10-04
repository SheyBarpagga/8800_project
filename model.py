import torch
import torch.nn as nn
from torchvision import models

#mostly tutorial stuff
#TODO bigger training set
#TODO finetuning

class MultiInputModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(MultiInputModel, self).__init__()
        
        # spectrogram branch
        self.resnet = models.resnet50(weights='IMAGENET1K_V1')  # pretrained model for demo purposes
        self.resnet.fc = nn.Identity() 
        
        # mfcc branch
        # TODO better understand what changing these values mean
        self.mfcc_fc = nn.Sequential(
            nn.Linear(40, 256), 
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
        #transvript branch
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, 128, batch_first=True)

        # combine
        self.fc = nn.Sequential(
            nn.Linear(512 + 128, 256),  # suggested sizes 
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, spectrogram, mfcc, transcript):

        s_features = self.resnet(spectrogram)
        mfcc_features = self.mfcc_fc(mfcc)

        embedded = self.embedding(transcript)
        lstm_out, (hn, cn) = self.lstm(embedded)
        transcript_features = hn[-1]  

        combined_features = torch.cat((s_features, mfcc_features, transcript_features), dim=1)
        output = self.fc(combined_features)
        
        return output
