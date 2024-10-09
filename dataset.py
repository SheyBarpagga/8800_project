import pandas as pd
import torch
from skimage import io
from PIL import Image
from torch.utils.data import Dataset
import torchtext
import torchtext.data
import torchtext.vocab
from torchvision import transforms



class phishingDataset(Dataset):
    def __init__(self, csv_file):
        self.annotations = pd.read_csv(csv_file)
        self.tokenizer = torchtext.data.utils.get_tokenizer('basic_english')
        transcripts = []
        for t in self.annotations.iloc[:, 2]:
            transcripts.append(t)
        self.vocab = torchtext.vocab.build_vocab_from_iterator(self.tokenizer(transcript) for transcript in transcripts)
        self.vocab.append_token('<pad>')


    
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        specto_path = self.annotations.iloc[index, 0]
        mfcc_path = self.annotations.iloc[index, 1]

        specto_img = Image.open(specto_path).convert('RGB')
        mfcc_img = Image.open(mfcc_path).convert('RGB')

        specto_img = transforms.ToTensor()(specto_img)
        mfcc_img = transforms.ToTensor()(mfcc_img)

        max = 20

        transcription = self.annotations.iloc[index, 2]
        tokens = self.tokenizer(transcription)
        numerical_tokens = [self.vocab[token] for token in tokens]

        #normalize lengths
        if len(numerical_tokens) < max:
            numerical_tokens += [self.vocab['<pad>']] * (max - len(numerical_tokens))
        else:
            numerical_tokens = numerical_tokens[:max]

        numerical_tokens = torch.tensor(numerical_tokens)

        y_label = torch.tensor(int(self.annotations.iloc[index, 3]))

        return (specto_img, mfcc_img, numerical_tokens, y_label)
