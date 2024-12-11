# import pandas as pd
# import torch
# from skimage import io
# from PIL import Image
# from torch.utils.data import Dataset
# import torchtext
# import torchtext.data
# import torchtext.vocab
# from torchvision import transforms



# class phishingDataset(Dataset):
#     def __init__(self, csv_file):
#         self.annotations = pd.read_csv(csv_file)
#         self.tokenizer = torchtext.data.utils.get_tokenizer('basic_english')
#         # transcripts = []
#         # for t in self.annotations.iloc[:, 2]:
#         #     transcripts.append(t)

#         #Replace NaN or non string entries
#         transcripts = [str(t) if isinstance(t, str) else "" for t in self.annotations.iloc[:, 2]]

#         self.vocab = torchtext.vocab.build_vocab_from_iterator(self.tokenizer(transcript) for transcript in transcripts)
#         self.vocab.append_token('<pad>')


    
#     def __len__(self):
#         return len(self.annotations)

#     def __getitem__(self, index):
#         specto_path = self.annotations.iloc[index, 0]
#         mfcc_path = self.annotations.iloc[index, 1]

#         specto_img = Image.open(specto_path).convert('RGB')
#         mfcc_img = Image.open(mfcc_path).convert('RGB')

#         specto_img = transforms.ToTensor()(specto_img)
#         mfcc_img = transforms.ToTensor()(mfcc_img)

#         max = 20

#         transcription = str(self.annotations.iloc[index, 2])
#         tokens = self.tokenizer(transcription)
#         tokens = [token for token in tokens if token != 'nan']
#         numerical_tokens = [self.vocab[token] for token in tokens]

#         #normalize lengths
#         if len(numerical_tokens) < max:
#             numerical_tokens += [self.vocab['<pad>']] * (max - len(numerical_tokens))
#         else:
#             numerical_tokens = numerical_tokens[:max]

#         numerical_tokens = torch.tensor(numerical_tokens)

#         y_label = torch.tensor(int(self.annotations.iloc[index, 3]))

#         return (specto_img, mfcc_img, numerical_tokens, y_label)




import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
import torch

nltk.download('punkt')

def nltk_tokenizer(text):
    return word_tokenize(text.lower())

class Vocabulary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.pad_token = '<pad>'
        self.add_word(self.pad_token)

    def add_word(self, word):
        if word not in self.word2idx:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

    def build_vocab(self, sentences):
        counter = Counter()
        for sentence in sentences:
            counter.update(nltk_tokenizer(sentence))
        for word, _ in counter.items():
            self.add_word(word)

    def __len__(self):
        return len(self.word2idx)

    def __getitem__(self, word):
        return self.word2idx.get(word, self.word2idx[self.pad_token])

class phishingDataset(Dataset):
    def __init__(self, csv_file):
        self.annotations = pd.read_csv(csv_file)
        
        # Replace NaN or non-string entries
        transcripts = [str(t) if isinstance(t, str) else "" for t in self.annotations.iloc[:, 2]]
        
        # Build vocabulary
        self.vocab = Vocabulary()
        self.vocab.build_vocab(transcripts)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        specto_path = self.annotations.iloc[index, 0]
        mfcc_path = self.annotations.iloc[index, 1]

        specto_img = Image.open(specto_path).convert('RGB')
        mfcc_img = Image.open(mfcc_path).convert('RGB')

        specto_img = transforms.ToTensor()(specto_img)
        mfcc_img = transforms.ToTensor()(mfcc_img)

        max_len = 20

        transcription = str(self.annotations.iloc[index, 2])
        tokens = nltk_tokenizer(transcription)
        numerical_tokens = [self.vocab[token] for token in tokens]

        # Normalize lengths
        if len(numerical_tokens) < max_len:
            numerical_tokens += [self.vocab[self.vocab.pad_token]] * (max_len - len(numerical_tokens))
        else:
            numerical_tokens = numerical_tokens[:max_len]
        
        numerical_tokens = torch.tensor(numerical_tokens)

        label = self.annotations.iloc[index, 3]

        return specto_img, mfcc_img, numerical_tokens, torch.tensor(label)