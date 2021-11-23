import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
      
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        self.batch_norm = nn.BatchNorm1d(embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        features = self.batch_norm(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn  = nn.LSTM(embed_size, hidden_size, num_layers, batch_first = True)
        self.linear = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):
        embeddings = self.embedding(captions[:,:-1])
#         print(embeddings.shape)
        inputs = torch.cat((features.unsqueeze(1), embeddings), 1)
#         print(features.unsqueeze(1).shape)
#         print(inputs.shape)
        h,c = self.rnn(inputs)
        res = self.linear(h)
        return res
    


    def sample(self, inputs, states=None, max_len=20):
        res_list = []
        step_in = inputs
        print(step_in.shape)
        for i in range(0,max_len):
            h, c = self.rnn(step_in)
            res = self.linear(h)
            out, inds = torch.max(res,dim=2)
            res_list.append(inds.item())
            step_in = self.embedding(inds)
        return res_list