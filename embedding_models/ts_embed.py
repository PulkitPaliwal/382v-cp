import torch
from torch import nn

class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)
        # x: [Batch Variate Time]
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        # x: [Batch Variate d_model]
        return self.dropout(x)
    
class TimeSeriesEmbedding(nn.Module):
    def __init__(self, window_len, skip_len, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(TimeSeriesEmbedding, self).__init__()
        self.embed_type = embed_type
        self.freq = freq
        self.window_len = window_len
        self.skip_len = skip_len
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        self.embed = DataEmbedding_inverted(c_in, d_model, embed_type, freq, dropout)

    def forward(self, x, x_mark):
        assert x.size(1) >= self.window_len
        returing_list = []
        for i in range(0, x.size(1) - self.window_len + 1, self.skip_len):
            x_win = x[:, i:i+self.window_len, :]
            x_mark_win = x_mark[:, i:i+self.window_len, :]
            x_win = self.embed(x_win, x_mark_win)
            returing_list.append(x_win)
        
        # sum up all the embeddings
        x = torch.stack(returing_list, dim=1)
        # x: [Batch, Num_windows, Variate, d_model]
        x = torch.sum(x, dim=1)
        # x: [Batch, Variate, d_model]
        return x