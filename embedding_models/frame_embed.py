import torch
from torch import nn
# import resnet
from torchvision.models import resnet18

class FrameEmbedding(nn.Module):
    def __init__(self, hidden_dim = 128, output_dim=64):
        super(FrameEmbedding, self).__init__()
        self.resnet = resnet18(pretrained=True)
        # remove the last layer
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        # add 3 linear linear layers,
        self.fc1 = nn.Linear(512, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.resnet(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        return x

class video_embedding(nn.Module):
    def __init__(self, hidden_dim = 128, output_dim=64):
        super(video_embedding, self).__init__()
        self.frame_embedding = FrameEmbedding(hidden_dim, output_dim)
        self.fc = nn.Linear(output_dim, output_dim)

    def forward(self, x):
        raise NotImplementedError
        for frames in x:
            frame_embedding = self.frame_embedding(frames)
            frame_embedding = self.fc(frame_embedding)
            frame_embedding = torch.relu(frame_embedding)
            if 'video_embedding' in locals():
                video_embedding = torch.cat((video_embedding, frame_embedding), 0)
            else:
                video_embedding = frame_embedding
    
if __name__ == '__main__':
    model = FrameEmbedding()
    print(model)
    x = torch.randn(2, 3, 224, 224)
    print(model(x).shape)