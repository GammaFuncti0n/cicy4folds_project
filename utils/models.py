import torch.nn as nn

'''
File with models:
CNN(), MLP()
'''

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(),   

            nn.Conv2d(32, 64, 3, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(),         
            
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*4, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):   
        x = x.unsqueeze(1)
        x = self.features(x)
        #print('x shape:',x.shape)
        return self.fc(x)
    

class MLP(nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()
        self.forw = nn.Sequential(
            nn.Flatten(),
            nn.Linear(config['MLP']['input_size'], config['MLP']['hidden_size_1']),
            nn.ReLU(),
            nn.Dropout(p=config['MLP']['dropout']),

            nn.Linear(config['MLP']['hidden_size_1'], config['MLP']['hidden_size_2']),
            nn.ReLU(),
            nn.Dropout(p=config['MLP']['dropout']),

            nn.Linear(config['MLP']['hidden_size_2'], config['MLP']['output_size']),
            nn.ReLU(),
            nn.Dropout(p=config['MLP']['dropout'])
        )
    
    def forward(self, x):
        return self.forw(x)


class DeepCNN(nn.Module):
    def __init__(self):
        super(DeepCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(),   

            nn.Conv2d(32, 64, 3, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(),   

            nn.Conv2d(128, 256, 3, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(),

            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(512, 1024, 3, padding=1),
            nn.ReLU(),
            
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):   
        x = x.unsqueeze(1)
        x = self.features(x)
        #print('x shape:',x.shape)
        return self.fc(x)