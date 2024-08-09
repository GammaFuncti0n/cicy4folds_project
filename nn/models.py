import torch
import torch.nn as nn

'''
File with models:
CNN(), MLP(), DeepCNN(), VisionTransformer()
'''

class CNN(nn.Module):
    def __init__(self, model_params):
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
    def __init__(self, model_params):
        super(MLP, self).__init__()
        self.forw = nn.Sequential(
            nn.Flatten(),
            nn.Linear(model_params['input_size'], model_params['hidden_size_1']),
            nn.ReLU(),
            nn.Dropout(p=model_params['dropout']),

            nn.Linear(model_params['hidden_size_1'], model_params['hidden_size_2']),
            nn.ReLU(),
            nn.Dropout(p=model_params['dropout']),

            nn.Linear(model_params['hidden_size_2'], model_params['output_size']),
            nn.ReLU(),
            nn.Dropout(p=model_params['dropout'])
        )
    
    def forward(self, x):
        return self.forw(x)


class DeepCNN(nn.Module):
    def __init__(self, model_params):
        super(DeepCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),   

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),   

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, 3, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(),

            nn.Conv2d(512, 1024, 3, padding=1),
            nn.MaxPool2d(2),
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


class VisionTransformer(nn.Module):
    def __init__(self, model_params):  #num_patches, patch_size, embed_dim, pool_size, num_layers, num_heads, mlp_dim, out_dim
        super(VisionTransformer, self).__init__()

        self.initializer_range = model_params.std # std for weight initializer
        
        self.patch_embed = nn.Conv2d(1, model_params.embed_dim, kernel_size=model_params.patch_size, stride=model_params.patch_size) # Convert matrix to patches
        self.positional_embeddings = nn.Parameter(torch.rand(model_params.num_patches, model_params.embed_dim)) #nn.Embedding(model_params.num_patches, model_params.embed_dim)
        self.pooling = nn.AvgPool2d(kernel_size=model_params.pool_size)
        self.positions = torch.arange(0, model_params.num_patches, dtype=torch.long).to()
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_params.embed_dim, nhead=model_params.num_heads, dim_feedforward=model_params.mlp_dim, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=model_params.num_layers)
        
        # MLP head
        input_size = (model_params.num_patches//model_params.pool_size)*(model_params.embed_dim//model_params.pool_size)

        self.mlp_head = nn.Sequential(
            nn.Linear(input_size, model_params.mlp_dim),
            nn.ReLU(),
            nn.Linear(model_params.mlp_dim, model_params.out_dim)
        )
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights.

        Examples:
        https://github.com/huggingface/transformers/blob/v4.25.1/src/transformers/models/gpt2/modeling_gpt2.py#L454
        https://recbole.io/docs/_modules/recbole/model/sequential_recommender/sasrec.html#SASRec
        """

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(self, x):
        if(x.dim()==2): # if only one matrix
            x = x.unsqueeze(dim=0)
            x = x.unsqueeze(dim=0)
        if(x.dim()==3): # if batch of matrices
            x = x.unsqueeze(dim=1)
            
        # Matrix to patch
        x = self.patch_embed(x)  # (B, embed_dim, H, W)
        x = x.flatten(2)  # (B, embed_dim, N_patches)
        x = x.transpose(1, 2)  # (B, N_patches, embed_dim)
        
        # Positional embeddings
        x = x + self.positional_embeddings

        # Transformer
        x = self.transformer_encoder(x)
        x = self.pooling(x)
        x = x.flatten(1)
        
        # MLP head
        out = self.mlp_head(x)  # (B, out_dim)
        
        return out