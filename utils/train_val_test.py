from tqdm import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score

'''
File with train, test functions
'''

def train_epoch(model, optimizer, criterion, train_dataloader, validate_dataloader, config):

    list_loss_train = []
    list_loss_val = []

    model.train()
    for batch in train_dataloader:
        optimizer.zero_grad()
        outputs_train = model(batch['matrix'].to(config['model']['device']))
    
        loss_train = criterion(outputs_train, batch['hodge_number'].to(config['model']['device']).unsqueeze(1))
        list_loss_train.append(loss_train.to('cpu').item())
        loss_train.backward()
        optimizer.step()
    
    model.eval()
    for batch in validate_dataloader:
        outputs_val = model(batch['matrix'].to(config['model']['device']))
        loss_val = criterion(outputs_val, batch['hodge_number'].to(config['model']['device']).unsqueeze(1))
        list_loss_val.append(loss_val.to('cpu').item())
    
    torch.cuda.empty_cache()
    return {'train_loss': np.mean(list_loss_train), 'validation_loss': np.mean(list_loss_val)}


def test(model, criterion, test_dataloader, config):
    list_loss_test = []
    predictions = np.array([])
    targets = np.array([])

    model.eval()
    for batch in tqdm(test_dataloader):
        outputs_test = model(batch['matrix'].to(config['model']['device']))
        ground_truth = batch['hodge_number'].to(config['model']['device'])
        loss_test = criterion(outputs_test, ground_truth.unsqueeze(1))
        list_loss_test.append(loss_test.to('cpu').item())

        predictions = np.concatenate((predictions, outputs_test.cpu().detach().squeeze(1).numpy()))
        targets = np.concatenate((targets, batch['hodge_number'].detach().numpy()))
    
    return {'test_loss': np.mean(list_loss_test), 
            'test_mse': mean_squared_error(targets, predictions),
            'test_accuracy': accuracy_score(targets, predictions.round()),
            'test_balanced_accuracy': balanced_accuracy_score(targets, predictions.round()),
            'test_f1': f1_score(targets, predictions.round(), average='weighted'),
            'test_precision': precision_score(targets, predictions.round(), average='weighted'),
            'test_recall': recall_score(targets, predictions.round(), average='weighted')
            }


def display_losses(train_loss, val_loss, config):
    plt.plot(np.arange(2,len(train_loss)+1), train_loss[1:], label='train loss')
    plt.plot(np.arange(2,len(val_loss)+1), val_loss[1:], label='validation loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title(f"Loss for {config['dataset']['target_name']}")
    plt.legend()
    plt.grid('on')
    plt.savefig(f"results/loss_{config['model']['name']}_for_{config['dataset']['target_name']}.png")