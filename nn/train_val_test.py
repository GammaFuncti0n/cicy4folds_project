from tqdm import tqdm
import torch
from time import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score

'''
File with train, test functions
'''

def train_epoch(model, optimizer, criterion, train_dataloader, validate_dataloader, config):

    list_loss_train = []
    list_loss_val = []

    list_accuracy_train = []
    list_accuracy_val = []

    model.train()
    for batch in train_dataloader:
        optimizer.zero_grad()
        outputs_train = model(batch['matrix'].to(config['model']['device']))

        loss_train = criterion(outputs_train, batch['hodge_number'].to(config['model']['device']).unsqueeze(1))
        loss_train.backward()
        optimizer.step()

        list_loss_train.append(loss_train.cpu().item())
        list_accuracy_train.append(accuracy_score(batch['hodge_number'], outputs_train.round().detach().cpu()))
    
    model.eval()
    for batch in validate_dataloader:
        outputs_val = model(batch['matrix'].to(config['model']['device']))
        loss_val = criterion(outputs_val, batch['hodge_number'].to(config['model']['device']).unsqueeze(1))
        list_loss_val.append(loss_val.cpu().item())
        list_accuracy_val.append(accuracy_score(batch['hodge_number'], outputs_val.round().detach().cpu()))
    
    torch.cuda.empty_cache()
    return {'train_loss': np.mean(list_loss_train), 
            'validation_loss': np.mean(list_loss_val), 
            'train_accuracy': np.mean(list_accuracy_train),
            'validation_accuracy': np.mean(list_accuracy_val)}

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
    
    return {'loss': np.mean(list_loss_test), 
            'rmse': root_mean_squared_error(targets, predictions),
            'mae': mean_absolute_error(targets, predictions),
            'mape': mean_absolute_percentage_error(targets, predictions),
            'accuracy': accuracy_score(targets, predictions.round()),
            'balanced_accuracy': balanced_accuracy_score(targets, predictions.round()),
            'f1': f1_score(targets, predictions.round(), average='weighted'),
            'precision': precision_score(targets, predictions.round(), average='weighted'),
            'recall': recall_score(targets, predictions.round(), average='weighted')
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