from tqdm import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt

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
    model.eval()
    correct = total = 0
    for batch in tqdm(test_dataloader):
        outputs_test = model(batch['matrix'].to(config['model']['device']))
        ground_truth = batch['hodge_number'].to(config['model']['device'])
        loss_test = criterion(outputs_test, ground_truth.unsqueeze(1))
        list_loss_test.append(loss_test.to('cpu').item())

        correct += (outputs_test.squeeze(-1).round()==ground_truth).sum()
        total += len(ground_truth)
    return {'test_loss': np.mean(list_loss_test), 'test_accuracy': correct/total}


def display_losses(train_loss, val_loss, config):
    plt.plot(np.arange(2,len(train_loss)+1), train_loss[1:], label='train loss')
    plt.plot(np.arange(2,len(val_loss)+1), val_loss[1:], label='validation loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title(f"Loss for {config['dataset']['target_name']}")
    plt.legend()
    plt.grid('on')
    plt.savefig(f"results/loss_{config['model']['name']}_for_{config['dataset']['target_name']}.png")