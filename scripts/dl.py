'''
script for running model
'''
import sys
sys.path.append('../')

import hydra
import numpy as np
import torch
import torch.nn as nn
import pickle
import pandas as pd
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split 
from tqdm import tqdm
from clearml import Task
from omegaconf import OmegaConf
import torch.optim as optim

from sklearn.metrics import (root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, 
                             accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score)


from nn.models import CNN, MLP, DeepCNN, VisionTransformer
from nn.train_val_test import train_epoch
from nn.dataset import MatricesDataset


@hydra.main(version_base=None, config_path="configs", config_name="trainer")
def main(config):
    # Create task for logging
    if hasattr(config, 'project_name'):
        task = Task.init(project_name=config.project_name, task_name=config.task_name,
                        reuse_last_task_id=False)
        task.connect(OmegaConf.to_container(config))
    else:
        task = None

    # Create dataset
    if config.dataset.capacity==0.1:
        df_clear, matrices_clear = load_small_dataset(config)
    else:
        df_clear, matrices_clear = load_and_prepare_dataset(config)
    
    # Create dataloaders
    train_dataloader, test_dataloader, validate_dataloader = create_dataloaders(df_clear, matrices_clear, config)

    # Model, criterion and optimizer
    model = create_model(config)
    criterion = nn.MSELoss()
    optimizer = optim.NAdam(model.parameters(), lr=config.trainer_params.learning_rate, weight_decay=config.trainer_params.weight_decay)

    # Train model
    model_train(model, optimizer, criterion, train_dataloader, validate_dataloader, config, task)
    model.load_state_dict(torch.load(f"{config.models_weight_path}{config.model.model_name}_for_{config.dataset.target_name}.pth"))

    # Validate model
    val_metrics = test_model(model, criterion, validate_dataloader, config)
    val_metrics = pd.DataFrame(val_metrics, index=[f"{config.dataset.target_name}_{config.model.model_name}"])
    print('Validation metrics:')
    print(val_metrics)
    if task:
        task.get_logger().report_table(title='Validation metrics table', series='pandas DataFrame', iteration=0, table_plot=val_metrics)

    # Test model 
    test_metrics = test_model(model, criterion, test_dataloader, config)
    test_metrics = pd.DataFrame(test_metrics, index=[f"{config.dataset.target_name}_{config.model.model_name}"])
    print('Test metrics:')
    print(test_metrics)
    if task:
        task.get_logger().report_table(title='Test metrics table', series='pandas DataFrame', iteration=0, table_plot=test_metrics)
    
    if task:
        task.close()
    return None

def load_small_dataset(config):
    # load matrices
    with open(config.data_path+'padded_matrices_small.pickle', 'rb') as f:
        matrices_clear = pickle.load(f)
    
    # load hodge numbers
    df_clear = pd.read_csv(config.data_path+'cicy4folds_extended_small.csv')
    assert len(df_clear)==len(matrices_clear)
    print(f"Size of dataset is {len(matrices_clear)} matrices")

    return df_clear, matrices_clear

def load_and_prepare_dataset(config):
    # load matrices
    with open(config.data_path+'padded_matrices.pickle', 'rb') as f:
        matrices = pickle.load(f)
    
    # load hodge numbers
    df = pd.read_csv(config.data_path+'cicy4folds_extended.csv')
    
    # drop null values
    size_old_dataset = len(df)
    df = df.replace('Null', np.nan)
    nan_indeces = df.index[df.isna().any(axis=1)].tolist()
    df_clear = df.drop(nan_indeces)
    print(f'Drop {size_old_dataset-len(df_clear)} null values, which is {(size_old_dataset-len(df_clear))/size_old_dataset*100:.2f} % of whole dataset')
    matrices_clear = np.delete(matrices, nan_indeces, axis=0)

    assert len(df_clear)==len(matrices_clear)

    # prepare smaller version of dataset
    size_old_dataset = len(df_clear)
    if config.dataset.capacity < 1:
        selected_indices = np.random.choice(np.arange(0, round(size_old_dataset)), size=round(config.dataset.capacity*size_old_dataset), replace=False)
        df_clear = df_clear.iloc[selected_indices]
        matrices_clear = matrices_clear[selected_indices]
    print(f"Size of dataset is {len(matrices_clear)} matrices, which is {len(matrices_clear)/size_old_dataset*100:.2f} % of full dataset")

    return df_clear, matrices_clear

def create_dataloaders(df_clear, matrices_clear, config):
    # split data
    df_train_validate, df_test, matrices_train_validate, matrices_test = train_test_split(df_clear, matrices_clear, test_size=config.dataset.test_size, shuffle=True)
    df_train, df_validate, matrices_train, matrices_validate = train_test_split(df_train_validate, matrices_train_validate, test_size=config.dataset.test_size, shuffle=True)

    # create class Dataset
    train_ds = MatricesDataset(df_train, matrices_train, config.dataset.target_name)
    validate_ds = MatricesDataset(df_validate, matrices_validate, config.dataset.target_name)
    test_ds = MatricesDataset(df_test, matrices_test, config.dataset.target_name)

    # create dataloaders
    train_dataloader = DataLoader(train_ds, batch_size=config.dataloader.train_batch, num_workers=config.dataloader.num_workers, shuffle=True)
    test_dataloader = DataLoader(test_ds, batch_size=config.dataloader.test_batch, num_workers=config.dataloader.num_workers, shuffle=False)
    validate_dataloader = DataLoader(validate_ds, batch_size=config.dataloader.validation_batch, num_workers=config.dataloader.num_workers, shuffle=True)

    return train_dataloader, test_dataloader, validate_dataloader

def create_model(config):
    if (config.model.model_name=='CNN'):
        model = CNN(config.model.model_params).to(config.model.device)
    elif (config.model.model_name=='MLP'):
        model = MLP(config.model.model_params).to(config.model.device)
    elif (config.model.model_name=='DeepCNN'):
        model = DeepCNN(config.model.model_params).to(config.model.device)
    elif (config.model.model_name=='VisionTransformer'):
        model = VisionTransformer(config.model.model_params).to(config.model.device)
    else:
        raise NameError(f"There isn't model with name {config.model.model_name}")
    
    return model

def model_train(model, optimizer, criterion, train_dataloader, validate_dataloader, config, task):
    train_loss = []
    val_loss = []
    min_val_loss = np.inf
    for epoch in tqdm(range(config.trainer_params.num_epochs)):
        losses = train_epoch(model, optimizer, criterion, train_dataloader, validate_dataloader, config)

        print(f"{epoch+1}: loss on train: {losses['train_loss']:.4f}, loss on validation: {losses['validation_loss']:.4f}")
        train_loss.append(losses['train_loss'])
        val_loss.append(losses['validation_loss'])
        if(losses['validation_loss'] < min_val_loss):
            min_val_loss = losses['validation_loss']
            torch.save(model.state_dict(), f"{config.models_weight_path}{config.model.model_name}_for_{config.dataset.target_name}.pth")

        if task:
            task.get_logger().report_scalar(title="Loss", series="train loss", iteration=(epoch+1), value=losses['train_loss'])
            task.get_logger().report_scalar(title="Loss", series="validation loss", iteration=(epoch+1), value=losses['validation_loss'])
            task.get_logger().report_scalar(title="Accuracy", series="train accuracy", iteration=(epoch+1), value=losses['train_accuracy'])
            task.get_logger().report_scalar(title="Accuracy", series="validation accuracy", iteration=(epoch+1), value=losses['validation_accuracy'])
    
    return model

def test_model(model, criterion, test_dataloader, config):
    list_loss_test = []
    predictions = np.array([])
    targets = np.array([])

    model.eval()
    for batch in tqdm(test_dataloader):
        outputs_test = model(batch['matrix'].to(config.model.device))
        ground_truth = batch['hodge_number'].to(config.model.device)
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

if __name__ == "__main__":
    main()