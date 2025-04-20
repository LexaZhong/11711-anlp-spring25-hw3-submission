import argparse
import os

import numpy as np
import pandas as pd
import torch
from dataloader import (Trial_Dataset, csv_three_feature_2_dataloader,
                        generate_admet_dataloader_lst)
from icdcode_encode import GRAM, build_icdcode2ancestor_dict
from model import HINTModel
from molecule_encode import ADMET, MPNN
from protocol_encode import Protocol_Embedding
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import wandb

parser = argparse.ArgumentParser()
parser.add_argument('--base_name', type=str)
args = parser.parse_args()

device = ('cuda' if torch.cuda.is_available() else
          'mps' if torch.backends.mps.is_available() else
          'cpu')


def train_one_epoch(epoch: int, model, dataloader, optimizer, criterion, scaler) -> float:
    model.train()
    train_loss = 0.0
    for nctid_lst, label_vec, smiles_lst2, icdcode_lst3, criteria_lst in tqdm(dataloader,
                                                                              desc=f"Epoch {epoch:02} Train"):
        labels = label_vec.to(device).float()
        optimizer.zero_grad()  # Initialize Gradients

        # Forward Propagation
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            logits = model.forward(smiles_lst2, icdcode_lst3, criteria_lst).view(-1)
            loss = criterion(logits, labels)

        # Backward Propagation
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item()

        # Release memory
        del labels, logits
        torch.cuda.empty_cache()

    train_loss /= len(dataloader)
    return train_loss


def eval_one_epoch(epoch: int, model, dataloader, criterion, desc='Eval') -> tuple:
    model.eval()  # set model in evaluation mode
    val_loss = 0.0
    y_true, y_score = [], []

    for nctid_lst, label_vec, smiles_lst2, icdcode_lst3, criteria_lst in tqdm(dataloader,
                                                                              desc=f"Epoch {epoch:02} {desc} "):
        labels = label_vec.to(device).float()
        # Forward Propagation
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            logits = model.forward(smiles_lst2, icdcode_lst3, criteria_lst).view(-1)
            loss = criterion(logits, labels)
        y_true.extend([i.item() for i in labels])
        y_score.extend([i.item() for i in torch.sigmoid(logits)])
        val_loss += loss.item()

    # Evaluate the whole batch
    y_pred = [1 if i >= 0.5 else 0 for i in y_score]
    val_loss /= len(dataloader)
    roc_auc = roc_auc_score(y_true, y_score)
    pr_auc = average_precision_score(y_true, y_score)
    accuracy = accuracy_score(y_true, y_pred)
    b_accuracy = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    return val_loss, roc_auc, pr_auc, accuracy, b_accuracy, f1, precision, recall


def train(epochs: int, model, train_loader, valid_loader,
          optimizer, scheduler, criterion, scaler) -> dict:
    best_val_loss = np.inf
    checkpoint = {}

    for epoch in tqdm(range(epochs), desc='In progress...'):
        train_loss = train_one_epoch(epoch, model, train_loader, optimizer, criterion, scaler)

        (val_loss, roc_auc, pr_auc,
         accuracy, b_accuracy,
         f1, precision, recall) = eval_one_epoch(epoch, model, valid_loader, criterion)

        scheduler.step(val_loss)

        # Log epoch results
        wandb.log({
            'curr_lr': float(optimizer.param_groups[0]['lr']),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_roc_auc': roc_auc,
            'val_pr_auc': pr_auc,
            'val_accuracy': accuracy,
            'val_balanced_accuracy': b_accuracy,
            'val_f1': f1,
            'val_precision': precision,
            'val_recall': recall
        })

        # Check if the current model is the best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }
            print(f"Best model at epoch {epoch} with val_loss: {val_loss:.4f}")

        # End training if no more improvement
        if (val_loss > best_val_loss and
                epoch - checkpoint.get('epoch', 0) > 5):
            print(f"Early stopping at epoch {epoch} with val_loss: {val_loss:.4f}")
            break

    return checkpoint


def test(epoch, model, test_loader, criterion):
    (test_loss, roc_auc, pr_auc,
     accuracy, b_accuracy,
     f1, precision, recall) = eval_one_epoch(epoch, model, test_loader, criterion, desc='Test')
    wandb.log({
        'test_roc_auc': roc_auc,
        'test_pr_auc': pr_auc,
        'test_accuracy': accuracy,
        'test_balanced_accuracy': b_accuracy,
        'test_f1': f1,
        'test_precision': precision,
        'test_recall': recall
    })


def get_dataloaders(config) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:

    def collate_fn(x):
        return [[v[i] for v in x] for i in range(len(x[0]))]

    datafolder = "data"
    train_file = os.path.join(datafolder, args.base_name + '_train.csv')
    valid_file = os.path.join(datafolder, args.base_name + '_valid.csv')
    test_file = os.path.join(datafolder, args.base_name + '_test.csv')

    train_set = Trial_Dataset(train_file, config['embedding_path'])
    valid_set = Trial_Dataset(valid_file, config['embedding_path'])
    test_set = Trial_Dataset(test_file, config['embedding_path'])

    train_loader = DataLoader(train_set, batch_size=config['batch_size'],
                              num_workers=4, shuffle=True,
                              collate_fn=collate_fn)
    valid_loader = DataLoader(valid_set, batch_size=config['batch_size'],
                              num_workers=2, shuffle=False,
                              collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=config['batch_size'],
                             num_workers=2, shuffle=False,
                             collate_fn=collate_fn)
    return train_loader, valid_loader, test_loader


def get_model(config: dict, pretrain=True):
    icdcode2ancestor_dict = build_icdcode2ancestor_dict()
    mpnn_model = MPNN(mpnn_hidden_size=config['output_dim'],
                      mpnn_depth=config['mpnn_depth'],
                      device=device)
    gram_model = GRAM(embedding_dim=config['output_dim'],
                      icdcode2ancestor=icdcode2ancestor_dict, device=device)
    protocol_model = Protocol_Embedding(input_dim=config['input_dim'],
                                        output_dim=config['output_dim'],
                                        device=device)
    model = HINTModel(molecule_encoder=mpnn_model,
                      disease_encoder=gram_model,
                      protocol_encoder=protocol_model,
                      device=device,
                      global_embed_size=config['output_dim'],
                      highway_num_layer=config['n_highway'],
                      gnn_hidden_size=config['output_dim'])

    # Pretraining model
    if pretrain:
        admet_dataloader_lst = generate_admet_dataloader_lst(batch_size=32)
        admet_trainloader_lst = [i[0] for i in admet_dataloader_lst]
        admet_testloader_lst = [i[1] for i in admet_dataloader_lst]
        admet_model = ADMET(molecule_encoder=mpnn_model,
                            highway_num=config['n_highway'],
                            device=device,
                            epoch=config['pre_epoch'],
                            lr=5e-4,
                            weight_decay=0,
                            save_name='admet_')
        admet_model.train(admet_trainloader_lst, admet_testloader_lst)
        model.init_pretrain(admet_model)
        print("Initialize pretrain model")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")

    return model


if __name__ == "__main__":

    wandb.login(key="c3a06f318f071ae7444755a93fa8a5cbff1f6a86")

    config ={
        'batch_size': 32,
        'lr': 1e-3,
        'scheduler_factor': 0.8,
        'scheduler_patience': 3,
        'epoch': 10,
        'pre_epoch': 10,
        'output_dim': 128,
        'n_highway': 5,
        'mpnn_depth': 5,
        'embedding_path': "embeddings/icd2embedding.pkl",
        'phase': args.base_name,
        'device': device,
    }
    if config['embedding_path'] == 'embeddings/icd2embedding.pkl':
        config['input_dim'] = 3072
    else:
        config['input_dim'] = 768

    train_loader, valid_loader, test_loader = get_dataloaders(config)
    model = get_model(config, pretrain=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',
                                                           factor=config['scheduler_factor'],
                                                           patience=config['scheduler_patience'],
                                                           min_lr=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.GradScaler(device)

    run = wandb.init(
        project="11711-hw4-ablation",
        config=config,
        reinit='finish_previous',
        #id     = "y28t31uz",
        #resume = "must",
    )

    checkpoint = train(config['epoch'], model, train_loader, valid_loader,
                       optimizer, scheduler, criterion, scaler)
    model.load_state_dict(checkpoint['model_state_dict'])
    test(config['epoch'], model, test_loader, criterion)

    # Save model
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(checkpoint, f'{checkpoint_dir}/{run.name}.pt')

    wandb.finish()
