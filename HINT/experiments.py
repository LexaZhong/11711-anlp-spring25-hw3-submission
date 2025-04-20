import os

import pandas as pd
import torch
from dataloader import (csv_three_feature_2_dataloader,
                        generate_admet_dataloader_lst)
from icdcode_encode import GRAM, build_icdcode2ancestor_dict
from model import HINTModel
from molecule_encode import ADMET, MPNN
from protocol_encode import Protocol_Embedding
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from torch import nn
from tqdm import tqdm

import wandb

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


def eval_one_epoch(epoch: int, model, dataloader, criterion) -> tuple:
    model.eval()  # set model in evaluation mode
    val_loss = 0.0
    y_true, y_score = [], []

    for nctid_lst, label_vec, smiles_lst2, icdcode_lst3, criteria_lst in tqdm(dataloader,
                                                                              desc=f"Epoch {epoch:02} Eval "):
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


def train(epochs: int, model, train_loader, valid_loader, optimizer, criterion, scaler):
    for epoch in tqdm(range(epochs), desc='In progress...'):
        train_loss = train_one_epoch(epoch, model, train_loader, optimizer, criterion, scaler)

        (val_loss, roc_auc, pr_auc,
         accuracy, b_accuracy,
         f1, precision, recall) = eval_one_epoch(epoch, model, valid_loader, criterion)

        wandb.log({
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


if __name__ == "__main__":
    device = ('cuda' if torch.cuda.is_available() else
              'mps' if torch.backends.mps.is_available() else
              'cpu')

    # Set wandb
    wandb.login(key="c3a06f318f071ae7444755a93fa8a5cbff1f6a86")
    config ={
        'lr': 1e-3,
        'epoch': 10,
        'device': device,
    }

    run = wandb.init(
        project="11711-hw4",  # Project should be created in your wandb account
        config=config,  # Wandb Config for your run
        reinit=True,  # Allows reinitalizing runs when you re-run this cell
        #id     = "y28t31uz", ### Insert specific run id here if you want to resume a previous run
        #resume = "must", ### You need this to resume previous runs, but comment out reinit = True when using this
    )

    # Set data
    base_name = 'phase_I'  # 'toy', 'phase_I', 'phase_II', 'phase_III', 'indication'
    datafolder = "data"
    train_file = os.path.join(datafolder, base_name + '_train.csv')
    valid_file = os.path.join(datafolder, base_name + '_valid.csv')
    test_file = os.path.join(datafolder, base_name + '_test.csv')

    # Pretrain
    mpnn_model = MPNN(mpnn_hidden_size=50, mpnn_depth=3, device=device)

    # Load data
    train_loader = csv_three_feature_2_dataloader(train_file, shuffle=True,
                                                  batch_size=32, num_workers=4)
    valid_loader = csv_three_feature_2_dataloader(valid_file, shuffle=False,
                                                  batch_size=32, num_workers=2)
    test_loader = csv_three_feature_2_dataloader(test_file, shuffle=False,
                                                 batch_size=32, num_workers=2)

    # Model
    icdcode2ancestor_dict = build_icdcode2ancestor_dict()
    gram_model = GRAM(embedding_dim=50, icdcode2ancestor=icdcode2ancestor_dict, device=device)
    protocol_model = Protocol_Embedding(output_dim=50, highway_num=3, device=device)
    model = HINTModel(molecule_encoder=mpnn_model,
                      disease_encoder=gram_model,
                      protocol_encoder=protocol_model,
                      device=device,
                      global_embed_size=50,
                      highway_num_layer=2,
                      prefix_name=base_name,
                      gnn_hidden_size=50,
                      epoch=3,
                      lr=1e-3)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'])
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.GradScaler(device)
    # train(config['epoch'], model, train_loader, valid_loader, optimizer, criterion, scaler)
