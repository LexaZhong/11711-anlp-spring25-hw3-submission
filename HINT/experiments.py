import os

import torch
from dataloader import (csv_three_feature_2_dataloader,
                        generate_admet_dataloader_lst)
from icdcode_encode import GRAM, build_icdcode2ancestor_dict
from model import HINTModel
from molecule_encode import ADMET, MPNN
from protocol_encode import Protocol_Embedding

if __name__ == "__main__":
    device = ('cuda' if torch.cuda.is_available() else
              'mps' if torch.backends.mps.is_available() else
              'cpu')

    # Set data
    base_name = 'phase_I'  # 'toy', 'phase_I', 'phase_II', 'phase_III', 'indication'
    datafolder = "data"
    train_file = os.path.join(datafolder, base_name + '_train.csv')
    valid_file = os.path.join(datafolder, base_name + '_valid.csv')
    test_file = os.path.join(datafolder, base_name + '_test.csv')

    # Pretrain
    mpnn_model = MPNN(mpnn_hidden_size=50, mpnn_depth=3, device=device)

    # Load data
    train_loader = csv_three_feature_2_dataloader(train_file, shuffle=True, batch_size=32)
    valid_loader = csv_three_feature_2_dataloader(valid_file, shuffle=False, batch_size=32)
    test_loader = csv_three_feature_2_dataloader(test_file, shuffle=False, batch_size=32)

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
    model.learn(train_loader, valid_loader, test_loader)
