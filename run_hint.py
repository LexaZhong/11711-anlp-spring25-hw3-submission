# this script is used to train or load a HINT model for a specified phase and run inference
# How TO USE:
# python run.py --base_name phase_I --datafolder data --save_dir save_model --admet_ckpt save_model/admet_model.ckpt --device cpu --epoch 3 --lr 1e-3 --weight_decay 0.0


import os
import torch
import argparse
import warnings

warnings.filterwarnings("ignore")
torch.manual_seed(0)

from HINT.dataloader import csv_three_feature_2_dataloader
from HINT.molecule_encode import MPNN, ADMET
from HINT.icdcode_encode import GRAM, build_icdcode2ancestor_dict
from HINT.protocol_encode import Protocol_Embedding
from HINT.model import HINTModel

def main():
    parser = argparse.ArgumentParser(
        description="Train or load a HINT model for a specified phase and run inference."
    )
    parser.add_argument("--base_name", type=str, default="phase_I",
                        help="Which phase to train/test (e.g. 'phase_I', 'phase_II', 'phase_III', or 'indication').")
    parser.add_argument("--datafolder", type=str, default="data",
                        help="Folder where CSV files are located.")
    parser.add_argument("--save_dir", type=str, default="save_model",
                        help="Directory where model checkpoints are saved/loaded.")
    parser.add_argument("--admet_ckpt", type=str, default="save_model/admet_model.ckpt",
                        help="Path to the pretrained ADMET model checkpoint.")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to use: 'cpu' or 'cuda:0' etc.")
    parser.add_argument("--epoch", type=int, default=3,
                        help="Number of epochs for training the HINT model.")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate for training the HINT model.")
    parser.add_argument("--weight_decay", type=float, default=0.0,
                        help="Weight decay (L2 regularization).")

    args = parser.parse_args()

   # SET DEVICE AND CREATE FOLDERS
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Make a figure folder if it doesn't exist (referring to the tutorial notebook)
    if not os.path.exists("figure"):
        os.makedirs("figure")

    base_name = args.base_name 
    datafolder = args.datafolder
    train_file = os.path.join(datafolder, base_name + '_train.csv')
    valid_file = os.path.join(datafolder, base_name + '_valid.csv')
    test_file  = os.path.join(datafolder, base_name + '_test.csv')

    print(f"[INFO] Phase: {base_name}")
    print(f"[INFO] Train file: {train_file}")
    print(f"[INFO] Valid file: {valid_file}")
    print(f"[INFO] Test file:  {test_file}")

    mpnn_model = MPNN(mpnn_hidden_size=50, mpnn_depth=3, device=device)

    admet_model_path = args.admet_ckpt
    if not os.path.exists(admet_model_path):
        raise FileNotFoundError(
            f"[ERROR] No ADMET model checkpoint found at '{admet_model_path}'. "
            f"Please provide a valid path via --admet_ckpt."
        )
    admet_model = torch.load(admet_model_path, map_location=device, weights_only=False)
    admet_model = admet_model.to(device)
    admet_model.set_device(device)
    print(f"[INFO] Loaded ADMET model from: {admet_model_path}")

    # CREATE DATA LOADERS FOR TRAIN, VALID, TEST
    train_loader = csv_three_feature_2_dataloader(train_file, shuffle=True,  batch_size=32)
    valid_loader = csv_three_feature_2_dataloader(valid_file, shuffle=False, batch_size=32)
    test_loader  = csv_three_feature_2_dataloader(test_file,  shuffle=False, batch_size=32)

    # BUILD DISEASE (GRAM) AND PROTOCOL EMBEDDING MODELS
    icdcode2ancestor_dict = build_icdcode2ancestor_dict()
    gram_model = GRAM(embedding_dim=50, icdcode2ancestor=icdcode2ancestor_dict, device=device)
    protocol_model = Protocol_Embedding(output_dim=50, highway_num=3, device=device)

    # CHECK IF HINT MODEL CHECKPOINT EXISTS; IF NOT, TRAIN
    os.makedirs(args.save_dir, exist_ok=True)
    hint_model_path = os.path.join(args.save_dir, base_name + ".ckpt")

    if not os.path.exists(hint_model_path):
        print(f"[INFO] No checkpoint for {base_name}. Creating and training a new HINT model.")

        # Create HINT model
        model = HINTModel(
            molecule_encoder=mpnn_model,
            disease_encoder=gram_model,
            protocol_encoder=protocol_model,
            device=device,
            global_embed_size=50,
            highway_num_layer=2,
            prefix_name=base_name,
            gnn_hidden_size=50,
            epoch=args.epoch,
            lr=args.lr,
            weight_decay=args.weight_decay
        )

        # Initialize model with the loaded ADMET model
        model.init_pretrain(admet_model)

        # Train HINT model
        model.learn(train_loader, valid_loader, test_loader)

        # Inference after training
        model.bootstrap_test(test_loader)

        # Save checkpoint
        torch.save(model, hint_model_path)
        print(f"[INFO] HINT model saved to: {hint_model_path}")

    else:
        print(f"[INFO] Found existing checkpoint: {hint_model_path}")
        # Load and run inference
        model = torch.load(hint_model_path, map_location=device)
        model.bootstrap_test(test_loader)


if __name__ == "__main__":
    main()
