"""
Implements training loop for the VAE model.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
import datetime
import argparse as ap
from dataset import FastaDataset
from vae_model import VAE, AE, CAE, CVAE
from constants import max_length, input_dim, latent_dim, hidden_dim, TOKENS, blosum62_tensor

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
log_dir = f"runs/experiment_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
writer = SummaryWriter(log_dir)

def parse_args():
    parser = ap.ArgumentParser(description="Train a VAE model on protein sequences.")
    parser.add_argument(
        "--train-file",
        type=str,
        required=True,
        help="Path to the input FASTA file."
    )
    parser.add_argument(
        "--cached-train",
        type=str,
        required=False,
        default=None,
        help="Path to the input FASTA file."
    )
    parser.add_argument(
        "--valid-file",
        type=str,
        required=True,
        help="Path to the input FASTA file."
    )
    parser.add_argument(
        "--cached-valid",
        type=str,
        required=False,
        default=None,
        help="Path to the input FASTA file."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training."
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=42
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.001
    )
    parser.add_argument(
        '--save-interval',
        type=int,
        default=20
    )
    parser.add_argument(
        '--log-interval',
        type=int,
        default=1
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None
    )
    parser.add_argument(
        '--output-prefix',
        type=str,
        default='vae_state.pth',
    )
    parser.add_argument(
        '--beta',
        type=float,
        default=1.,
    )
    parser.add_argument(
        '--anneal-epoch',
        type=int,
        default=20,
    )
    parser.add_argument(
        '--anneal-period',
        type=int,
        default=20,
    )
    parser.add_argument(
        '--kl-upper-limit',
        type=int,
        default=100000,
    )
    parser.add_argument(
        '--kl-lower-limit',
        type=int,
        default=10,
    )
    parser.add_argument(
        '--kl-pressure-period',
        type=int,
        default=150,
    )
    return parser.parse_args()


def get_linear_warmup_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))
    return LambdaLR(optimizer, lr_lambda)


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, global_step, path="checkpoint.pth"):
    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved at {path}")


def load_checkpoint(model, optimizer, scheduler, scaler, path="checkpoint.pth", device="cuda"):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scheduler.load_state_dict(checkpoint["scheduler_state"])
    scaler.load_state_dict(checkpoint["scaler_state"])
    epoch = checkpoint["epoch"] + 1
    global_step = checkpoint["global_step"]
    print(f"Checkpoint loaded from {path}, resuming from epoch {epoch}, step {global_step}")
    return epoch, global_step


def init_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv1d)):
        nn.init.kaiming_normal_(m.weight)


def main(
        train_file: str,
        cached_train: str,
        valid_file: str,
        cached_valid: str,
        batch_size: int,
        epochs: int,
        learning_rate: float,
        output_prefix: str,
        save_interval: int,
        log_interval: int,
        checkpoint: str,
        hyperparameter: dict,
        blosum_tensor
):
    # Load the dataset
    dataset = FastaDataset(train_file, max_length=max_length, h5file=cached_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    testset = FastaDataset(valid_file, max_length=max_length, h5file=cached_valid)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)

    #vae = VAE(input_dim=input_dim, latent_dim=latent_dim, hidden_dim=hidden_dim).to(device)
    #vae.weight_init()
    #optimizer = optim.Adam(vae.parameters(), lr=learning_rate)

    #ae = AE(latent_dim=latent_dim, hidden_dim=hidden_dim).to(device)
    #ae.weight_init()
    #optimizer = optim.Adam(ae.parameters(), lr=learning_rate)

    model = CVAE(latent_dim=latent_dim).to(device)
    model.apply(init_weights)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    scaler = GradScaler('cuda')  # For FP16
    clip_norm = 1.0

    train_steps_per_epoch = len(dataset) // batch_size + 1
    total_steps = epochs * train_steps_per_epoch
    warmup_steps = int(0.1 * total_steps)  # 10% of total steps for warmup
    scheduler = get_linear_warmup_scheduler(optimizer, warmup_steps, total_steps)

    kl_anneal_epoch = hyperparameter['anneal_epoch']
    kl_anneal_period = hyperparameter['anneal_period']
    #blosum_tensor = blosum_tensor.to(device)

    if checkpoint:
        trained_epoch, global_step = load_checkpoint(model, optimizer, scheduler, scaler, checkpoint, device=device)
    else:
        trained_epoch, global_step = 0, 0

    # Training loop
    for epoch in range(trained_epoch, trained_epoch + epochs):
        model.train()
        total_loss = 0
        train_recon = 0
        train_kl_div = 0
        beta = 0.0 if epoch < kl_anneal_epoch else hyperparameter['beta'] * min(1.0, (epoch+1-kl_anneal_epoch) / kl_anneal_period)
        #beta = 0.01# * min(1.0, (epoch+1)/kl_anneal_epoch)

        for i, batch in enumerate(dataloader):
            batch = batch.to(device)
            optimizer.zero_grad()
            with autocast('cuda'):  # Enable FP16
                with torch.autograd.set_detect_anomaly(True):
                    recon_x, mean, logvar = model(batch)
                #recon_x = model(batch)
                recon_loss, kl_div = CVAE.loss_fn(recon_x, batch, mean, logvar)
                #recon_loss, blosum = CAE.loss_fn(recon_x, batch, blosum_tensor)
                loss = recon_loss + beta * kl_div
                #loss = recon_loss + beta * blosum
                if torch.isnan(loss).any():
                    print("NaN detected in logits!")
            with torch.autograd.set_detect_anomaly(True):
                scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_recon += recon_loss.item()
            train_kl_div += kl_div.item()
            total_loss += loss.item()
            global_step += 1

            for name, param in model.named_parameters():
                if torch.isnan(param).any():
                    print(f"NaN in weights: {name}")

        #print(
        #    f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}")

        if train_kl_div / len(dataloader) > hyperparameter['kl_upper_limit'] and epoch < kl_anneal_epoch:
            print('Activate KL limit')
            kl_anneal_epoch = epoch

        if train_kl_div / len(dataloader) < hyperparameter['kl_lower_limit'] and epoch >= kl_anneal_epoch:
            print('Deactivate KL limit')
            kl_anneal_epoch = epoch + 1

        if epoch > hyperparameter['kl_pressure_period']:
            hyperparameter['kl_lower_limit'] = 0
            kl_anneal_epoch = epoch

        print(
            f"Epoch {epoch + 1}/{epochs}, Recon: {train_recon / len(dataloader):.4f}, KL Div: {train_kl_div / len(dataloader):.4f}, Loss: {total_loss / len(dataloader):.4f}")

        if (epoch+1) % save_interval == 0:
            # Save the model weights
            output_state = output_prefix + '_' + str(epoch+1) + '.pth'
            save_checkpoint(model, optimizer, scheduler, scaler, epoch, global_step, output_state)

        if (epoch+1) % log_interval == 0:
            # Save the model weights
            writer.add_scalar("Loss/train_recon", train_recon / len(dataloader), global_step)
            writer.add_scalar("Loss/train_kl_div", train_kl_div / len(dataloader), global_step)
            writer.add_scalar("Loss/train_total", total_loss / len(dataloader), global_step)
            writer.add_scalar("LR", scheduler.get_last_lr()[0], global_step)
            model.eval()
            test_recon = 0
            test_kl_div = 0
            total_tloss = 0
            for j, batch in enumerate(testloader):
                batch = batch.to(device)
                optimizer.zero_grad()
                with autocast('cuda'):  # Enable FP16
                    recon_x, mean, logvar = model(batch)
                    #recon_x = model(batch)
                    trecon_loss, tkl_div = CVAE.loss_fn(recon_x, batch, mean, logvar)
                #trecon_loss, blosum = CAE.loss_fn(recon_x, batch, blosum_tensor)
                test_recon += trecon_loss.item()
                test_kl_div += tkl_div.item()
                #test_recon += trecon_loss.item()
                #test_kl_div += blosum.item()
                tloss = trecon_loss + beta * tkl_div
                total_tloss += tloss.item()
            writer.add_scalar("Loss/test_recon", test_recon / len(testloader), global_step)
            writer.add_scalar("Loss/test_kl", test_kl_div / len(testloader), global_step)
            writer.add_scalar("Loss/test_total", total_tloss / len(testloader), global_step)


if __name__ == "__main__":
    args = parse_args()
    train_file = args.train_file
    cached_train = args.cached_train
    valid_file = args.valid_file
    cached_valid = args.cached_valid
    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.learning_rate
    output_prefix = args.output_prefix
    save_interval = args.save_interval
    log_interval = args.log_interval
    checkpoint = args.checkpoint
    beta = args.beta
    anneal_epoch = args.anneal_epoch
    anneal_period = args.anneal_period
    kl_upper_limit = args.kl_upper_limit
    kl_lower_limit = args.kl_upper_limit
    kl_pressure_period = args.kl_pressure_period
    hyperparameter = {
        'beta': beta,
        'anneal_epoch': anneal_epoch,
        'anneal_period': anneal_period,
        'kl_upper_limit': kl_upper_limit,
        'kl_lower_limit': kl_lower_limit,
        'kl_pressure_period': kl_pressure_period
    }

    main(
        train_file=train_file,
        cached_train=cached_train,
        valid_file=valid_file,
        cached_valid=cached_valid,
        batch_size=batch_size,
        hyperparameter=hyperparameter,
        epochs=epochs,
        learning_rate=learning_rate,
        output_prefix=output_prefix,
        save_interval=save_interval,
        log_interval=log_interval,
        checkpoint=checkpoint,
        blosum_tensor=blosum62_tensor
    )
