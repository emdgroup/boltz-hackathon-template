#!/usr/bin/env python
"""
Training script for Flow Matching model.

This script shows how to train the flow matching model either:
1. Fine-tuning from pretrained score-based weights (RECOMMENDED)
2. Training from scratch

Usage:
    # Fine-tune from pretrained
    python train_flow_matching.py --pretrained_checkpoint path/to/score_model.pt
    
    # Train from scratch
    python train_flow_matching.py
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from pathlib import Path
import time
import json

from src.boltz.model.modules.diffusionv3_flow_matching import AtomDiffusion


def create_model(args, pretrained_checkpoint=None):
    """
    Create flow matching model, optionally loading pretrained weights.
    """
    # Model configuration (adjust to match your setup)
    score_model_args = {
        "token_s": args.token_s,
        "atom_s": args.atom_s,
        "atoms_per_window_queries": 32,
        "atoms_per_window_keys": 128,
        "sigma_data": 16,
        "dim_fourier": 256,
        "atom_encoder_depth": 3,
        "atom_encoder_heads": 4,
        "token_transformer_depth": 24,
        "token_transformer_heads": 8,
        "atom_decoder_depth": 3,
        "atom_decoder_heads": 4,
        "conditioning_transition_layers": 2,
        "activation_checkpointing": args.activation_checkpointing,
    }
    
    # Create model
    model = AtomDiffusion(
        score_model_args=score_model_args,
        num_sampling_steps=args.num_sampling_steps,
        sigma_data=16.0,
        coordinate_augmentation=True,
    )
    
    # Load pretrained weights if provided
    if pretrained_checkpoint is not None:
        print(f"Loading pretrained weights from {pretrained_checkpoint}")
        checkpoint = torch.load(pretrained_checkpoint, map_location='cpu')
        
        # Try to load score_model weights
        if "score_model" in checkpoint:
            model.score_model.load_state_dict(checkpoint["score_model"])
            print("✓ Loaded score_model weights")
        elif "model_state_dict" in checkpoint:
            model.score_model.load_state_dict(checkpoint["model_state_dict"])
            print("✓ Loaded model_state_dict weights")
        else:
            # Assume checkpoint is the state dict directly
            model.score_model.load_state_dict(checkpoint)
            print("✓ Loaded checkpoint weights")
        
        print("✓ Successfully initialized from pretrained score model!")
        print("  Network already knows protein geometry - training will converge 5-10x faster!")
    else:
        print("Training from scratch (will take longer)")
    
    return model


def train_epoch(model, dataloader, optimizer, device, epoch, args):
    """
    Train for one epoch.
    """
    model.train()
    
    total_loss = 0.0
    total_velocity_mse = 0.0
    total_lddt_loss = 0.0
    num_batches = 0
    
    start_time = time.time()
    
    for batch_idx, batch in enumerate(dataloader):
        # Move batch to device
        batch = {k: v.to(device) if torch.is_tensor(v) else v 
                for k, v in batch.items()}
        
        # Forward pass - flow matching
        out_dict = model(
            s_inputs=batch["s_inputs"],
            s_trunk=batch["s_trunk"],
            feats=batch["feats"],
            diffusion_conditioning=batch["diffusion_conditioning"],
            multiplicity=args.multiplicity,
        )
        
        # Compute flow matching loss
        loss_dict = model.compute_loss(
            feats=batch["feats"],
            out_dict=out_dict,
            add_smooth_lddt_loss=args.use_lddt_loss,
            multiplicity=args.multiplicity,
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss_dict["loss"].backward()
        
        # Gradient clipping (important for stability)
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        
        optimizer.step()
        
        # Track metrics
        total_loss += loss_dict["loss"].item()
        total_velocity_mse += loss_dict["loss_breakdown"]["velocity_mse_loss"].item()
        if args.use_lddt_loss:
            total_lddt_loss += loss_dict["loss_breakdown"]["smooth_lddt_loss"].item()
        num_batches += 1
        
        # Log progress
        if batch_idx % args.log_interval == 0:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch} [{batch_idx}/{len(dataloader)}] "
                  f"Loss: {loss_dict['loss'].item():.4f} "
                  f"Velocity MSE: {loss_dict['loss_breakdown']['velocity_mse_loss'].item():.4f} "
                  f"Time: {elapsed:.1f}s")
    
    # Epoch statistics
    avg_loss = total_loss / num_batches
    avg_velocity_mse = total_velocity_mse / num_batches
    avg_lddt_loss = total_lddt_loss / num_batches if args.use_lddt_loss else 0.0
    
    epoch_time = time.time() - start_time
    
    print(f"\nEpoch {epoch} Summary:")
    print(f"  Avg Loss: {avg_loss:.4f}")
    print(f"  Avg Velocity MSE: {avg_velocity_mse:.4f}")
    if args.use_lddt_loss:
        print(f"  Avg lDDT Loss: {avg_lddt_loss:.4f}")
    print(f"  Time: {epoch_time:.1f}s")
    
    return {
        "loss": avg_loss,
        "velocity_mse": avg_velocity_mse,
        "lddt_loss": avg_lddt_loss,
        "time": epoch_time,
    }


def validate(model, dataloader, device, args):
    """
    Validate the model.
    """
    model.eval()
    
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v 
                    for k, v in batch.items()}
            
            out_dict = model(
                s_inputs=batch["s_inputs"],
                s_trunk=batch["s_trunk"],
                feats=batch["feats"],
                diffusion_conditioning=batch["diffusion_conditioning"],
            )
            
            loss_dict = model.compute_loss(
                feats=batch["feats"],
                out_dict=out_dict,
                add_smooth_lddt_loss=args.use_lddt_loss,
            )
            
            total_loss += loss_dict["loss"].item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    print(f"\nValidation Loss: {avg_loss:.4f}")
    
    return avg_loss


def main(args):
    """
    Main training loop.
    """
    print("=" * 60)
    print("FLOW MATCHING TRAINING")
    print("=" * 60)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    pretrained_checkpoint = args.pretrained_checkpoint if args.pretrained_checkpoint else None
    model = create_model(args, pretrained_checkpoint)
    model = model.to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )
    
    # Learning rate scheduler (optional but recommended)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs,
        eta_min=args.learning_rate * 0.01,
    )
    
    # TODO: Load your actual data
    # This is a placeholder - replace with your actual dataloader
    print("\n⚠️  WARNING: You need to replace this with your actual dataloader!")
    print("   See comments in the code for how to set up your data.\n")
    
    # train_loader = DataLoader(your_train_dataset, batch_size=args.batch_size, ...)
    # val_loader = DataLoader(your_val_dataset, batch_size=args.batch_size, ...)
    
    # For demonstration purposes only:
    train_loader = None
    val_loader = None
    
    if train_loader is None:
        print("ERROR: No training data provided. Please set up your dataloader.")
        print("\nExample dataloader setup:")
        print("""
        from your.data.module import ProteinDataset
        
        train_dataset = ProteinDataset(
            data_dir='path/to/training/data',
            # ... other args
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        """)
        return
    
    # Training loop
    print(f"\nStarting training for {args.num_epochs} epochs...")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Batch size: {args.batch_size}")
    print(f"Gradient clipping: {args.grad_clip}")
    
    best_val_loss = float('inf')
    training_history = []
    
    for epoch in range(1, args.num_epochs + 1):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, device, epoch, args
        )
        
        # Validate
        if epoch % args.val_interval == 0:
            val_loss = validate(model, val_loader, device, args)
            train_metrics["val_loss"] = val_loss
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = Path(args.output_dir) / "best_model.pt"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "args": vars(args),
                }, save_path)
                print(f"✓ Saved best model to {save_path}")
        
        # Update learning rate
        scheduler.step()
        
        # Save training history
        training_history.append(train_metrics)
        
        # Save checkpoint
        if epoch % args.save_interval == 0:
            save_path = Path(args.output_dir) / f"checkpoint_epoch_{epoch}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "args": vars(args),
            }, save_path)
            print(f"✓ Saved checkpoint to {save_path}")
    
    # Save final model
    save_path = Path(args.output_dir) / "final_model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "training_history": training_history,
        "args": vars(args),
    }, save_path)
    print(f"\n✓ Training complete! Final model saved to {save_path}")
    
    # Save training history
    history_path = Path(args.output_dir) / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    print(f"✓ Training history saved to {history_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Flow Matching Model")
    
    # Model architecture
    parser.add_argument("--token_s", type=int, default=384)
    parser.add_argument("--atom_s", type=int, default=128)
    parser.add_argument("--activation_checkpointing", action="store_true")
    
    # Training
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--multiplicity", type=int, default=1)
    
    # Sampling
    parser.add_argument("--num_sampling_steps", type=int, default=20)
    
    # Loss
    parser.add_argument("--use_lddt_loss", action="store_true", default=True)
    
    # Pretrained weights
    parser.add_argument("--pretrained_checkpoint", type=str, default=None,
                       help="Path to pretrained score model checkpoint")
    
    # Logging
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--val_interval", type=int, default=1)
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    main(args)

