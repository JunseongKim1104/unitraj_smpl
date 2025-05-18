import os
import sys
import argparse
import yaml
from pathlib import Path
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from omegaconf import OmegaConf

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unitraj.models.simpl_mae import SimplMAEPretrain, SimplMAEFinetune
from unitraj.datasets.simpl_mae_dataset import SimplMAEDataset


def load_config(config_path):
    # Load YAML config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Convert to OmegaConf
    config = OmegaConf.create(config)
    return config


def train_pretrain(config, output_dir):
    """Train the pretraining model"""
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Set up dataset
    config.method.is_pretraining = True
    train_dataset = SimplMAEDataset(config.method, split='train')
    val_dataset = SimplMAEDataset(config.method, split='val')
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=config.method.train_batch_size,
        shuffle=True,
        num_workers=8,
        collate_fn=train_dataset.collate_fn
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=config.method.eval_batch_size,
        shuffle=False,
        num_workers=8,
        collate_fn=val_dataset.collate_fn
    )
    
    # Create model
    model = SimplMAEPretrain(config.method)
    
    # Set up callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename='simpl_mae_pretrain-{epoch:02d}-{val_loss:.4f}',
        save_top_k=3,
        monitor='val/total_loss',
        mode='min'
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=config.method.max_epochs,
        callbacks=[checkpoint_callback, lr_monitor],
        default_root_dir=output_dir,
        gradient_clip_val=config.method.grad_clip_norm,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1
    )
    
    # Train model
    trainer.fit(model, train_loader, val_loader)
    
    return checkpoint_callback.best_model_path


def train_finetune(config, pretrained_path, output_dir):
    """Fine-tune the model using a pretrained checkpoint"""
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Update config with pretrained path
    config.method.use_pretrained = True
    config.method.pretrained_path = pretrained_path
    
    # Set up dataset
    train_dataset = SimplMAEDataset(config.method, split='train')
    val_dataset = SimplMAEDataset(config.method, split='val')
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=config.method.train_batch_size,
        shuffle=True,
        num_workers=8,
        collate_fn=train_dataset.collate_fn
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=config.method.eval_batch_size,
        shuffle=False,
        num_workers=8,
        collate_fn=val_dataset.collate_fn
    )
    
    # Create model
    model = SimplMAEFinetune(config.method)
    
    # Set up callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename='simpl_mae_finetune-{epoch:02d}-{val_loss:.4f}',
        save_top_k=3,
        monitor='val/total_loss',
        mode='min'
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=config.method.max_epochs,
        callbacks=[checkpoint_callback, lr_monitor],
        default_root_dir=output_dir,
        gradient_clip_val=config.method.grad_clip_norm,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1
    )
    
    # Train model
    trainer.fit(model, train_loader, val_loader)
    
    return checkpoint_callback.best_model_path


def export_onnx(config, checkpoint_path, output_path):
    """Export the fine-tuned model to ONNX format"""
    # Load model
    config.method.use_pretrained = True
    config.method.pretrained_path = checkpoint_path
    model = SimplMAEFinetune(config.method)
    model.eval()
    
    # Create dummy input
    # Note: This is a simplified example, you'll need to adjust based on your model's inputs
    batch_size = 1
    seq_len = config.method.g_obs_len
    dummy_actors = torch.randn(batch_size, 10, seq_len, config.method.in_actor)
    dummy_actor_idcs = [torch.arange(10)]
    dummy_lanes = torch.randn(batch_size, 50, 20, config.method.in_lane)
    dummy_lane_idcs = [torch.arange(50)]
    
    # Additional dummy inputs based on your model's requirements
    dummy_rpe_prep = {
        'scene': torch.randn(60, 60, config.method.d_rpe_in),
        'scene_mask': torch.zeros(60, 60, dtype=torch.bool)
    }
    
    dummy_input = {
        'input_dict': {
            'actors': dummy_actors,
            'actor_idcs': dummy_actor_idcs,
            'lanes': dummy_lanes,
            'lane_idcs': dummy_lane_idcs,
            'rpe_prep': dummy_rpe_prep,
            'actors_gt': torch.randn(10, 4, config.method.g_pred_len)
        }
    }
    
    # Export model
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['actors', 'actor_idcs', 'lanes', 'lane_idcs', 'rpe_prep'],
        output_names=['predicted_probability', 'predicted_trajectory'],
        dynamic_axes={
            'actors': {0: 'batch_size', 1: 'num_actors'},
            'lanes': {0: 'batch_size', 1: 'num_lanes'},
            'predicted_probability': {0: 'batch_size'},
            'predicted_trajectory': {0: 'batch_size'}
        }
    )
    
    print(f"Model exported to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SimplMAE models")
    parser.add_argument('--mode', type=str, required=True, choices=['pretrain', 'finetune', 'export'],
                        help='Training mode: pretrain, finetune, or export')
    parser.add_argument('--pretrain_config', type=str, default='unitraj/configs/method/simpl_mae_pretrain.yaml',
                        help='Path to pretraining config file')
    parser.add_argument('--finetune_config', type=str, default='unitraj/configs/method/simpl_mae_finetune.yaml',
                        help='Path to fine-tuning config file')
    parser.add_argument('--pretrained_path', type=str, 
                        help='Path to pretrained model checkpoint for fine-tuning or exporting')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Output directory for checkpoints')
    parser.add_argument('--export_path', type=str, default='outputs/simpl_mae_finetune.onnx',
                        help='Path to save the exported ONNX model')
    
    args = parser.parse_args()
    
    if args.mode == 'pretrain':
        config = load_config(args.pretrain_config)
        pretrained_path = train_pretrain(config, args.output_dir)
        print(f"Pretraining completed. Best model saved at: {pretrained_path}")
    
    elif args.mode == 'finetune':
        if args.pretrained_path is None:
            raise ValueError("--pretrained_path must be specified for fine-tuning")
        
        config = load_config(args.finetune_config)
        finetuned_path = train_finetune(config, args.pretrained_path, args.output_dir)
        print(f"Fine-tuning completed. Best model saved at: {finetuned_path}")
    
    elif args.mode == 'export':
        if args.pretrained_path is None:
            raise ValueError("--pretrained_path must be specified for exporting")
        
        config = load_config(args.finetune_config)
        export_onnx(config, args.pretrained_path, args.export_path)
        print(f"Model exported to: {args.export_path}") 