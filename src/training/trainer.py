"""
Training module for Autoregressive Diffusion Models.

This module implements the training loop that handles iterative refinement
and dynamic overwrite probabilities during training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, List
import wandb
import tqdm
import numpy as np
from pathlib import Path

from ..models.ardm import ARDM
from .losses import DiffusionLoss, OverwriteLoss


class ARDMTrainer:
    """
    Trainer for Autoregressive Diffusion Models.
    
    Handles the training loop with iterative refinement, dynamic overwrite
    probabilities, and various loss functions.
    """
    
    def __init__(
        self,
        model: ARDM,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-2,
        max_epochs: int = 100,
        gradient_clip_val: float = 1.0,
        use_wandb: bool = False,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        save_dir: str = "checkpoints",
        overwrite_loss_weight: float = 0.1,
        diffusion_loss_weight: float = 1.0,
    ):
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.max_epochs = max_epochs
        self.gradient_clip_val = gradient_clip_val
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Loss functions
        self.diffusion_loss = DiffusionLoss()
        self.overwrite_loss = OverwriteLoss()
        self.diffusion_loss_weight = diffusion_loss_weight
        self.overwrite_loss_weight = overwrite_loss_weight
        
        # Optimizer and scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max_epochs
        )
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'diffusion_loss': [],
            'overwrite_loss': [],
            'learning_rate': []
        }
        
        # Logging
        self.use_wandb = use_wandb
        if use_wandb:
            wandb.init(project="autoregressive-diffusions")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_diffusion_loss = 0.0
        total_overwrite_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm.tqdm(self.train_dataloader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            if isinstance(batch, (list, tuple)):
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device) if len(batch) > 1 else None
            else:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch.get('attention_mask', None)
            
            batch_size, seq_len = input_ids.shape
            
            # Sample random timesteps for diffusion
            timesteps = torch.randint(
                0, self.model.diffusion_steps,
                (batch_size,), device=self.device
            )
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # Get model predictions
            logits = self.model(input_ids, timesteps, attention_mask)
            
            # Calculate losses
            diffusion_loss = self.diffusion_loss(
                logits, input_ids, timesteps, self.model.alphas_cumprod
            )
            
            # Calculate overwrite loss based on token confidence
            overwrite_loss = self.overwrite_loss(
                logits, input_ids, timesteps, self.model.overwrite_probs
            )
            
            # Combined loss
            total_batch_loss = (
                self.diffusion_loss_weight * diffusion_loss +
                self.overwrite_loss_weight * overwrite_loss
            )
            
            # Backward pass
            total_batch_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.gradient_clip_val
            )
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += total_batch_loss.item()
            total_diffusion_loss += diffusion_loss.item()
            total_overwrite_loss += overwrite_loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{total_batch_loss.item():.4f}",
                'diff_loss': f"{diffusion_loss.item():.4f}",
                'overwrite_loss': f"{overwrite_loss.item():.4f}"
            })
            
            # Log to wandb if enabled
            if self.use_wandb and batch_idx % 10 == 0:
                wandb.log({
                    'batch_loss': total_batch_loss.item(),
                    'batch_diffusion_loss': diffusion_loss.item(),
                    'batch_overwrite_loss': overwrite_loss.item(),
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })
        
        # Calculate average losses
        avg_loss = total_loss / num_batches
        avg_diffusion_loss = total_diffusion_loss / num_batches
        avg_overwrite_loss = total_overwrite_loss / num_batches
        
        return {
            'loss': avg_loss,
            'diffusion_loss': avg_diffusion_loss,
            'overwrite_loss': avg_overwrite_loss
        }
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        if self.val_dataloader is None:
            return {}
            
        self.model.eval()
        total_loss = 0.0
        total_diffusion_loss = 0.0
        total_overwrite_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                # Move batch to device
                if isinstance(batch, (list, tuple)):
                    input_ids = batch[0].to(self.device)
                    attention_mask = batch[1].to(self.device) if len(batch) > 1 else None
                else:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch.get('attention_mask', None)
                
                batch_size, seq_len = input_ids.shape
                
                # Sample random timesteps for diffusion
                timesteps = torch.randint(
                    0, self.model.diffusion_steps,
                    (batch_size,), device=self.device
                )
                
                # Forward pass
                logits = self.model(input_ids, timesteps, attention_mask)
                
                # Calculate losses
                diffusion_loss = self.diffusion_loss(
                    logits, input_ids, timesteps, self.model.alphas_cumprod
                )
                
                overwrite_loss = self.overwrite_loss(
                    logits, input_ids, timesteps, self.model.overwrite_probs
                )
                
                total_batch_loss = (
                    self.diffusion_loss_weight * diffusion_loss +
                    self.overwrite_loss_weight * overwrite_loss
                )
                
                # Update metrics
                total_loss += total_batch_loss.item()
                total_diffusion_loss += diffusion_loss.item()
                total_overwrite_loss += overwrite_loss.item()
                num_batches += 1
        
        # Calculate average losses
        avg_loss = total_loss / num_batches
        avg_diffusion_loss = total_diffusion_loss / num_batches
        avg_overwrite_loss = total_overwrite_loss / num_batches
        
        return {
            'loss': avg_loss,
            'diffusion_loss': avg_diffusion_loss,
            'overwrite_loss': avg_overwrite_loss
        }
    
    def train(self) -> Dict[str, List[float]]:
        """Main training loop."""
        print(f"Starting training for {self.max_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(self.max_epochs):
            self.current_epoch = epoch
            
            # Training
            train_metrics = self.train_epoch()
            
            # Validation
            val_metrics = self.validate_epoch()
            
            # Update learning rate
            self.scheduler.step()
            
            # Log metrics
            self.training_history['train_loss'].append(train_metrics['loss'])
            self.training_history['diffusion_loss'].append(train_metrics['diffusion_loss'])
            self.training_history['overwrite_loss'].append(train_metrics['overwrite_loss'])
            self.training_history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            if val_metrics:
                self.training_history['val_loss'].append(val_metrics['loss'])
                
                # Save best model
                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    self.save_checkpoint('best_model.pt')
            
            # Print epoch summary
            print(f"\nEpoch {epoch + 1}/{self.max_epochs}")
            print(f"Train Loss: {train_metrics['loss']:.4f}")
            print(f"Train Diffusion Loss: {train_metrics['diffusion_loss']:.4f}")
            print(f"Train Overwrite Loss: {train_metrics['overwrite_loss']:.4f}")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            if val_metrics:
                print(f"Val Loss: {val_metrics['loss']:.4f}")
                print(f"Best Val Loss: {self.best_val_loss:.4f}")
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pt')
            
            # Log to wandb
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_metrics['loss'],
                    'train_diffusion_loss': train_metrics['diffusion_loss'],
                    'train_overwrite_loss': train_metrics['overwrite_loss'],
                    'val_loss': val_metrics.get('loss', None),
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })
        
        print("\nTraining completed!")
        return self.training_history
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint_path = self.save_dir / filename
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'training_history': self.training_history,
            'model_config': {
                'vocab_size': self.model.vocab_size,
                'max_seq_len': self.model.max_seq_len,
                'hidden_dim': self.model.hidden_dim,
                'num_layers': self.model.transformer.layers.__len__(),
                'num_heads': self.model.transformer.layers[0].self_attn.num_heads,
                'diffusion_steps': self.model.diffusion_steps,
                'overwrite_prob_base': self.model.overwrite_prob_base,
                'overwrite_decay': self.model.overwrite_decay,
            }
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.training_history = checkpoint['training_history']
        
        print(f"Checkpoint loaded from {checkpoint_path}")
        print(f"Resuming from epoch {self.current_epoch + 1}")
    
    def generate_sample_text(self, prompt: str = "", max_length: int = 100) -> str:
        """Generate sample text using the trained model."""
        self.model.eval()
        
        # Tokenize prompt if provided
        # Note: This is a simplified tokenizer - in practice, use a proper tokenizer
        if prompt:
            # Simple character-level tokenization for demonstration
            prompt_tokens = [ord(c) % self.model.vocab_size for c in prompt]
            prompt_tokens = torch.tensor([prompt_tokens], device=self.device)
        else:
            prompt_tokens = None
        
        with torch.no_grad():
            # Generate tokens
            generated_tokens = self.model.sample(
                batch_size=1,
                seq_len=max_length,
                temperature=0.8,
                top_k=50,
                top_p=0.9
            )
            
            # Convert tokens back to text
            # Note: This is simplified - in practice, use proper detokenization
            text = ''.join([chr(t.item() % 128) for t in generated_tokens[0]])
            
            if prompt:
                text = prompt + text[len(prompt):]
            
            return text 