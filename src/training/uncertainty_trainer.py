"""
Enhanced trainer for Uncertainty-driven ARDMs.

This module implements training strategies for the uncertainty-driven
overwrite probability mechanism, including relaxed Bernoulli sampling
and auxiliary loss functions.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, List, Tuple
import wandb
import tqdm
import numpy as np
from pathlib import Path

from ..models.uncertainty_gate import UncertaintyARDM
from .losses import DiffusionLoss, OverwriteLoss


class UncertaintyARDMTrainer:
    """
    Enhanced trainer for Uncertainty-driven ARDMs.
    
    Implements the research training strategies including:
    1. Relaxed Bernoulli sampling (Gumbel-Sigmoid)
    2. Auxiliary loss for overwrite probability learning
    3. Sparsity and stability regularization
    """
    
    def __init__(
        self,
        model: UncertaintyARDM,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-2,
        max_epochs: int = 100,
        gradient_clip_val: float = 1.0,
        use_wandb: bool = False,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        save_dir: str = "checkpoints",
        # Loss weights
        diffusion_loss_weight: float = 1.0,
        overwrite_loss_weight: float = 0.1,
        sparsity_weight: float = 0.01,
        stability_weight: float = 0.001,
        # Training strategy
        use_relaxed_bernoulli: bool = True,
        gumbel_temperature: float = 1.0,
        gumbel_anneal_rate: float = 0.95,
        # Overwrite control
        target_overwrite_rate: float = 0.3,
        overwrite_budget_weight: float = 0.1,
    ):
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.max_epochs = max_epochs
        self.gradient_clip_val = gradient_clip_val
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Loss weights
        self.diffusion_loss_weight = diffusion_loss_weight
        self.overwrite_loss_weight = overwrite_loss_weight
        self.sparsity_weight = sparsity_weight
        self.stability_weight = stability_weight
        
        # Training strategy
        self.use_relaxed_bernoulli = use_relaxed_bernoulli
        self.gumbel_temperature = gumbel_temperature
        self.gumbel_anneal_rate = gumbel_anneal_rate
        
        # Overwrite control
        self.target_overwrite_rate = target_overwrite_rate
        self.overwrite_budget_weight = overwrite_budget_weight
        
        # Loss functions
        self.diffusion_loss = DiffusionLoss()
        self.overwrite_loss = OverwriteLoss()
        
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
            'sparsity_loss': [],
            'stability_loss': [],
            'overwrite_rate': [],
            'learning_rate': [],
            'gumbel_temperature': []
        }
        
        # Logging
        self.use_wandb = use_wandb
        if use_wandb:
            wandb.init(project="uncertainty-ardm")
    
    def _compute_relaxed_overwrite_mask(
        self, 
        overwrite_probs: torch.Tensor, 
        temperature: float
    ) -> torch.Tensor:
        """
        Compute relaxed overwrite mask using Gumbel-Sigmoid.
        
        Args:
            overwrite_probs: [batch_size, seq_len] overwrite probabilities
            temperature: Gumbel temperature for annealing
            
        Returns:
            Relaxed mask: [batch_size, seq_len] in (0, 1)
        """
        if not self.use_relaxed_bernoulli:
            # Hard Bernoulli sampling
            return torch.bernoulli(overwrite_probs)
        
        # Gumbel-Sigmoid relaxation
        logits = torch.log(overwrite_probs + 1e-8) - torch.log(1 - overwrite_probs + 1e-8)
        
        # Sample Gumbel noise
        gumbel_noise = torch.distributions.Logistic(0, 1).sample(logits.shape).to(logits.device)
        
        # Relaxed sampling
        relaxed_mask = torch.sigmoid((logits + gumbel_noise) / temperature)
        
        return relaxed_mask
    
    def _compute_auxiliary_loss(
        self, 
        overwrite_probs: torch.Tensor, 
        targets: torch.Tensor,
        predicted_tokens: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute auxiliary loss for overwrite probability learning.
        
        Args:
            overwrite_probs: [batch_size, seq_len] overwrite probabilities
            targets: [batch_size, seq_len] ground truth tokens
            predicted_tokens: [batch_size, seq_len] current predictions
            
        Returns:
            Auxiliary loss scalar
        """
        # Target overwrite mask: overwrite when prediction is wrong
        target_overwrite = (predicted_tokens != targets).float()
        
        # Binary cross-entropy loss
        aux_loss = F.binary_cross_entropy(overwrite_probs, target_overwrite, reduction='mean')
        
        return aux_loss
    
    def _compute_sparsity_loss(self, overwrite_probs: torch.Tensor) -> torch.Tensor:
        """
        Compute sparsity loss to encourage fewer overwrites.
        
        Args:
            overwrite_probs: [batch_size, seq_len] overwrite probabilities
            
        Returns:
            Sparsity loss scalar
        """
        # Encourage average overwrite rate to be close to target
        avg_overwrite_rate = overwrite_probs.mean()
        sparsity_loss = F.mse_loss(avg_overwrite_rate, torch.tensor(self.target_overwrite_rate, device=overwrite_probs.device))
        
        return sparsity_loss
    
    def _compute_stability_loss(self, overwrite_probs: torch.Tensor) -> torch.Tensor:
        """
        Compute stability loss to avoid oscillatory overwrite patterns.
        
        Args:
            overwrite_probs: [batch_size, seq_len] overwrite probabilities
            
        Returns:
            Stability loss scalar
        """
        # Total variation penalty across sequence
        if overwrite_probs.size(1) > 1:
            tv_loss = torch.abs(overwrite_probs[:, 1:] - overwrite_probs[:, :-1]).mean()
        else:
            tv_loss = torch.tensor(0.0, device=overwrite_probs.device)
        
        return tv_loss
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch with uncertainty-driven overwrite."""
        self.model.train()
        total_loss = 0.0
        total_diffusion_loss = 0.0
        total_overwrite_loss = 0.0
        total_sparsity_loss = 0.0
        total_stability_loss = 0.0
        total_overwrite_rate = 0.0
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
            
            # Get model predictions and hidden states
            logits, hidden_states = self.model(input_ids, timesteps, attention_mask)
            
            # Compute overwrite probabilities using uncertainty gate
            overwrite_probs, debug_info = self.model.overwrite_gate(
                h=hidden_states,
                logits=logits,
                step_t=timesteps[0].item(),  # Use first timestep for simplicity
                targets=input_ids
            )
            
            # Compute relaxed overwrite mask
            relaxed_mask = self._compute_relaxed_overwrite_mask(
                overwrite_probs, self.gumbel_temperature
            )
            
            # Calculate losses
            diffusion_loss = self.diffusion_loss(
                logits, input_ids, timesteps, self.model.alphas_cumprod
            )
            
            # Overwrite loss using relaxed mask
            overwrite_loss = self.overwrite_loss(
                logits, input_ids, timesteps, overwrite_probs
            )
            
            # Auxiliary loss for overwrite learning
            predicted_tokens = logits.argmax(dim=-1)
            aux_loss = self._compute_auxiliary_loss(overwrite_probs, input_ids, predicted_tokens)
            
            # Regularization losses
            sparsity_loss = self._compute_sparsity_loss(overwrite_probs)
            stability_loss = self._compute_stability_loss(overwrite_probs)
            
            # Combined loss
            total_batch_loss = (
                self.diffusion_loss_weight * diffusion_loss +
                self.overwrite_loss_weight * (overwrite_loss + aux_loss) +
                self.sparsity_weight * sparsity_loss +
                self.stability_weight * stability_loss
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
            total_overwrite_loss += (overwrite_loss + aux_loss).item()
            total_sparsity_loss += sparsity_loss.item()
            total_stability_loss += stability_loss.item()
            total_overwrite_rate += relaxed_mask.mean().item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{total_batch_loss.item():.4f}",
                'diff_loss': f"{diffusion_loss.item():.4f}",
                'overwrite_loss': f"{(overwrite_loss + aux_loss).item():.4f}",
                'overwrite_rate': f"{relaxed_mask.mean().item():.3f}"
            })
            
            # Log to wandb if enabled
            if self.use_wandb and batch_idx % 10 == 0:
                wandb.log({
                    'batch_loss': total_batch_loss.item(),
                    'batch_diffusion_loss': diffusion_loss.item(),
                    'batch_overwrite_loss': (overwrite_loss + aux_loss).item(),
                    'batch_sparsity_loss': sparsity_loss.item(),
                    'batch_stability_loss': stability_loss.item(),
                    'batch_overwrite_rate': relaxed_mask.mean().item(),
                    'learning_rate': self.optimizer.param_groups[0]['lr'],
                    'gumbel_temperature': self.gumbel_temperature
                })
        
        # Calculate average losses
        avg_loss = total_loss / num_batches
        avg_diffusion_loss = total_diffusion_loss / num_batches
        avg_overwrite_loss = total_overwrite_loss / num_batches
        avg_sparsity_loss = total_sparsity_loss / num_batches
        avg_stability_loss = total_stability_loss / num_batches
        avg_overwrite_rate = total_overwrite_rate / num_batches
        
        return {
            'loss': avg_loss,
            'diffusion_loss': avg_diffusion_loss,
            'overwrite_loss': avg_overwrite_loss,
            'sparsity_loss': avg_sparsity_loss,
            'stability_loss': avg_stability_loss,
            'overwrite_rate': avg_overwrite_rate
        }
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        if self.val_dataloader is None:
            return {}
            
        self.model.eval()
        total_loss = 0.0
        total_diffusion_loss = 0.0
        total_overwrite_loss = 0.0
        total_overwrite_rate = 0.0
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
                logits, hidden_states = self.model(input_ids, timesteps, attention_mask)
                
                # Compute overwrite probabilities
                overwrite_probs, _ = self.model.overwrite_gate(
                    h=hidden_states,
                    logits=logits,
                    step_t=timesteps[0].item(),
                    targets=input_ids
                )
                
                # Calculate losses
                diffusion_loss = self.diffusion_loss(
                    logits, input_ids, timesteps, self.model.alphas_cumprod
                )
                
                overwrite_loss = self.overwrite_loss(
                    logits, input_ids, timesteps, overwrite_probs
                )
                
                total_batch_loss = (
                    self.diffusion_loss_weight * diffusion_loss +
                    self.overwrite_loss_weight * overwrite_loss
                )
                
                # Update metrics
                total_loss += total_batch_loss.item()
                total_diffusion_loss += diffusion_loss.item()
                total_overwrite_loss += overwrite_loss.item()
                total_overwrite_rate += overwrite_probs.mean().item()
                num_batches += 1
        
        # Calculate average losses
        avg_loss = total_loss / num_batches
        avg_diffusion_loss = total_diffusion_loss / num_batches
        avg_overwrite_loss = total_overwrite_loss / num_batches
        avg_overwrite_rate = total_overwrite_rate / num_batches
        
        return {
            'loss': avg_loss,
            'diffusion_loss': avg_diffusion_loss,
            'overwrite_loss': avg_overwrite_loss,
            'overwrite_rate': avg_overwrite_rate
        }
    
    def train(self) -> Dict[str, List[float]]:
        """Main training loop with uncertainty-driven overwrite."""
        print(f"Starting uncertainty-driven ARDM training for {self.max_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Gumbel temperature: {self.gumbel_temperature}")
        print(f"Target overwrite rate: {self.target_overwrite_rate}")
        
        for epoch in range(self.max_epochs):
            self.current_epoch = epoch
            
            # Training
            train_metrics = self.train_epoch()
            
            # Validation
            val_metrics = self.validate_epoch()
            
            # Update learning rate
            self.scheduler.step()
            
            # Anneal Gumbel temperature
            if self.use_relaxed_bernoulli:
                self.gumbel_temperature *= self.gumbel_anneal_rate
                self.gumbel_temperature = max(self.gumbel_temperature, 0.1)
            
            # Log metrics
            self.training_history['train_loss'].append(train_metrics['loss'])
            self.training_history['diffusion_loss'].append(train_metrics['diffusion_loss'])
            self.training_history['overwrite_loss'].append(train_metrics['overwrite_loss'])
            self.training_history['sparsity_loss'].append(train_metrics['sparsity_loss'])
            self.training_history['stability_loss'].append(train_metrics['stability_loss'])
            self.training_history['overwrite_rate'].append(train_metrics['overwrite_rate'])
            self.training_history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            self.training_history['gumbel_temperature'].append(self.gumbel_temperature)
            
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
            print(f"Train Sparsity Loss: {train_metrics['sparsity_loss']:.4f}")
            print(f"Train Stability Loss: {train_metrics['stability_loss']:.4f}")
            print(f"Train Overwrite Rate: {train_metrics['overwrite_rate']:.3f}")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            print(f"Gumbel Temperature: {self.gumbel_temperature:.4f}")
            
            if val_metrics:
                print(f"Val Loss: {val_metrics['loss']:.4f}")
                print(f"Val Overwrite Rate: {val_metrics['overwrite_rate']:.3f}")
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
                    'train_sparsity_loss': train_metrics['sparsity_loss'],
                    'train_stability_loss': train_metrics['stability_loss'],
                    'train_overwrite_rate': train_metrics['overwrite_rate'],
                    'val_loss': val_metrics.get('loss', None),
                    'val_overwrite_rate': val_metrics.get('overwrite_rate', None),
                    'learning_rate': self.optimizer.param_groups[0]['lr'],
                    'gumbel_temperature': self.gumbel_temperature
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
            'gumbel_temperature': self.gumbel_temperature,
            'model_config': {
                'vocab_size': self.model.vocab_size,
                'max_seq_len': self.model.max_seq_len,
                'hidden_dim': self.model.hidden_dim,
                'num_layers': self.model.transformer.layers.__len__(),
                'num_heads': self.model.transformer.layers[0].self_attn.num_heads,
                'diffusion_steps': self.model.diffusion_steps,
                'use_mlp_gate': self.model.overwrite_gate.use_mlp,
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
        
        if 'gumbel_temperature' in checkpoint:
            self.gumbel_temperature = checkpoint['gumbel_temperature']
        
        print(f"Checkpoint loaded from {checkpoint_path}")
        print(f"Resuming from epoch {self.current_epoch + 1}")
    
    def generate_sample_text(
        self, 
        prompt: str = "", 
        max_length: int = 100,
        return_debug: bool = False
    ) -> str:
        """Generate sample text using the trained uncertainty-driven model."""
        self.model.eval()
        
        # Tokenize prompt if provided
        if prompt:
            # Simple character-level tokenization for demonstration
            prompt_tokens = [ord(c) % self.model.vocab_size for c in prompt]
            prompt_tokens = torch.tensor([prompt_tokens], device=self.device)
        else:
            prompt_tokens = None
        
        with torch.no_grad():
            # Generate tokens
            if return_debug:
                generated_tokens, debug_info = self.model.sample(
                    batch_size=1,
                    seq_len=max_length,
                    temperature=0.8,
                    top_k=50,
                    top_p=0.9,
                    return_debug=True
                )
                
                # Convert tokens back to text
                text = ''.join([chr(t.item() % 128) for t in generated_tokens[0]])
                
                if prompt:
                    text = prompt + text[len(prompt):]
                
                return text, debug_info
            else:
                generated_tokens = self.model.sample(
                    batch_size=1,
                    seq_len=max_length,
                    temperature=0.8,
                    top_k=50,
                    top_p=0.9
                )
                
                # Convert tokens back to text
                text = ''.join([chr(t.item() % 128) for t in generated_tokens[0]])
                
                if prompt:
                    text = prompt + text[len(prompt):]
                
                return text 