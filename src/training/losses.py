"""
Loss functions for Autoregressive Diffusion Models.

This module implements specialized loss functions for training ARDMs,
including diffusion loss and overwrite probability loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DiffusionLoss(nn.Module):
    """
    Loss function for diffusion-based text generation.
    
    Implements the standard diffusion loss that measures how well the model
    can predict the noise added to the input tokens.
    """
    
    def __init__(self, loss_type: str = "l2"):
        super().__init__()
        self.loss_type = loss_type
        
    def forward(
        self,
        predicted_logits: torch.Tensor,
        target_tokens: torch.Tensor,
        timesteps: torch.Tensor,
        alphas_cumprod: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate diffusion loss.
        
        Args:
            predicted_logits: Model predictions [batch_size, seq_len, vocab_size]
            target_tokens: Ground truth tokens [batch_size, seq_len]
            timesteps: Diffusion timesteps [batch_size]
            alphas_cumprod: Cumulative alpha values for noise schedule
            
        Returns:
            Diffusion loss scalar
        """
        batch_size, seq_len, vocab_size = predicted_logits.shape
        
        # Get current noise level
        alpha_cumprod_t = alphas_cumprod[timesteps].unsqueeze(-1).unsqueeze(-1)
        
        # Convert target tokens to one-hot
        target_one_hot = F.one_hot(target_tokens, num_classes=vocab_size).float()
        
        # Calculate predicted distribution
        predicted_probs = F.softmax(predicted_logits, dim=-1)
        
        if self.loss_type == "l2":
            # L2 loss between predicted and target distributions
            loss = F.mse_loss(predicted_probs, target_one_hot, reduction='none')
            loss = loss.mean(dim=-1)  # Average over vocabulary dimension
            
        elif self.loss_type == "kl":
            # KL divergence loss
            loss = F.kl_div(
                F.log_softmax(predicted_logits, dim=-1),
                target_one_hot,
                reduction='none'
            )
            loss = loss.sum(dim=-1)  # Sum over vocabulary dimension
            
        elif self.loss_type == "ce":
            # Cross-entropy loss
            loss = F.cross_entropy(
                predicted_logits.view(-1, vocab_size),
                target_tokens.view(-1),
                reduction='none'
            ).view(batch_size, seq_len)
            
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        # Apply noise schedule weighting
        # Higher noise levels (lower alpha_cumprod) get higher weight
        noise_weight = 1.0 / (alpha_cumprod_t.squeeze(-1) + 1e-8)
        weighted_loss = loss * noise_weight
        
        # Average over sequence length and batch
        return weighted_loss.mean()


class OverwriteLoss(nn.Module):
    """
    Loss function for overwrite probability learning.
    
    Encourages the model to learn appropriate overwrite probabilities
    based on token confidence and position.
    """
    
    def __init__(self, confidence_threshold: float = 0.8):
        super().__init__()
        self.confidence_threshold = confidence_threshold
        
    def forward(
        self,
        predicted_logits: torch.Tensor,
        target_tokens: torch.Tensor,
        timesteps: torch.Tensor,
        overwrite_probs: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate overwrite loss.
        
        Args:
            predicted_logits: Model predictions [batch_size, seq_len, vocab_size]
            target_tokens: Ground truth tokens [batch_size, seq_len]
            timesteps: Diffusion timesteps [batch_size]
            overwrite_probs: Current overwrite probabilities [diffusion_steps]
            
        Returns:
            Overwrite loss scalar
        """
        batch_size, seq_len, vocab_size = predicted_logits.shape
        
        # Get current overwrite probability
        current_overwrite_prob = overwrite_probs[timesteps].unsqueeze(-1)  # [batch_size, 1]
        
        # Calculate confidence for predicted tokens
        predicted_probs = F.softmax(predicted_logits, dim=-1)
        confidence = predicted_probs.max(dim=-1)[0]  # [batch_size, seq_len]
        
        # Calculate confidence for target tokens
        target_probs = predicted_probs.gather(-1, target_tokens.unsqueeze(-1)).squeeze(-1)
        
        # Ideal overwrite probability should be:
        # - Low when confidence is high (token is already good)
        # - High when confidence is low (token needs improvement)
        # - Higher for earlier positions (more context available later)
        
        # Position factor: earlier positions get higher overwrite probability
        position_factor = torch.linspace(1.0, 0.5, seq_len, device=predicted_logits.device).unsqueeze(0)
        
        # Confidence factor: lower confidence gets higher overwrite probability
        confidence_factor = 1.0 - confidence
        
        # Target overwrite probability
        target_overwrite_prob = position_factor * confidence_factor
        
        # Loss: encourage current overwrite probability to match target
        loss = F.mse_loss(
            current_overwrite_prob,
            target_overwrite_prob.mean(dim=-1, keepdim=True),
            reduction='none'
        ).squeeze(-1)
        
        # Additional regularization: encourage overwrite probability to be reasonable
        # Not too high (would cause too many changes) or too low (no refinement)
        regularization = torch.abs(current_overwrite_prob.squeeze(-1) - 0.5)
        
        # Combine main loss with regularization
        total_loss = loss + 0.1 * regularization
        
        return total_loss.mean()


class CombinedLoss(nn.Module):
    """
    Combined loss function that balances diffusion and overwrite losses.
    """
    
    def __init__(
        self,
        diffusion_weight: float = 1.0,
        overwrite_weight: float = 0.1,
        diffusion_loss_type: str = "l2"
    ):
        super().__init__()
        self.diffusion_loss = DiffusionLoss(loss_type=diffusion_loss_type)
        self.overwrite_loss = OverwriteLoss()
        self.diffusion_weight = diffusion_weight
        self.overwrite_weight = overwrite_weight
        
    def forward(
        self,
        predicted_logits: torch.Tensor,
        target_tokens: torch.Tensor,
        timesteps: torch.Tensor,
        alphas_cumprod: torch.Tensor,
        overwrite_probs: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate combined loss.
        
        Args:
            predicted_logits: Model predictions [batch_size, seq_len, vocab_size]
            target_tokens: Ground truth tokens [batch_size, seq_len]
            timesteps: Diffusion timesteps [batch_size]
            alphas_cumprod: Cumulative alpha values for noise schedule
            overwrite_probs: Current overwrite probabilities [diffusion_steps]
            
        Returns:
            Combined loss scalar
        """
        diffusion_loss = self.diffusion_loss(
            predicted_logits, target_tokens, timesteps, alphas_cumprod
        )
        
        overwrite_loss = self.overwrite_loss(
            predicted_logits, target_tokens, timesteps, overwrite_probs
        )
        
        total_loss = (
            self.diffusion_weight * diffusion_loss +
            self.overwrite_weight * overwrite_loss
        )
        
        return total_loss 