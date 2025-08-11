#!/usr/bin/env python3
"""
Demo script for Uncertainty-driven Autoregressive Diffusion Models.

This script demonstrates the research framework for dynamic overwrite probabilities
based on uncertainty signals: entropy, margin, and confidence change.
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Tuple
import argparse

from models.uncertainty_gate import (
    UncertaintyARDM, 
    OverwriteGate, 
    SchedulePrior,
    entropy_from_logits,
    margin_from_logits
)
from training.uncertainty_trainer import UncertaintyARDMTrainer


def create_synthetic_data(
    vocab_size: int = 64,
    seq_len: int = 16,
    batch_size: int = 8,
    num_samples: int = 100
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create synthetic training data for demonstration.
    
    Args:
        vocab_size: Size of vocabulary
        seq_len: Length of sequences
        batch_size: Batch size for training
        num_samples: Number of training samples
        
    Returns:
        input_ids: [num_samples, seq_len] token sequences
        attention_mask: [num_samples, seq_len] attention masks
    """
    # Generate random token sequences
    input_ids = torch.randint(0, vocab_size, (num_samples, seq_len))
    
    # Create attention masks (all valid tokens for simplicity)
    attention_mask = torch.ones(num_samples, seq_len)
    
    return input_ids, attention_mask


class SyntheticDataset(torch.utils.data.Dataset):
    """Synthetic dataset for demonstration."""
    
    def __init__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx]
        }


def analyze_overwrite_patterns(
    model: UncertaintyARDM,
    seq_len: int = 16,
    diffusion_steps: int = 100,
    batch_size: int = 1
) -> Dict[str, np.ndarray]:
    """
    Analyze overwrite patterns across diffusion steps.
    
    Args:
        model: Trained uncertainty-driven ARDM
        seq_len: Length of sequences to analyze
        diffusion_steps: Number of diffusion steps
        batch_size: Batch size for analysis
        
    Returns:
        Dictionary containing analysis results
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Initialize with random tokens
    tokens = torch.randint(0, model.vocab_size, (batch_size, seq_len), device=device)
    
    # Store analysis data
    overwrite_probs_history = []
    entropy_history = []
    margin_history = []
    confidence_change_history = []
    schedule_prior_history = []
    
    # Reset gate state
    model.overwrite_gate.reset_state()
    
    # Iterate through diffusion steps
    for t in range(diffusion_steps - 1, -1, -1):
        timesteps = torch.full((batch_size,), t, device=device, dtype=torch.long)
        
        with torch.no_grad():
            # Get model predictions
            logits, hidden_states = model(tokens, timesteps)
            
            # Compute overwrite probabilities
            overwrite_probs, debug_info = model.overwrite_gate(
                h=hidden_states,
                logits=logits,
                step_t=t
            )
            
            # Store analysis data
            overwrite_probs_history.append(overwrite_probs.cpu().numpy())
            entropy_history.append(debug_info['entropy'].cpu().numpy())
            margin_history.append(debug_info['margin'].cpu().numpy())
            confidence_change_history.append(debug_info['confidence_change'].cpu().numpy())
            schedule_prior_history.append(debug_info['schedule_prior'].cpu().numpy())
            
            # Sample overwrite mask
            overwrite_mask = torch.bernoulli(overwrite_probs)
            
            # Sample new tokens for overwrite positions
            new_tokens = model._sample_tokens(logits, temperature=0.8)
            
            # Apply overwrites
            tokens = torch.where(overwrite_mask.bool(), new_tokens, tokens)
    
    # Convert to numpy arrays
    analysis_results = {
        'overwrite_probs': np.array(overwrite_probs_history),  # [steps, batch, seq]
        'entropy': np.array(entropy_history),                  # [steps, batch, seq]
        'margin': np.array(margin_history),                    # [steps, batch, seq]
        'confidence_change': np.array(confidence_change_history),  # [steps, batch, seq]
        'schedule_prior': np.array(schedule_prior_history),    # [steps, batch, seq]
        'diffusion_steps': diffusion_steps,
        'seq_len': seq_len
    }
    
    return analysis_results


def plot_overwrite_heatmap(
    analysis_results: Dict[str, np.ndarray],
    save_path: str = "overwrite_heatmap.png"
):
    """
    Plot overwrite probability heatmap across diffusion steps and positions.
    
    Args:
        analysis_results: Results from analyze_overwrite_patterns
        save_path: Path to save the plot
    """
    overwrite_probs = analysis_results['overwrite_probs']
    diffusion_steps = analysis_results['diffusion_steps']
    seq_len = analysis_results['seq_len']
    
    # Average over batch dimension
    avg_overwrite_probs = overwrite_probs.mean(axis=1)  # [steps, seq]
    
    # Create the heatmap
    plt.figure(figsize=(12, 8))
    
    # Plot overwrite probabilities
    plt.subplot(2, 2, 1)
    sns.heatmap(
        avg_overwrite_probs.T,  # Transpose to get positions on y-axis
        cmap='viridis',
        xticklabels=range(diffusion_steps - 1, -1, -1),
        yticklabels=range(seq_len),
        cbar_kws={'label': 'Overwrite Probability'}
    )
    plt.title('Overwrite Probability Heatmap')
    plt.xlabel('Diffusion Step (t)')
    plt.ylabel('Position (i)')
    
    # Plot entropy
    plt.subplot(2, 2, 2)
    entropy = analysis_results['entropy'].mean(axis=1)
    sns.heatmap(
        entropy.T,
        cmap='plasma',
        xticklabels=range(diffusion_steps - 1, -1, -1),
        yticklabels=range(seq_len),
        cbar_kws={'label': 'Entropy'}
    )
    plt.title('Entropy Heatmap')
    plt.xlabel('Diffusion Step (t)')
    plt.ylabel('Position (i)')
    
    # Plot margin
    plt.subplot(2, 2, 3)
    margin = analysis_results['margin'].mean(axis=1)
    sns.heatmap(
        margin.T,
        cmap='RdBu_r',
        xticklabels=range(diffusion_steps - 1, -1, -1),
        yticklabels=range(seq_len),
        cbar_kws={'label': 'Top-1/Top-2 Margin'}
    )
    plt.title('Margin Heatmap')
    plt.xlabel('Diffusion Step (t)')
    plt.ylabel('Position (i)')
    
    # Plot schedule prior
    plt.subplot(2, 2, 4)
    schedule_prior = analysis_results['schedule_prior'].mean(axis=1)
    sns.heatmap(
        schedule_prior.T,
        cmap='coolwarm',
        xticklabels=range(diffusion_steps - 1, -1, -1),
        yticklabels=range(seq_len),
        cbar_kws={'label': 'Schedule Prior'}
    )
    plt.title('Schedule Prior Heatmap')
    plt.xlabel('Diffusion Step (t)')
    plt.ylabel('Position (i)')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Overwrite heatmap saved to {save_path}")


def plot_overwrite_statistics(
    analysis_results: Dict[str, np.ndarray],
    save_path: str = "overwrite_statistics.png"
):
    """
    Plot statistical analysis of overwrite patterns.
    
    Args:
        analysis_results: Results from analyze_overwrite_patterns
        save_path: Path to save the plot
    """
    overwrite_probs = analysis_results['overwrite_probs']
    diffusion_steps = analysis_results['diffusion_steps']
    seq_len = analysis_results['seq_len']
    
    # Average over batch dimension
    avg_overwrite_probs = overwrite_probs.mean(axis=1)  # [steps, seq]
    
    plt.figure(figsize=(15, 10))
    
    # 1. Average overwrite rate per step
    plt.subplot(2, 3, 1)
    step_overwrite_rate = avg_overwrite_probs.mean(axis=1)
    plt.plot(range(diffusion_steps - 1, -1, -1), step_overwrite_rate, 'b-', linewidth=2)
    plt.xlabel('Diffusion Step (t)')
    plt.ylabel('Average Overwrite Rate')
    plt.title('Overwrite Rate vs Diffusion Step')
    plt.grid(True, alpha=0.3)
    
    # 2. Average overwrite rate per position
    plt.subplot(2, 3, 2)
    pos_overwrite_rate = avg_overwrite_probs.mean(axis=0)
    plt.bar(range(seq_len), pos_overwrite_rate, alpha=0.7, color='green')
    plt.xlabel('Position (i)')
    plt.ylabel('Average Overwrite Rate')
    plt.title('Overwrite Rate vs Position')
    plt.grid(True, alpha=0.3)
    
    # 3. Overwrite rate distribution
    plt.subplot(2, 3, 3)
    plt.hist(overwrite_probs.flatten(), bins=50, alpha=0.7, color='orange', edgecolor='black')
    plt.xlabel('Overwrite Probability')
    plt.ylabel('Frequency')
    plt.title('Distribution of Overwrite Probabilities')
    plt.grid(True, alpha=0.3)
    
    # 4. Position vs Step analysis
    plt.subplot(2, 3, 4)
    # Show how early positions mature faster
    early_pos = avg_overwrite_probs[:, 0]  # First position
    mid_pos = avg_overwrite_probs[:, seq_len // 2]  # Middle position
    late_pos = avg_overwrite_probs[:, -1]  # Last position
    
    steps = range(diffusion_steps - 1, -1, -1)
    plt.plot(steps, early_pos, 'r-', linewidth=2, label='Early Position (i=0)')
    plt.plot(steps, mid_pos, 'g-', linewidth=2, label=f'Middle Position (i={seq_len//2})')
    plt.plot(steps, late_pos, 'b-', linewidth=2, label='Late Position (i={seq_len-1})')
    plt.xlabel('Diffusion Step (t)')
    plt.ylabel('Overwrite Probability')
    plt.title('Position Maturity Analysis')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. Uncertainty correlation
    plt.subplot(2, 3, 5)
    entropy = analysis_results['entropy'].mean(axis=1).flatten()
    margin = analysis_results['margin'].mean(axis=1).flatten()
    overwrite_probs_flat = overwrite_probs.mean(axis=1).flatten()
    
    # Scatter plot: entropy vs overwrite probability
    plt.scatter(entropy, overwrite_probs_flat, alpha=0.6, color='purple')
    plt.xlabel('Entropy')
    plt.ylabel('Overwrite Probability')
    plt.title('Entropy vs Overwrite Probability')
    plt.grid(True, alpha=0.3)
    
    # 6. Margin correlation
    plt.subplot(2, 3, 6)
    plt.scatter(margin, overwrite_probs_flat, alpha=0.6, color='red')
    plt.xlabel('Top-1/Top-2 Margin')
    plt.ylabel('Overwrite Probability')
    plt.title('Margin vs Overwrite Probability')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Overwrite statistics saved to {save_path}")


def run_training_demo(
    vocab_size: int = 64,
    seq_len: int = 16,
    hidden_dim: int = 128,
    num_layers: int = 4,
    diffusion_steps: int = 50,
    batch_size: int = 8,
    num_epochs: int = 10,
    use_mlp_gate: bool = True
):
    """
    Run a complete training demo for the uncertainty-driven ARDM.
    
    Args:
        vocab_size: Size of vocabulary
        seq_len: Length of sequences
        hidden_dim: Hidden dimension of transformer
        num_layers: Number of transformer layers
        diffusion_steps: Number of diffusion steps
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        use_mlp_gate: Whether to use MLP-based gate
    """
    print("=" * 60)
    print("UNCERTAINTY-DRIVEN ARDM TRAINING DEMO")
    print("=" * 60)
    
    # Create synthetic data
    print("\n1. Creating synthetic training data...")
    input_ids, attention_mask = create_synthetic_data(
        vocab_size=vocab_size,
        seq_len=seq_len,
        batch_size=batch_size,
        num_samples=1000
    )
    
    # Create dataset and dataloader
    dataset = SyntheticDataset(input_ids, attention_mask)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    
    # Create model
    print("\n2. Creating uncertainty-driven ARDM...")
    model = UncertaintyARDM(
        vocab_size=vocab_size,
        max_seq_len=seq_len,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        diffusion_steps=diffusion_steps,
        use_mlp_gate=use_mlp_gate,
        gate_mlp_width=128
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Gate type: {'MLP' if use_mlp_gate else 'Linear'}")
    
    # Create trainer
    print("\n3. Creating trainer...")
    trainer = UncertaintyARDMTrainer(
        model=model,
        train_dataloader=dataloader,
        val_dataloader=None,  # No validation for demo
        learning_rate=1e-3,
        max_epochs=num_epochs,
        use_relaxed_bernoulli=True,
        gumbel_temperature=1.0,
        gumbel_anneal_rate=0.95,
        target_overwrite_rate=0.3,
        use_wandb=False
    )
    
    # Train the model
    print("\n4. Starting training...")
    training_history = trainer.train()
    
    print("\n5. Training completed! Analyzing overwrite patterns...")
    
    # Analyze overwrite patterns
    analysis_results = analyze_overwrite_patterns(
        model=model,
        seq_len=seq_len,
        diffusion_steps=diffusion_steps,
        batch_size=1
    )
    
    # Plot results
    print("\n6. Generating visualizations...")
    plot_overwrite_heatmap(analysis_results, "demo_overwrite_heatmap.png")
    plot_overwrite_statistics(analysis_results, "demo_overwrite_statistics.png")
    
    print("\n7. Demo completed! Check the generated plots.")
    
    return model, trainer, analysis_results


def run_sampling_demo(
    model: UncertaintyARDM,
    prompt: str = "Hello world",
    max_length: int = 50,
    temperature: float = 0.8,
    return_debug: bool = True
):
    """
    Run a sampling demo to show text generation with uncertainty-driven overwrite.
    
    Args:
        model: Trained uncertainty-driven ARDM
        prompt: Text prompt to start generation
        max_length: Maximum length of generated text
        temperature: Sampling temperature
        return_debug: Whether to return debug information
    """
    print("\n" + "=" * 60)
    print("UNCERTAINTY-DRIVEN ARDM SAMPLING DEMO")
    print("=" * 60)
    
    print(f"Prompt: '{prompt}'")
    print(f"Max length: {max_length}")
    print(f"Temperature: {temperature}")
    
    # Generate text
    if return_debug:
        generated_text, debug_info = model.generate_sample_text(
            prompt=prompt,
            max_length=max_length,
            return_debug=True
        )
        
        print(f"\nGenerated text: {generated_text}")
        
        # Analyze debug information
        print(f"\nDebug information available for {len(debug_info)} diffusion steps")
        print("Each step contains: entropy, margin, confidence_change, schedule_prior, uncertainty_gate, final_prob")
        
        return generated_text, debug_info
    else:
        generated_text = model.generate_sample_text(
            prompt=prompt,
            max_length=max_length,
            return_debug=False
        )
        
        print(f"\nGenerated text: {generated_text}")
        return generated_text


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description="Uncertainty-driven ARDM Demo")
    parser.add_argument("--mode", choices=["train", "sample", "both"], default="both",
                       help="Demo mode: train, sample, or both")
    parser.add_argument("--vocab_size", type=int, default=64, help="Vocabulary size")
    parser.add_argument("--seq_len", type=int, default=16, help="Sequence length")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--diffusion_steps", type=int, default=50, help="Number of diffusion steps")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--use_mlp_gate", action="store_true", help="Use MLP-based gate")
    parser.add_argument("--prompt", type=str, default="Hello world", help="Text prompt for sampling")
    parser.add_argument("--max_length", type=int, default=50, help="Maximum generation length")
    
    args = parser.parse_args()
    
    print("Uncertainty-driven Autoregressive Diffusion Model Demo")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Vocabulary size: {args.vocab_size}")
    print(f"Sequence length: {args.seq_len}")
    print(f"Hidden dimension: {args.hidden_dim}")
    print(f"Number of layers: {args.num_layers}")
    print(f"Diffusion steps: {args.diffusion_steps}")
    print(f"Batch size: {args.batch_size}")
    print(f"Number of epochs: {args.num_epochs}")
    print(f"Gate type: {'MLP' if args.use_mlp_gate else 'Linear'}")
    print("=" * 60)
    
    if args.mode in ["train", "both"]:
        # Run training demo
        model, trainer, analysis_results = run_training_demo(
            vocab_size=args.vocab_size,
            seq_len=args.seq_len,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            diffusion_steps=args.diffusion_steps,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            use_mlp_gate=args.use_mlp_gate
        )
    else:
        # Load a pre-trained model (you would need to implement this)
        print("Training mode not selected. Please provide a pre-trained model for sampling.")
        return
    
    if args.mode in ["sample", "both"]:
        # Run sampling demo
        run_sampling_demo(
            model=model,
            prompt=args.prompt,
            max_length=args.max_length,
            temperature=0.8,
            return_debug=True
        )
    
    print("\nDemo completed successfully!")


if __name__ == "__main__":
    main() 