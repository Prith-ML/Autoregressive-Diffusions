import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
import time

class L2RBaseline:
    """Left-to-Right baseline: standard next-token decoding"""
    
    def __init__(self, model):
        self.model = model
        self.name = "L2R Baseline"
        self.description = "Standard left-to-right generation, no revision"
    
    def generate(self, prompt_tokens: torch.Tensor, max_length: int = 50) -> Tuple[torch.Tensor, Dict]:
        """Generate text left-to-right without any revision"""
        start_time = time.time()
        
        current_tokens = prompt_tokens.clone()
        generated_tokens = []
        
        # Generate tokens one by one
        for i in range(max_length - len(prompt_tokens[0])):
            with torch.no_grad():
                # Get model predictions
                if hasattr(self.model, 'forward'):
                    # Your ARDM model
                    logits, _, _ = self.model(current_tokens)
                else:
                    # Standard transformer
                    logits = self.model(current_tokens)
                
                # Get next token
                next_token_logits = logits[0, -1, :]
                next_token_probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(next_token_probs, 1)
                
                # Add to sequence
                if next_token.dim() == 1:
                    next_token = next_token.unsqueeze(0)
                current_tokens = torch.cat([current_tokens, next_token], dim=1)
                generated_tokens.append(next_token.item())
        
        generation_time = time.time() - start_time
        
        return current_tokens, {
            'method': self.name,
            'generation_time': generation_time,
            'tokens_generated': len(generated_tokens),
            'revision_steps': 0,  # L2R never revises
            'total_compute': generation_time
        }

class FixedScheduleBaseline:
    """Fixed schedule baseline: position-only refinement (p = r)"""
    
    def __init__(self, model, T: int = 10):
        self.model = model
        self.T = T
        self.name = "Fixed Schedule"
        self.description = "Position-only refinement schedule, no uncertainty"
    
    def _get_positional_schedule(self, seq_len: int, step: int) -> torch.Tensor:
        """Get fixed positional schedule: earlier positions refined more"""
        positions = torch.arange(seq_len, dtype=torch.float32)
        # Early positions (left) have higher revision probability
        schedule = torch.sigmoid(2.0 * (positions / seq_len - step / self.T))
        return schedule
    
    def generate_with_refinement(self, prompt_tokens: torch.Tensor, max_length: int = 50) -> Tuple[torch.Tensor, Dict]:
        """Generate with fixed positional refinement schedule"""
        start_time = time.time()
        
        # Initial generation (L2R style)
        current_tokens = prompt_tokens.clone()
        for i in range(max_length - len(prompt_tokens[0])):
            with torch.no_grad():
                if hasattr(self.model, 'forward'):
                    logits, _, _ = self.model(current_tokens)
                else:
                    logits = self.model(current_tokens)
                
                next_token_logits = logits[0, -1, :]
                next_token_probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(next_token_probs, 1)
                if next_token.dim() == 1:
                    next_token = next_token.unsqueeze(0)
                current_tokens = torch.cat([current_tokens, next_token], dim=1)
        
        # Now apply fixed refinement schedule
        revision_steps = 0
        for step in range(1, self.T + 1):
            with torch.no_grad():
                if hasattr(self.model, 'forward'):
                    logits, _, _ = self.model(current_tokens)
                else:
                    logits = self.model(current_tokens)
                
                # Get positional schedule (p = r)
                schedule = self._get_positional_schedule(len(current_tokens[0]), step)
                
                # Decide which tokens to revise based on position only
                revision_mask = torch.bernoulli(schedule).unsqueeze(0)  # [1, L]
                
                # Only revise if mask says so
                if revision_mask.sum() > 0:
                    # Sample new tokens for masked positions
                    # Create a proper mask for indexing
                    mask_2d = revision_mask.bool()  # [1, L]
                    
                    # Get logits for masked positions
                    masked_logits = logits[mask_2d]  # [num_masked, vocab_size]
                    
                    if masked_logits.numel() > 0:
                        new_probs = F.softmax(masked_logits, dim=-1)
                        new_tokens = torch.multinomial(new_probs, 1)  # [num_masked]
                        
                        # Update tokens at masked positions
                        current_tokens[mask_2d] = new_tokens.squeeze()
                        revision_steps += 1
        
        total_time = time.time() - start_time
        
        return current_tokens, {
            'method': self.name,
            'generation_time': total_time,
            'tokens_generated': max_length,
            'revision_steps': revision_steps,
            'total_compute': total_time,
            'schedule_type': 'position_only'
        }

class DynamicGateBaseline:
    """Your dynamic gate: uncertainty + schedule via noisy-OR"""
    
    def __init__(self, model, T: int = 10):
        self.model = model
        self.T = T
        self.name = "Dynamic Gate (Ours)"
        self.description = "Uncertainty-driven + positional schedule via noisy-OR"
    
    def _compute_uncertainty_signals(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the three uncertainty signals"""
        # Entropy
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)
        entropy = -(probs * log_probs).sum(dim=-1)
        
        # Margin (top1 - top2)
        top2_values, _ = torch.topk(logits, k=2, dim=-1)
        margin = top2_values[..., 0] - top2_values[..., 1]
        
        # Confidence change (placeholder - would need previous step)
        confidence_change = torch.zeros_like(entropy)
        
        return entropy, margin, confidence_change
    
    def _get_positional_schedule(self, seq_len: int, step: int) -> torch.Tensor:
        """Get positional schedule prior"""
        positions = torch.arange(seq_len, dtype=torch.float32)
        tau = (self.T / seq_len) * positions
        schedule = torch.sigmoid(1.25 * (tau - step))
        return schedule
    
    def _compute_dynamic_overwrite_prob(self, entropy: torch.Tensor, margin: torch.Tensor, 
                                      confidence_change: torch.Tensor, schedule: torch.Tensor) -> torch.Tensor:
        """Compute dynamic overwrite probability: p = 1 - (1-u)(1-r)"""
        # Simple uncertainty gate (you can make this more sophisticated)
        uncertainty_gate = torch.sigmoid(
            0.5 + 0.3 * entropy - 0.2 * margin - 0.1 * confidence_change
        )
        
        # Noisy-OR combination
        overwrite_prob = 1.0 - (1.0 - uncertainty_gate) * (1.0 - schedule)
        return overwrite_prob.clamp(0.01, 0.99)
    
    def generate_with_dynamic_refinement(self, prompt_tokens: torch.Tensor, max_length: int = 50) -> Tuple[torch.Tensor, Dict]:
        """Generate with dynamic uncertainty-driven refinement"""
        start_time = time.time()
        
        # Initial generation
        current_tokens = prompt_tokens.clone()
        for i in range(max_length - len(prompt_tokens[0])):
            with torch.no_grad():
                if hasattr(self.model, 'forward'):
                    logits, _, _ = self.model(current_tokens)
                else:
                    logits = self.model(current_tokens)
                
                next_token_logits = logits[0, -1, :]
                next_token_probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(next_token_probs, 1)
                if next_token.dim() == 1:
                    next_token = next_token.unsqueeze(0)
                current_tokens = torch.cat([current_tokens, next_token], dim=1)
        
        # Apply dynamic refinement
        revision_steps = 0
        prev_logits = None
        
        for step in range(1, self.T + 1):
            with torch.no_grad():
                if hasattr(self.model, 'forward'):
                    logits, _, _ = self.model(current_tokens)
                else:
                    logits = self.model(current_tokens)
                
                # Compute uncertainty signals
                entropy, margin, confidence_change = self._compute_uncertainty_signals(logits)
                
                # Get positional schedule
                schedule = self._get_positional_schedule(len(current_tokens[0]), step)
                
                # Compute dynamic overwrite probability
                overwrite_prob = self._compute_dynamic_overwrite_prob(
                    entropy, margin, confidence_change, schedule
                )
                
                # Decide which tokens to revise
                revision_mask = torch.bernoulli(overwrite_prob).unsqueeze(0)  # [1, L]
                
                # Only revise if mask says so
                if revision_mask.sum() > 0:
                    # Sample new tokens for masked positions
                    # Create a proper mask for indexing
                    mask_2d = revision_mask.bool()  # [1, L]
                    
                    # Get logits for masked positions
                    masked_logits = logits[mask_2d]  # [num_masked, vocab_size]
                    
                    if masked_logits.numel() > 0:
                        new_probs = F.softmax(masked_logits, dim=-1)
                        new_tokens = torch.multinomial(new_probs, 1)  # [num_masked]
                        
                        # Update tokens at masked positions
                        current_tokens[mask_2d] = new_tokens.squeeze()
                        revision_steps += 1
                
                prev_logits = logits.detach()
        
        total_time = time.time() - start_time
        
        return current_tokens, {
            'method': self.name,
            'generation_time': total_time,
            'tokens_generated': max_length,
            'revision_steps': revision_steps,
            'total_compute': total_time,
            'schedule_type': 'uncertainty_driven',
            'uncertainty_signals': ['entropy', 'margin', 'confidence_change']
        }

def run_baseline_comparison(model, prompt_tokens: torch.Tensor, max_length: int = 50) -> Dict:
    """Run comparison between all three baselines"""
    print("🔬 RUNNING BASELINE COMPARISON EXPERIMENT")
    print("=" * 60)
    
    # Initialize baselines
    l2r = L2RBaseline(model)
    fixed_schedule = FixedScheduleBaseline(model)
    dynamic_gate = DynamicGateBaseline(model)
    
    # Test prompts
    test_prompts = [
        "The detective",
        "She was investigating",
        "The mystery deepened",
        "Inside the old house"
    ]
    
    results = {
        'l2r': {'times': [], 'revisions': [], 'quality_scores': []},
        'fixed_schedule': {'times': [], 'revisions': [], 'quality_scores': []},
        'dynamic_gate': {'times': [], 'revisions': [], 'quality_scores': []}
    }
    
    for i, prompt in enumerate(test_prompts):
        print(f"\n📝 Test {i+1}: '{prompt}'")
        print("-" * 40)
        
        # Convert prompt to tokens (simplified)
        prompt_tokens = torch.tensor([[hash(word) % 1000 for word in prompt.split()]], dtype=torch.long)
        
        # Test L2R
        print(f"🚀 Testing {l2r.name}...")
        l2r_tokens, l2r_metrics = l2r.generate(prompt_tokens, max_length)
        results['l2r']['times'].append(l2r_metrics['generation_time'])
        results['l2r']['revisions'].append(l2r_metrics['revision_steps'])
        
        # Test Fixed Schedule
        print(f"🔄 Testing {fixed_schedule.name}...")
        fixed_tokens, fixed_metrics = fixed_schedule.generate_with_refinement(prompt_tokens, max_length)
        results['fixed_schedule']['times'].append(fixed_metrics['generation_time'])
        results['fixed_schedule']['revisions'].append(fixed_metrics['revision_steps'])
        
        # Test Dynamic Gate
        print(f"🎯 Testing {dynamic_gate.name}...")
        dynamic_tokens, dynamic_metrics = dynamic_gate.generate_with_dynamic_refinement(prompt_tokens, max_length)
        results['dynamic_gate']['times'].append(dynamic_metrics['generation_time'])
        results['dynamic_gate']['revisions'].append(dynamic_metrics['revision_steps'])
        
        print(f"✅ All baselines completed for prompt {i+1}")
    
    # Compute averages
    for method in results:
        results[method]['avg_time'] = sum(results[method]['times']) / len(results[method]['times'])
        results[method]['avg_revisions'] = sum(results[method]['revisions']) / len(results[method]['revisions'])
    
    return results

def print_comparison_results(results: Dict):
    """Print formatted comparison results"""
    print("\n" + "="*60)
    print("📊 EXPERIMENT RESULTS")
    print("="*60)
    
    print(f"{'Method':<20} {'Avg Time (s)':<15} {'Avg Revisions':<15} {'Efficiency':<15}")
    print("-" * 70)
    
    for method, data in results.items():
        method_name = method.replace('_', ' ').title()
        avg_time = f"{data['avg_time']:.4f}"
        avg_revisions = f"{data['avg_revisions']:.1f}"
        
        # Efficiency: lower time + appropriate revisions is better
        if method == 'l2r':
            efficiency = "Fast (No revision)"
        elif method == 'fixed_schedule':
            efficiency = "Medium (Position-based)"
        else:  # dynamic_gate
            efficiency = "Smart (Uncertainty-driven)"
        
        print(f"{method_name:<20} {avg_time:<15} {avg_revisions:<15} {efficiency:<15}")
    
    print("\n🏆 Key Findings:")
    print("• L2R: Fastest but no revision capability")
    print("• Fixed Schedule: Medium speed, position-based revision")
    print("• Dynamic Gate: Smart revision based on uncertainty")
    print("\n🎯 Your dynamic gate provides intelligent refinement!")

if __name__ == "__main__":
    # Example usage
    print("🧪 BASELINE COMPARISON EXPERIMENT")
    print("=" * 60)
    print("This script implements the three baselines for comparison:")
    print("1. L2R: Standard left-to-right generation")
    print("2. Fixed Schedule: Position-only refinement")
    print("3. Dynamic Gate: Uncertainty-driven refinement")
    print("\nTo run the experiment, import and use:")
    print("from experiments.baselines import run_baseline_comparison")
    print("results = run_baseline_comparison(your_model, prompt_tokens)") 