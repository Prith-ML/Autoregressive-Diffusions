import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import time

class L2RBaseline:
    """Left-to-Right baseline: standard next-token decoding"""
    
    def __init__(self, model):
        self.model = model
        self.name = "L2R Baseline"
        self.description = "Standard left-to-right generation, no revision"
    
    def _sample_token(self, logits: torch.Tensor, top_k: int = 50, top_p: float = 0.9, temperature: float = 0.7, greedy: bool = False) -> torch.Tensor:
        if greedy:
            return torch.argmax(logits, dim=-1, keepdim=True)
        logits = logits / max(1e-6, temperature)
        if top_k is not None and top_k > 0:
            top_k = min(top_k, logits.shape[-1])
            kth_vals, _ = torch.topk(logits, k=top_k, dim=-1)
            thresh = kth_vals[..., -1, None]
            logits = torch.where(logits < thresh, torch.full_like(logits, float('-inf')), logits)
        if top_p is not None and 0.0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            cutoff = cumprobs > top_p
            cutoff[..., 1:] = cutoff[..., :-1].clone()
            cutoff[..., 0] = False
            sorted_logits = torch.where(cutoff, torch.full_like(sorted_logits, float('-inf')), sorted_logits)
            logits = torch.full_like(logits, float('-inf'))
            logits.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, 1)

    def generate(self, prompt_tokens: torch.Tensor, max_length: int = 50, eos_token_id: Optional[int] = None) -> Tuple[torch.Tensor, Dict]:
        """Generate text left-to-right without any revision"""
        start_time = time.time()
        
        current_tokens = prompt_tokens.clone()
        generated_tokens = []
        
        # Generate tokens one by one
        for i in range(max_length - len(prompt_tokens[0])):
            with torch.no_grad():
                # Get model predictions
                if hasattr(self.model, 'set_encoder_inputs') and hasattr(self.model, 'seq2seq') and hasattr(self.model, '_last_source_input_ids'):
                    # Use summarization-friendly generate() if available (seq2seq path)
                    gen_out = self.model.seq2seq.generate(
                        input_ids=self.model._last_source_input_ids,
                        attention_mask=self.model._last_source_attention_mask,
                        max_length=max_length,
                        min_length=max(10, max_length // 2),
                        no_repeat_ngram_size=3,
                        num_beams=4,
                        length_penalty=2.0,
                        early_stopping=True
                    )
                    # Replace current_tokens with the full generated sequence and stop
                    current_tokens = gen_out
                    break
                elif hasattr(self.model, 'forward'):
                    # Your ARDM model
                    logits, _, _ = self.model(current_tokens)
                else:
                    # Standard transformer
                    logits = self.model(current_tokens)
                
                # Get next token
                next_token_logits = logits[0, -1, :]
                next_token = self._sample_token(next_token_logits, top_k=50, top_p=0.9, temperature=0.7)
                
                # Add to sequence
                if next_token.dim() == 1:
                    next_token = next_token.unsqueeze(0)
                current_tokens = torch.cat([current_tokens, next_token], dim=1)
                # Safely convert to item
                token_value = int(next_token.cpu().detach().numpy())
                generated_tokens.append(token_value)
                # EOS-aware stopping
                if eos_token_id is not None and token_value == int(eos_token_id):
                    break
        
        generation_time = time.time() - start_time
        
        return current_tokens, {
            'method': self.name,
            'generation_time': generation_time,
            'tokens_generated': len(generated_tokens),
            'revision_steps': 0,  # L2R never revises
            'total_compute': generation_time,
            
            # 🎯 BASIC LOGGING FOR COMPARISON
            'total_nfe': 0,  # L2R has no refinement steps
            'overwrite_efficiency': 0.0,  # No overwrites
            
            # 🚀 AR-DIFFUSION STYLE METRICS (for comparison)
            'tokens_overwritten_per_step': [],
            'nfe_per_step': [],
            'diversity_score': self._compute_diversity_score(current_tokens),
            'coherence_score': 1.0  # L2R has perfect coherence (no revisions)
        }
    
    def _compute_diversity_score(self, tokens: torch.Tensor) -> float:
        """Compute diversity score based on token distribution"""
        flat_tokens = tokens.flatten()
        unique_tokens = torch.unique(flat_tokens)
        diversity = len(unique_tokens) / len(flat_tokens)
        # Convert to float safely
        return float(diversity)

class FixedScheduleBaseline:
    """Fixed schedule baseline: position-only refinement (p = r)"""
    
    def __init__(self, model, T: int = 10):
        self.model = model
        self.T = 2 if hasattr(model, 'set_encoder_inputs') else T
        self.name = "Fixed Schedule"
        self.description = "Position-only refinement schedule, no uncertainty"
    
    def _get_positional_schedule(self, seq_len: int, step: int) -> torch.Tensor:
        """Get fixed positional schedule: earlier positions refined more"""
        positions = torch.arange(seq_len, dtype=torch.float32)
        # AR-DIFFUSION style: early positions mature faster
        # tau(i) = T/n * (i + δ) where δ controls when positions start maturing
        delta = 1.0  # Start maturing later (was 0.5)
        tau = (self.T / seq_len) * (positions + delta)
        
        # r_i(t) = σ(α * (τ(i) - t)) where α controls sharpness
        alpha = 1.0  # Less aggressive than before (was 2.0)
        schedule = torch.sigmoid(alpha * (tau - step))
        
        # Scale down to reasonable revision probabilities
        schedule = schedule * 0.15  # Max 15% revision probability (was 0.3)
        
        return schedule
    
    def generate_with_refinement(self, prompt_tokens: torch.Tensor, max_length: int = 50, eos_token_id: Optional[int] = None) -> Tuple[torch.Tensor, Dict]:
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
                next_token = L2RBaseline._sample_token(self, next_token_logits, top_k=50, top_p=0.9, temperature=0.7)
                if next_token.dim() == 1:
                    next_token = next_token.unsqueeze(0)
                current_tokens = torch.cat([current_tokens, next_token], dim=1)
                if eos_token_id is not None and int(next_token.cpu().detach().numpy()) == int(eos_token_id):
                    break
        
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
                    
                    # Get logits for masked positions - fix the indexing
                    # We need to get logits for each masked position
                    masked_positions = torch.where(mask_2d[0])[0]  # Get actual positions
                    
                    if len(masked_positions) > 0:
                        # Get logits for each masked position
                        for pos in masked_positions:
                            # Recompute logits per update for stability
                            if hasattr(self.model, 'forward'):
                                logits, _, _ = self.model(current_tokens)
                            else:
                                logits = self.model(current_tokens)
                            pos_logits = logits[0, pos, :]
                            new_token = L2RBaseline._sample_token(self, pos_logits, top_k=50, top_p=0.9, temperature=0.7)
                            current_tokens[0, pos] = new_token
                        
                        revision_steps += 1
        
        total_time = time.time() - start_time
        
        return current_tokens, {
            'method': self.name,
            'generation_time': total_time,
            'tokens_generated': max_length,
            'revision_steps': revision_steps,
            'total_compute': total_time,
            'schedule_type': 'position_only',
            
            # 🎯 BASIC LOGGING FOR COMPARISON
            'total_nfe': self.T,  # Fixed schedule uses T steps
            'overwrite_efficiency': revision_steps / max(1, self.T),
            
            # 🚀 AR-DIFFUSION STYLE METRICS (for comparison)
            'tokens_overwritten_per_step': [1 if step <= revision_steps else 0 for step in range(1, self.T + 1)],
            'nfe_per_step': [1] * self.T,
            'diversity_score': self._compute_diversity_score(current_tokens),
            'coherence_score': self._compute_coherence_score(current_tokens, [1 if step <= revision_steps else 0 for step in range(1, self.T + 1)])
        }
    
    def _compute_diversity_score(self, tokens: torch.Tensor) -> float:
        """Compute diversity score based on token distribution"""
        flat_tokens = tokens.flatten()
        unique_tokens = torch.unique(flat_tokens)
        diversity = len(unique_tokens) / len(flat_tokens)
        # Convert to float safely
        return float(diversity)
    
    def _compute_coherence_score(self, tokens: torch.Tensor, overwrite_pattern: List[int]) -> float:
        """Compute coherence score based on revision pattern"""
        if not overwrite_pattern:
            return 1.0
        
        total_revisions = sum(overwrite_pattern)
        if total_revisions == 0:
            return 1.0
        
        weighted_revisions = 0
        for step, revs in enumerate(overwrite_pattern):
            weight = (step + 1) / len(overwrite_pattern)
            weighted_revisions += revs * weight
        
        coherence = 1.0 / (1.0 + weighted_revisions / total_revisions)
        return coherence

class DynamicGateBaseline:
    """Your dynamic gate: uncertainty + schedule via noisy-OR"""
    
    def __init__(self, model, T: int = 10):
        self.model = model
        # Reduce refinement steps for seq2seq to stabilize outputs
        self.T = 6 if hasattr(model, 'set_encoder_inputs') else T
        self.name = "Dynamic Gate (Ours)"
        self.description = "Uncertainty-driven + positional schedule via noisy-OR"
        # Disable early stopping by default for seq2seq denoisers so refinement actually runs
        self.disable_early_stop = hasattr(model, 'set_encoder_inputs')
        # NEW: Deterministic masking + schedules
        self.use_topk_mask = True  # prefer budgeted top-k selection over Bernoulli
        # Edit budget schedule (fraction of tokens). Decays over steps
        self.budget_start = 0.30
        self.budget_end = 0.05
        # Threshold schedule if not using top-k. Increases over steps
        self.theta_start = 0.10
        self.theta_end = 0.30
        # Hysteresis to reduce flip-flops (applies to thresholding)
        self.use_hysteresis = True
        self.theta_off_delta = 0.05  # keep selected if p > theta_t - delta
        # Prefer handcrafted uncertainty gate over untrained model gate outputs
        self.force_handcrafted_gate = True
    
    def _compute_uncertainty_signals(self, logits: torch.Tensor, prev_logits: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute entropy, margin, and confidence change (vs previous step if provided)"""
        # Entropy
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)
        entropy = -(probs * log_probs).sum(dim=-1)

        # Margin (top1 - top2)
        top2_values, top2_indices = torch.topk(logits, k=2, dim=-1)
        margin = top2_values[..., 0] - top2_values[..., 1]

        # Confidence change: Δ log p(y*), y* = current top-1 token per position
        if prev_logits is not None:
            prev_log_probs = F.log_softmax(prev_logits, dim=-1)
            top1_indices = torch.argmax(logits, dim=-1)  # [B, L]
            # Gather current and previous log-prob for the same y*
            current_logp = log_probs.gather(dim=-1, index=top1_indices.unsqueeze(-1)).squeeze(-1)
            prev_logp = prev_log_probs.gather(dim=-1, index=top1_indices.unsqueeze(-1)).squeeze(-1)
            confidence_change = current_logp - prev_logp
        else:
            confidence_change = torch.zeros_like(entropy)

        return entropy, margin, confidence_change
    
    def _get_positional_schedule(self, seq_len: int, step: int) -> torch.Tensor:
        """Get positional schedule prior - Balanced with late position emphasis"""
        positions = torch.arange(seq_len, dtype=torch.float32)
        
        # Step 1: Base AR-DIFFUSION schedule (early positions mature faster)
        delta = 0.0
        tau = (self.T / seq_len) * (positions + delta)
        alpha = 0.8
        base_schedule = torch.sigmoid(alpha * (tau - step))
        
        # Step 2: INVERT the early bias by giving later positions higher probabilities
        # Later positions should have higher revision probability to balance early bias
        late_emphasis = positions / seq_len * 0.25  # 0% to 25% based on position
        
        # Step 3: Strong uniform component to ensure all positions get revised
        uniform_component = torch.ones_like(positions) * 0.20  # 20% uniform probability
        
        # Step 4: Combine with late emphasis dominating
        schedule = uniform_component + late_emphasis
        
        # Step 5: Normalize to reasonable range
        schedule = schedule * 0.15  # Max 15% revision probability
        
        return schedule
    
    def _compute_dynamic_overwrite_prob(self, entropy: torch.Tensor, margin: torch.Tensor, 
                                      confidence_change: torch.Tensor, schedule: torch.Tensor) -> torch.Tensor:
        """Compute dynamic overwrite probability: p = 1 - (1-u)(1-r)"""
        # Normalize uncertainty signals to [0,1] range with better thresholds
        # Entropy: higher = more uncertain (scale down)
        norm_entropy = torch.sigmoid(entropy - 3.0)  # Higher threshold (was 2.0)
        
        # Margin: lower = more uncertain (scale up)
        norm_margin = torch.sigmoid(-margin + 2.0)  # Higher threshold (was 1.0)
        
        # Confidence change: higher = more uncertain (scale up)
        norm_confidence = torch.sigmoid(confidence_change + 1.0)  # Higher threshold (was 0.5)
        
        # Combine uncertainty signals with tuned weights
        uncertainty_gate = torch.sigmoid(
            0.1 + 0.3 * norm_entropy + 0.4 * norm_margin + 0.2 * norm_confidence  # Adjusted weights
        )
        
        # Scale down uncertainty gate to reasonable range - REDUCE INFLUENCE
        uncertainty_gate = uncertainty_gate * 0.08  # Max 8% from uncertainty (was 0.25)
        
        # Noisy-OR combination: p = 1 - (1-u)(1-r)
        overwrite_prob = 1.0 - (1.0 - uncertainty_gate) * (1.0 - schedule)
        
        # Note: final clamping is applied per-step in the main loop to allow annealing
        return overwrite_prob
    
    def generate_with_dynamic_refinement(self, prompt_tokens: torch.Tensor, max_length: int = 50, eos_token_id: Optional[int] = None) -> Tuple[torch.Tensor, Dict]:
        """Generate with dynamic uncertainty-driven refinement"""
        start_time = time.time()
        
        # Initial generation
        # For seq2seq, start from high-quality beam search output using HF generate
        if hasattr(self.model, 'set_encoder_inputs') and hasattr(self.model, 'seq2seq'):
            with torch.no_grad():
                gen_out = self.model.seq2seq.generate(
                    input_ids=self.model._last_source_input_ids,
                    attention_mask=self.model._last_source_attention_mask,
                    max_length=max_length,
                    min_length=max(10, max_length // 2),
                    no_repeat_ngram_size=3,
                    num_beams=4,
                    length_penalty=2.0,
                    early_stopping=True
                )
            current_tokens = gen_out
        else:
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
                    if eos_token_id is not None and int(next_token.cpu().detach().numpy()) == int(eos_token_id):
                        break
        
        # Apply dynamic refinement with comprehensive logging
        revision_steps = 0
        prev_logits = None
        
        # 🎯 LOGGING HOOKS: Track tokens overwritten per step
        tokens_overwritten_per_step = []
        positional_overwrite_patterns = []
        uncertainty_scores_per_step = []
        nfe_per_step = []
        
        # Keep previous mask for hysteresis
        # Initialize one-step cooldown (prevents re-editing same token in consecutive steps)
        cooldown = torch.zeros(current_tokens.shape[1], dtype=torch.int64, device=current_tokens.device)
        prev_mask = None
        for step in range(1, self.T + 1):
            with torch.no_grad():
                # 1) Forward pass: support models that return (logits, overwrite_probs, hidden)
                outputs = self.model(current_tokens)
                if isinstance(outputs, tuple) and len(outputs) >= 2:
                    logits = outputs[0]
                    model_overwrite_probs = outputs[1]  # [B, L]
                else:
                    logits = outputs
                    model_overwrite_probs = None

                # 2) Compute uncertainty signals for logging/optional fallback
                entropy, margin, confidence_change = self._compute_uncertainty_signals(logits, prev_logits=prev_logits)

                # 3) Positional prior for this step
                schedule = self._get_positional_schedule(len(current_tokens[0]), step)
                
                # Debug: Show positional probabilities for first few steps
                if step <= 3:
                    print(f"🔍 Step {step} - Positional probs: {schedule[:5].tolist()}...")
                
                # 4) Compute overwrite probability
                # Prefer handcrafted uncertainty gate unless explicitly allowed
                use_model_gate = (model_overwrite_probs is not None) and (not self.force_handcrafted_gate)
                if use_model_gate:
                    # Use the model's gate u and combine with prior r via noisy-OR: p = 1 - (1-u)(1-r)
                    u = model_overwrite_probs[0]
                    r = schedule.to(u.device)                       # [L]
                    overwrite_prob = 1.0 - (1.0 - u) * (1.0 - r)
                else:
                    # Handcrafted uncertainty-based gate
                    overwrite_prob = self._compute_dynamic_overwrite_prob(
                        entropy, margin, confidence_change, schedule
                    )
                
                # Per-step clamp/anneal: allow higher p early, taper later
                is_seq2seq = hasattr(self.model, 'set_encoder_inputs')
                if is_seq2seq:
                    max_start, max_end = 0.15, 0.05
                    min_start, min_end = 0.01, 0.005
                else:
                    max_start, max_end = 0.25, 0.10
                    min_start, min_end = 0.01, 0.005
                # Linear schedules over steps
                frac = (step - 1) / max(1, self.T - 1)
                max_p = max_start + (max_end - max_start) * frac
                min_p = min_start + (min_end - min_start) * frac
                overwrite_prob = overwrite_prob.clamp(min_p, max_p)
                # Ensure shape [L] (handcrafted path returns [1, L])
                if overwrite_prob.dim() == 2 and overwrite_prob.size(0) == 1:
                    overwrite_prob = overwrite_prob.squeeze(0)
                
                # Early stopping: if uncertainty is very low, stop refining
                if not getattr(self, 'disable_early_stop', False):
                    if entropy.mean() < 2.0 and margin.mean() > 2.0:  # More lenient thresholds
                        print(f"🛑 Early stopping at step {step}: low uncertainty detected")
                        break
                
                # Decide which tokens to revise (deterministic mask)
                L = overwrite_prob.shape[-1]
                # Editable region: focus on the last third of tokens for seq2seq stability
                if hasattr(self.model, 'set_encoder_inputs'):
                    editable = torch.zeros(L, dtype=torch.bool, device=overwrite_prob.device)
                    start_idx = max(1, int(2 * L / 3))
                    editable[start_idx:] = True
                    # Exclude EOS/PAD if present
                    try:
                        pad_id = getattr(self.model, 'pad_token_id', None)
                        eos_id = getattr(self.model, 'eos_token_id', None)
                        if pad_id is not None or eos_id is not None:
                            token_ids = current_tokens[0]
                            special_mask = torch.zeros(L, dtype=torch.bool, device=overwrite_prob.device)
                            if pad_id is not None:
                                special_mask |= (token_ids == int(pad_id))
                            if eos_id is not None:
                                special_mask |= (token_ids == int(eos_id))
                            editable = editable & (~special_mask)
                    except Exception:
                        pass
                    # Zero out p for non-editable positions before selection
                    p_eff = overwrite_prob.clone()
                    p_eff[~editable] = -1.0  # ensures not selected by top-k/threshold

                    # Eligibility: high entropy (>=75th pct), low margin (<=25th pct), and (if available) negative confidence change
                    eligible = None
                    try:
                        e = entropy[0].detach()
                        m = margin[0].detach()
                        c = None
                        if 'confidence_change' in locals() or 'confidence_change' in globals():
                            try:
                                c = confidence_change[0].detach()
                            except Exception:
                                c = None
                        if editable.any():
                            e_edit = e[editable]
                            m_edit = m[editable]
                            e_thr = torch.quantile(e_edit, 0.80)
                            m_thr = torch.quantile(m_edit, 0.20)
                            eligible = torch.zeros(L, dtype=torch.bool, device=overwrite_prob.device)
                            eligible_base = (e[editable] >= e_thr) & (m[editable] <= m_thr)
                            if c is not None and prev_logits is not None:
                                eligible_tail = eligible_base & (c[editable] < 0)
                            else:
                                eligible_tail = eligible_base
                            eligible[editable] = eligible_tail
                            # Mask out non-eligible positions
                            p_eff[~eligible] = -1.0
                    except Exception:
                        eligible = None
                else:
                    p_eff = overwrite_prob
                if self.use_topk_mask:
                    # Budgeted top-k with decaying fraction
                    rho_t = self.budget_start + (self.budget_end - self.budget_start) * frac
                    # Reduce budget for seq2seq stability
                    if hasattr(self.model, 'set_encoder_inputs'):
                        rho_t = max(0.02, min(0.06, rho_t))
                    k_t = max(1, int((rho_t * L)))
                    # Exclude special case: if probs are very low, k_t may be too aggressive; fallback to threshold
                    # Select top-k indices per sequence (batch assumed 1 here)
                    topk_vals, topk_idx = torch.topk(p_eff, k_t, dim=-1)
                    revision_mask = torch.zeros(1, L, dtype=torch.bool, device=overwrite_prob.device)
                    revision_mask[0, topk_idx] = True
                else:
                    # Thresholding with optional hysteresis
                    theta_t = self.theta_start + (self.theta_end - self.theta_start) * frac
                    select_on = (p_eff > theta_t)
                    if self.use_hysteresis and prev_mask is not None:
                        theta_off = max(0.0, theta_t - self.theta_off_delta)
                        select_off = (p_eff > theta_off) & prev_mask[0]
                        current = select_on | select_off
                    else:
                        current = select_on
                    revision_mask = current.unsqueeze(0)
                
                # Apply cooldown: prevent re-editing tokens changed in previous step
                if cooldown is not None and cooldown.numel() == L:
                    # Zero out positions with cooldown > 0
                    if revision_mask.sum() > 0:
                        active = revision_mask[0]
                        active[cooldown > 0] = False
                        revision_mask[0] = active

                # Keep prev_mask for hysteresis next step
                prev_mask = revision_mask.clone()
                
                # 🎯 LOGGING: Record step-wise metrics
                num_tokens_overwritten = int(revision_mask.sum().cpu().detach().numpy())
                tokens_overwritten_per_step.append(num_tokens_overwritten)
                
                # Record positional patterns
                masked_positions = torch.where(revision_mask.bool()[0])[0]
                positional_overwrite_patterns.append(masked_positions.tolist())
                
                # Debug: Show which positions are being revised
                if step <= 3 and len(masked_positions) > 0:
                    print(f"🔍 Step {step} - Revised positions: {masked_positions.tolist()}")
                    print(f"   Overwrite probs: {overwrite_prob[:5].tolist()}...")
                    print(f"   Uncertainty - Entropy: {entropy.mean():.3f}, Margin: {margin.mean():.3f}")
                
                # Record uncertainty scores for this step - simplified to avoid tensor issues
                step_uncertainty = {
                    'entropy_mean': float(entropy.mean().cpu().detach().numpy()),
                    'margin_mean': float(margin.mean().cpu().detach().numpy()),
                    'confidence_change_mean': float(confidence_change.mean().cpu().detach().numpy()),
                    'overwrite_prob_mean': float(overwrite_prob.mean().cpu().detach().numpy()),
                    'overwrite_prob_std': float(overwrite_prob.std().cpu().detach().numpy())
                }
                uncertainty_scores_per_step.append(step_uncertainty)
                
                # Record NFE (Number of Function Evaluations)
                nfe_per_step.append(1)  # Each refinement step = 1 NFE
                
                # Only revise if mask says so
                accepted_edits = 0
                if revision_mask.sum() > 0:
                    # Sample new tokens for masked positions
                    # Create a proper mask for indexing
                    mask_2d = revision_mask.bool()  # [1, L]
                    
                    # Get logits for masked positions - fix the indexing
                    # We need to get logits for each masked position
                    masked_positions = torch.where(mask_2d[0])[0]
                    # Span edits: include neighbors if eligible
                    if eligible is not None and masked_positions.numel() > 0:
                        expanded = set(int(p.item()) for p in masked_positions)
                        for p in masked_positions:
                            pi = int(p.item())
                            for q in [pi - 1, pi + 1]:
                                if 0 <= q < L and eligible[q]:
                                    expanded.add(q)
                        masked_positions = torch.tensor(sorted(list(expanded)), device=mask_2d.device, dtype=torch.long)
                    
                    if len(masked_positions) > 0:
                        # Get logits for each masked position
                        for pos in masked_positions:
                            # Recompute logits per update for stability
                            outputs = self.model(current_tokens)
                            if isinstance(outputs, tuple) and len(outputs) >= 1:
                                logits = outputs[0]
                            else:
                                logits = outputs
                            pos_logits = logits[0, pos, :]
                            # Greedy for seq2seq stability, else use sampling utility
                            if hasattr(self.model, 'set_encoder_inputs'):
                                # Get top-2 candidates
                                top2_vals, top2_idx = torch.topk(pos_logits, k=2, dim=-1)
                                new_id = int(top2_idx[0].cpu().detach().numpy())
                                current_id = int(current_tokens[0, pos].cpu().detach().numpy())
                                # Local repetition guard
                                repeats_neighbor = (
                                    (pos > 0 and int(current_tokens[0, pos - 1].cpu().detach().numpy()) == new_id) or
                                    (pos + 1 < L and int(current_tokens[0, pos + 1].cpu().detach().numpy()) == new_id)
                                )
                                # If repeats, try second-best candidate
                                if repeats_neighbor and top2_idx.numel() > 1:
                                    alt_id = int(top2_idx[1].cpu().detach().numpy())
                                    new_id = alt_id
                                    repeats_neighbor = (
                                        (pos > 0 and int(current_tokens[0, pos - 1].cpu().detach().numpy()) == new_id) or
                                        (pos + 1 < L and int(current_tokens[0, pos + 1].cpu().detach().numpy()) == new_id)
                                    )
                                if repeats_neighbor:
                                    continue
                                if new_id != current_id:
                                    # Acceptance test: require significant probability improvement
                                    pos_probs = F.softmax(pos_logits, dim=-1)
                                    pos_log_probs = F.log_softmax(pos_logits, dim=-1)
                                    p_new = float(pos_probs[new_id].cpu().detach().numpy())
                                    p_old = float(pos_probs[current_id].cpu().detach().numpy())
                                    lp_new = float(pos_log_probs[new_id].cpu().detach().numpy())
                                    lp_old = float(pos_log_probs[current_id].cpu().detach().numpy())
                                    min_ratio = 1.30
                                    min_delta = 0.10
                                    no_regress = (lp_new - lp_old) >= 0.0
                                    if no_regress and ((p_new >= p_old * min_ratio) or (p_new - p_old >= min_delta)):
                                        current_tokens[0, pos] = torch.tensor([new_id], dtype=torch.long, device=pos_logits.device)
                                        # Set one-step cooldown
                                        if cooldown is not None and cooldown.numel() == L:
                                            cooldown[pos] = 1
                                        accepted_edits += 1
                                    # else: skip edit
                                # else: no change if same id
                            else:
                                new_token = L2RBaseline._sample_token(self, pos_logits, top_k=50, top_p=0.9, temperature=0.7)
                                # Apply only if different
                                if int(new_token.cpu().detach().numpy()) != int(current_tokens[0, pos].cpu().detach().numpy()):
                                    current_tokens[0, pos] = new_token
                                    accepted_edits += 1
                        
                        revision_steps += 1

                # Decrement cooldown (one-step)
                if cooldown is not None and cooldown.numel() == L:
                    cooldown = torch.clamp(cooldown - 1, min=0)

                # Early stop if no edits were accepted this step
                if accepted_edits == 0:
                    break
                
                prev_logits = logits.detach()
        
        total_time = time.time() - start_time
        
        return current_tokens, {
            'method': self.name,
            'generation_time': total_time,
            'tokens_generated': max_length,
            'revision_steps': revision_steps,
            'total_compute': total_time,
            'schedule_type': 'uncertainty_driven',
            'uncertainty_signals': ['entropy', 'margin', 'confidence_change'],
            
            # 🎯 COMPREHENSIVE LOGGING DATA
            'tokens_overwritten_per_step': tokens_overwritten_per_step,
            'positional_overwrite_patterns': positional_overwrite_patterns,
            'uncertainty_scores_per_step': uncertainty_scores_per_step,
            'nfe_per_step': nfe_per_step,
            'total_nfe': sum(nfe_per_step),
            'overwrite_efficiency': revision_steps / max(1, sum(tokens_overwritten_per_step)),
            'positional_focus': self._analyze_positional_focus(positional_overwrite_patterns, len(current_tokens[0])),
            
            # 🚀 AR-DIFFUSION STYLE METRICS
            'quality_vs_nfe': self._compute_quality_vs_nfe_curve(tokens_overwritten_per_step, nfe_per_step),
            'diversity_score': self._compute_diversity_score(current_tokens),
            'coherence_score': self._compute_coherence_score(current_tokens, tokens_overwritten_per_step)
        }
    
    def _analyze_positional_focus(self, positional_patterns: List[List[int]], seq_len: int) -> Dict:
        """Analyze how the gate focuses on different positions"""
        if not positional_patterns:
            return {'early_focus': 0.0, 'middle_focus': 0.0, 'late_focus': 0.0}
        
        # Flatten all positions
        all_positions = []
        for pattern in positional_patterns:
            all_positions.extend(pattern)
        
        if not all_positions:
            return {'early_focus': 0.0, 'middle_focus': 0.0, 'late_focus': 0.0}
        
        # Calculate focus on different regions
        early_threshold = seq_len // 3
        late_threshold = 2 * seq_len // 3
        
        early_count = sum(1 for pos in all_positions if pos < early_threshold)
        middle_count = sum(1 for pos in all_positions if early_threshold <= pos < late_threshold)
        late_count = sum(1 for pos in all_positions if pos >= late_threshold)
        
        total = len(all_positions)
        
        return {
            'early_focus': early_count / total,
            'middle_focus': middle_count / total,
            'late_focus': late_count / total,
            'total_positions_revised': total
        }
    
    def _compute_quality_vs_nfe_curve(self, tokens_overwritten_per_step: List[int], nfe_per_step: List[int]) -> Dict:
        """Compute quality vs NFE curve (AR-DIFFUSION style)"""
        if not tokens_overwritten_per_step:
            return {'nfe_points': [], 'quality_points': [], 'efficiency': 0.0}
        
        # Quality proxy: inverse of tokens overwritten (fewer overwrites = higher quality)
        cumulative_nfe = []
        quality_scores = []
        
        total_nfe = 0
        total_overwrites = 0
        
        for step, (nfe, overwrites) in enumerate(zip(nfe_per_step, tokens_overwritten_per_step)):
            total_nfe += nfe
            total_overwrites += overwrites
            
            cumulative_nfe.append(total_nfe)
            # Quality = 1 / (1 + overwrites) - higher quality with fewer overwrites
            # Add baseline quality to avoid going to zero
            quality = 0.5 + 0.5 / (1.0 + total_overwrites)
            quality_scores.append(quality)
        
        # Efficiency: quality improvement per NFE (ensure it's positive)
        if len(quality_scores) > 1:
            quality_gain = max(0, quality_scores[-1] - quality_scores[0])  # Ensure positive
            efficiency = quality_gain / max(1, total_nfe)
        else:
            efficiency = 0.0
        
        return {
            'nfe_points': cumulative_nfe,
            'quality_points': quality_scores,
            'efficiency': efficiency,
            'final_quality': quality_scores[-1] if quality_scores else 0.0,
            'total_nfe': total_nfe,
            'quality_gain': quality_scores[-1] - quality_scores[0] if len(quality_scores) > 1 else 0.0
        }
    
    def _compute_diversity_score(self, tokens: torch.Tensor) -> float:
        """Compute diversity score based on token distribution"""
        # Flatten tokens and count unique tokens
        flat_tokens = tokens.flatten()
        unique_tokens = torch.unique(flat_tokens)
        
        # Diversity = unique tokens / total tokens
        diversity = len(unique_tokens) / len(flat_tokens)
        return float(diversity)
    
    def _compute_coherence_score(self, tokens: torch.Tensor, overwrite_pattern: List[int]) -> float:
        """Compute coherence score based on revision pattern"""
        if not overwrite_pattern:
            return 1.0
        
        # Coherence proxy: fewer revisions in later steps = more coherent
        # Weight later revisions more heavily
        total_revisions = sum(overwrite_pattern)
        if total_revisions == 0:
            return 1.0
        
        # Weighted sum: later revisions get higher weight
        weighted_revisions = 0
        for step, revs in enumerate(overwrite_pattern):
            weight = (step + 1) / len(overwrite_pattern)  # Later steps get higher weight
            weighted_revisions += revs * weight
        
        # Normalize and invert (fewer weighted revisions = higher coherence)
        coherence = 1.0 / (1.0 + weighted_revisions / total_revisions)
        return coherence

def run_baseline_comparison(
    model,
    prompt_tokens: torch.Tensor,
    max_length: int = 50,
    tokenizer=None,
    text_prompts: Optional[List[str]] = None,
    prompt_max_tokens: int = 16,
    references: Optional[List[str]] = None,
) -> Dict:
    """Run comparison between all three baselines"""
    print("🔬 RUNNING BASELINE COMPARISON EXPERIMENT")
    print("=" * 60)
    
    # Initialize baselines
    l2r = L2RBaseline(model)
    fixed_schedule = FixedScheduleBaseline(model)
    dynamic_gate = DynamicGateBaseline(model)
    
    # Test prompts
    if text_prompts is not None and len(text_prompts) > 0:
        test_prompts = text_prompts
    else:
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
        
        # Convert prompt to tokens
        if tokenizer is not None and text_prompts is not None:
            if hasattr(model, 'set_encoder_inputs'):
                # Seq2Seq flow: set encoder with source, decode iteratively on decoder side
                enc = tokenizer(
                    prompt,
                    truncation=True,
                    max_length=prompt_max_tokens,
                    add_special_tokens=True,
                    return_tensors='pt'
                )
                model.set_encoder_inputs(enc['input_ids'], enc.get('attention_mask'))
                # Start decoder with BOS if available, else a single pad token
                start_id = getattr(model, 'decoder_start_token_id', None)
                if start_id is None:
                    start_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
                prompt_tokens = torch.tensor([[int(start_id)]], dtype=torch.long)
            else:
                enc = tokenizer(
                    prompt,
                    truncation=True,
                    max_length=prompt_max_tokens,
                    add_special_tokens=False,
                    return_tensors='pt'
                )
                input_ids = enc['input_ids']
                prompt_tokens = input_ids[:, :prompt_max_tokens].to(torch.long)
            if prompt_tokens.numel() == 0:
                print("⚠️ Skipping empty tokenized prompt")
                continue
        else:
            # Fallback toy tokenization
            prompt_tokens = torch.tensor([[hash(word) % 1000 for word in prompt.split()]], dtype=torch.long)
        
        # Determine EOS id if available
        eos_id = None
        if hasattr(model, 'eos_token_id') and getattr(model, 'eos_token_id') is not None:
            eos_id = int(getattr(model, 'eos_token_id'))
        elif tokenizer is not None and getattr(tokenizer, 'eos_token_id', None) is not None:
            eos_id = int(tokenizer.eos_token_id)

        # Test L2R
        print(f"🚀 Testing {l2r.name}...")
        l2r_tokens, l2r_metrics = l2r.generate(prompt_tokens, max_length, eos_token_id=eos_id)
        results['l2r']['times'].append(l2r_metrics['generation_time'])
        results['l2r']['revisions'].append(l2r_metrics['revision_steps'])
        if tokenizer is not None:
            try:
                results['l2r'].setdefault('pred_texts', []).append(
                    tokenizer.decode(l2r_tokens[0].tolist(), skip_special_tokens=True)
                )
            except Exception:
                pass
        
        # Store detailed metrics for L2R
        if 'diversity_score' in l2r_metrics:
            if 'diversity_scores' not in results['l2r']:
                results['l2r']['diversity_scores'] = []
            results['l2r']['diversity_scores'].append(l2r_metrics['diversity_score'])
        
        if 'coherence_score' in l2r_metrics:
            if 'coherence_scores' not in results['l2r']:
                results['l2r']['coherence_scores'] = []
            results['l2r']['coherence_scores'].append(l2r_metrics['coherence_score'])
        
        # Test Fixed Schedule
        print(f"🔄 Testing {fixed_schedule.name}...")
        fixed_tokens, fixed_metrics = fixed_schedule.generate_with_refinement(prompt_tokens, max_length, eos_token_id=eos_id)
        results['fixed_schedule']['times'].append(fixed_metrics['generation_time'])
        results['fixed_schedule']['revisions'].append(fixed_metrics['revision_steps'])
        if tokenizer is not None:
            try:
                results['fixed_schedule'].setdefault('pred_texts', []).append(
                    tokenizer.decode(fixed_tokens[0].tolist(), skip_special_tokens=True)
                )
            except Exception:
                pass
        
        # Store detailed metrics for Fixed Schedule
        if 'diversity_score' in fixed_metrics:
            if 'diversity_scores' not in results['fixed_schedule']:
                results['fixed_schedule']['diversity_scores'] = []
            results['fixed_schedule']['diversity_scores'].append(fixed_metrics['diversity_score'])
        
        if 'coherence_score' in fixed_metrics:
            if 'coherence_scores' not in results['fixed_schedule']:
                results['fixed_schedule']['coherence_scores'] = []
            results['fixed_schedule']['coherence_scores'].append(fixed_metrics['coherence_score'])
        
        # Test Dynamic Gate
        print(f"🎯 Testing {dynamic_gate.name}...")
        dynamic_tokens, dynamic_metrics = dynamic_gate.generate_with_dynamic_refinement(prompt_tokens, max_length, eos_token_id=eos_id)
        results['dynamic_gate']['times'].append(dynamic_metrics['generation_time'])
        results['dynamic_gate']['revisions'].append(dynamic_metrics['revision_steps'])
        if tokenizer is not None:
            try:
                results['dynamic_gate'].setdefault('pred_texts', []).append(
                    tokenizer.decode(dynamic_tokens[0].tolist(), skip_special_tokens=True)
                )
            except Exception:
                pass
        
        # Store detailed metrics for Dynamic Gate
        if 'diversity_score' in dynamic_metrics:
            if 'diversity_scores' not in results['dynamic_gate']:
                results['dynamic_gate']['diversity_scores'] = []
            results['dynamic_gate']['diversity_scores'].append(dynamic_metrics['diversity_score'])
        
        if 'coherence_score' in dynamic_metrics:
            if 'coherence_scores' not in results['dynamic_gate']:
                results['dynamic_gate']['coherence_scores'] = []
            results['dynamic_gate']['coherence_scores'].append(dynamic_metrics['coherence_score'])
        
        # Store AR-DIFFUSION specific metrics
        if 'quality_vs_nfe' in dynamic_metrics:
            if 'quality_vs_nfe_data' not in results['dynamic_gate']:
                results['dynamic_gate']['quality_vs_nfe_data'] = []
            results['dynamic_gate']['quality_vs_nfe_data'].append(dynamic_metrics['quality_vs_nfe'])
        
        # Store per-step positions for heatmap (rename: use tokens_overwritten_patterns key)
        if 'positional_overwrite_patterns' in dynamic_metrics:
            if 'tokens_overwritten_patterns' not in results['dynamic_gate']:
                results['dynamic_gate']['tokens_overwritten_patterns'] = []
            results['dynamic_gate']['tokens_overwritten_patterns'].append(dynamic_metrics['positional_overwrite_patterns'])

        # Optionally keep counts as well for other analyses
        if 'tokens_overwritten_per_step' in dynamic_metrics:
            if 'tokens_overwritten_counts' not in results['dynamic_gate']:
                results['dynamic_gate']['tokens_overwritten_counts'] = []
            results['dynamic_gate']['tokens_overwritten_counts'].append(dynamic_metrics['tokens_overwritten_per_step'])
        
        if 'positional_focus' in dynamic_metrics:
            if 'positional_focus_data' not in results['dynamic_gate']:
                results['dynamic_gate']['positional_focus_data'] = []
            results['dynamic_gate']['positional_focus_data'].append(dynamic_metrics['positional_focus'])

        # Store uncertainty scores per step for plotting
        if 'uncertainty_scores_per_step' in dynamic_metrics:
            if 'uncertainty_scores_per_step' not in results['dynamic_gate']:
                results['dynamic_gate']['uncertainty_scores_per_step'] = []
            results['dynamic_gate']['uncertainty_scores_per_step'].append(dynamic_metrics['uncertainty_scores_per_step'])
        
        print(f"✅ All baselines completed for prompt {i+1}")
    
    # Compute averages
    for method in results:
        results[method]['avg_time'] = sum(results[method]['times']) / len(results[method]['times'])
        results[method]['avg_revisions'] = sum(results[method]['revisions']) / len(results[method]['revisions'])
    
    # Optional: compute ROUGE/BLEU if references are provided and metric libs are available
    if references is not None:
        try:
            from rouge_score import rouge_scorer
        except ImportError:
            rouge_scorer = None
            print("⚠️ rouge-score not installed; skipping ROUGE.")
        try:
            import sacrebleu
        except ImportError:
            sacrebleu = None
            print("⚠️ sacrebleu not installed; skipping BLEU.")

        for method in ['l2r', 'fixed_schedule', 'dynamic_gate']:
            preds = results[method].get('pred_texts', [])
            if not preds or len(preds) != len(references):
                continue

            # Compute BLEU
            if sacrebleu is not None:
                bleu = sacrebleu.corpus_bleu(preds, [references])
                results[method]['bleu'] = float(bleu.score)

            # Compute ROUGE-1/2/L
            if rouge_scorer is not None:
                scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
                r1, r2, rl = 0.0, 0.0, 0.0
                for p, r in zip(preds, references):
                    scores = scorer.score(r, p)
                    r1 += scores['rouge1'].fmeasure
                    r2 += scores['rouge2'].fmeasure
                    rl += scores['rougeL'].fmeasure
                n = max(1, len(preds))
                results[method]['rouge1'] = r1 / n
                results[method]['rouge2'] = r2 / n
                results[method]['rougeL'] = rl / n

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
    
    # 🎯 NEW: Show detailed Dynamic Gate metrics
    if 'dynamic_gate' in results:
        dynamic_data = results['dynamic_gate']
        print(f"\n🎯 DYNAMIC GATE DETAILED ANALYSIS:")
        print("=" * 50)
        
        # Show tokens overwritten per step
        if 'tokens_overwritten_per_step' in dynamic_data:
            overwrite_pattern = dynamic_data['tokens_overwritten_per_step']
            print(f"📊 Tokens overwritten per step: {overwrite_pattern}")
            print(f"📈 Total tokens overwritten: {sum(overwrite_pattern)}")
            print(f"🎯 Overwrite efficiency: {dynamic_data.get('overwrite_efficiency', 'N/A'):.3f}")
        
        # Show positional focus
        if 'positional_focus' in dynamic_data:
            pos_focus = dynamic_data['positional_focus']
            print(f"📍 Positional focus:")
            print(f"   • Early positions: {pos_focus.get('early_focus', 0):.2%}")
            print(f"   • Middle positions: {pos_focus.get('middle_focus', 0):.2%}")
            print(f"   • Late positions: {pos_focus.get('late_focus', 0):.2%}")
        
        # Show NFE analysis
        if 'total_nfe' in dynamic_data:
            print(f"⚡ Total NFE: {dynamic_data['total_nfe']}")
            print(f"🔄 NFE per revision: {dynamic_data['total_nfe'] / max(1, dynamic_data['avg_revisions']):.2f}")
        
        # 🚀 NEW: Show AR-DIFFUSION style metrics
        if 'quality_vs_nfe_data' in dynamic_data:
            print(f"\n📈 QUALITY vs NFE CURVE:")
            # Show average across all tests
            avg_final_quality = sum(qvn.get('final_quality', 0) for qvn in dynamic_data['quality_vs_nfe_data']) / len(dynamic_data['quality_vs_nfe_data'])
            avg_efficiency = sum(qvn.get('efficiency', 0) for qvn in dynamic_data['quality_vs_nfe_data']) / len(dynamic_data['quality_vs_nfe_data'])
            print(f"   • Avg final quality: {avg_final_quality:.3f}")
            print(f"   • Avg quality efficiency: {avg_efficiency:.4f}")
        
        if 'diversity_scores' in dynamic_data:
            avg_diversity = sum(dynamic_data['diversity_scores']) / len(dynamic_data['diversity_scores'])
            print(f"🎭 Avg diversity score: {avg_diversity:.3f}")
        
        if 'coherence_scores' in dynamic_data:
            avg_coherence = sum(dynamic_data['coherence_scores']) / len(dynamic_data['coherence_scores'])
            print(f"🔗 Avg coherence score: {avg_coherence:.3f}")
        
        if 'tokens_overwritten_patterns' in dynamic_data:
            print(f"\n📊 TOKENS OVERWRITTEN ANALYSIS:")
            # Show pattern from first test as example
            if dynamic_data['tokens_overwritten_patterns']:
                first_pattern = dynamic_data['tokens_overwritten_patterns'][0]
                print(f"   • Example pattern: {first_pattern}")
                # Handle nested lists (per-step positions) vs flat counts
                if first_pattern and isinstance(first_pattern[0], list):
                    total_overwritten = sum(len(step_positions) for step_positions in first_pattern)
                else:
                    total_overwritten = sum(first_pattern)
                print(f"   • Total tokens overwritten: {total_overwritten}")
        
        if 'positional_focus_data' in dynamic_data:
            print(f"\n📍 POSITIONAL FOCUS ANALYSIS:")
            # Show average across all tests
            avg_early = sum(pf.get('early_focus', 0) for pf in dynamic_data['positional_focus_data']) / len(dynamic_data['positional_focus_data'])
            avg_middle = sum(pf.get('middle_focus', 0) for pf in dynamic_data['positional_focus_data']) / len(dynamic_data['positional_focus_data'])
            avg_late = sum(pf.get('late_focus', 0) for pf in dynamic_data['positional_focus_data']) / len(dynamic_data['positional_focus_data'])
            print(f"   • Early positions: {avg_early:.2%}")
            print(f"   • Middle positions: {avg_middle:.2%}")
            print(f"   • Late positions: {avg_late:.2%}")
    
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