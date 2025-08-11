import sys
sys.path.append('src')

import torch
from models.uncertainty_gate import UncertaintyARDM

print('=' * 60)
print('UNCERTAINTY-DRIVEN ARDM RESEARCH DEMO')
print('=' * 60)

# Create model
print('1. Creating Uncertainty ARDM...')
model = UncertaintyARDM(vocab_size=64, max_seq_len=16, hidden_dim=128)
print(f'Model created with {sum(p.numel() for p in model.parameters()):,} parameters')

# Create synthetic data
print('\n2. Creating synthetic training data...')
batch_size = 8
seq_len = 16
num_samples = 100

input_ids = torch.randint(0, 64, (num_samples, seq_len))
print(f'Created {num_samples} training samples of length {seq_len}')

# Training loop
print('\n3. Starting training...')
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(5):
    total_loss = 0
    num_batches = 0
    
    for i in range(0, num_samples, batch_size):
        batch = input_ids[i:i+batch_size]
        
        # Forward pass
        optimizer.zero_grad()
        logits, overwrite_probs, hidden = model(batch)
        
        # Simple loss (cross-entropy on logits)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, 64), 
            batch.view(-1)
        )
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_overwrite = overwrite_probs.mean().item()
    
    print(f'Epoch {epoch+1}/5: Loss = {avg_loss:.4f}, Avg Overwrite Rate = {avg_overwrite:.3f}')

print('\n4. Training completed!')
print('\n5. Testing generation...')

# Test generation
model.eval()
with torch.no_grad():
    x = torch.randint(0, 64, (1, 16))
    logits, overwrite_probs, hidden = model(x)
    
    print(f'Generated logits shape: {logits.shape}')
    print(f'Overwrite probabilities: {overwrite_probs[0].detach().numpy()}')
    print(f'Average overwrite rate: {overwrite_probs.mean():.3f}')

print('\n🎉 Research demo completed successfully!')
print('Your uncertainty-driven ARDM is working!')
