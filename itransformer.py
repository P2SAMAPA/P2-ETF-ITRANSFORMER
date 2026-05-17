import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class iTransformer(nn.Module):
    def __init__(self, num_vars, seq_len, d_model=64, n_heads=8, n_layers=3, dropout=0.1):
        super().__init__()
        self.num_vars = num_vars
        self.seq_len = seq_len
        self.d_model = d_model
        # Linear projection from input to d_model (each time step)
        self.input_proj = nn.Linear(seq_len, d_model)
        # Transformer encoder (treats each variable as a token? Actually we want to model across time steps)
        # Standard iTransformer: embed each variable as a token with sequence dimension as features.
        # We'll use a transformer over the time dimension for each variable separately, then aggregate.
        # Simpler: treat each variable as a token, and the sequence length as the feature dimension.
        # That's the iTransformer idea: reverse the dimensions.
        # We'll use a TransformerEncoder with input shape (batch, num_vars, seq_len) -> we project to (batch, num_vars, d_model)
        self.embedding = nn.Linear(seq_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        # Variable selection: attention across variables? We'll just use linear readout
        self.readout = nn.Linear(d_model, 1)   # predict return for each variable? We need to predict the target variable (ETF returns)
        # For each ETF, we treat its own sequence as a variable, and we want to predict its next return.
        # So we output a scalar per ETF.

    def forward(self, x):
        # x: (batch, num_vars, seq_len)
        # Project to embedding
        x = self.embedding(x)   # (batch, num_vars, d_model)
        # Transformer over variables (treat each variable as a token)
        x = self.transformer(x) # (batch, num_vars, d_model)
        # Readout: predict return for each variable
        out = self.readout(x).squeeze(-1)  # (batch, num_vars)
        return out

def train_model(X, y, seq_len, num_vars, epochs=30, batch_size=32, lr=1e-3, device='cpu'):
    """
    X: numpy array of shape (n_samples, num_vars, seq_len)
    y: numpy array of shape (n_samples, num_vars)  (targets for each ETF)
    """
    model = iTransformer(num_vars=num_vars, seq_len=seq_len)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    dataset = torch.tensor(X, dtype=torch.float32)
    targets = torch.tensor(y, dtype=torch.float32)
    n = len(dataset)
    # Simple train/val split
    split = int(0.8 * n)
    X_train, X_val = dataset[:split], dataset[split:]
    y_train, y_val = targets[:split], targets[split:]
    for epoch in range(epochs):
        model.train()
        for i in range(0, n, batch_size):
            batch_X = X_train[i:i+batch_size].to(device)
            batch_y = y_train[i:i+batch_size].to(device)
            optimizer.zero_grad()
            pred = model(batch_X)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
        if (epoch+1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val.to(device))
                val_loss = criterion(val_pred, y_val.to(device))
            print(f"    Epoch {epoch+1}/{epochs}, train loss: {loss.item():.4f}, val loss: {val_loss.item():.4f}")
    return model

def predict(model, X, device='cpu'):
    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        pred = model(X_t).cpu().numpy()
    return pred
