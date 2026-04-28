# models/autoencoder.py
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import joblib

class FraudAutoencoder(nn.Module):
    def __init__(self, input_dim=30, encoding_dim=10):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 20),
            nn.ReLU(),
            nn.Linear(20, encoding_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 20),
            nn.ReLU(),
            nn.Linear(20, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def reconstruction_error(self, x):
        with torch.no_grad():
            reconstructed = self.forward(x)
            mse = nn.MSELoss(reduction="none")
            error = mse(reconstructed, x).mean(dim=1)
        return error.cpu().numpy()

class FraudDetector:
    def __init__(self, input_dim=30, encoding_dim=10, threshold_percentile=95):
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.threshold_percentile = threshold_percentile
        self.model = None
        self.scaler = StandardScaler()
        self.threshold = None

    def train(self, X_train, epochs=50, batch_size=256, lr=1e-3):
        X_scaled = self.scaler.fit_transform(X_train)
        X_tensor = torch.FloatTensor(X_scaled)
        dataset = TensorDataset(X_tensor, X_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model = FraudAutoencoder(self.input_dim, self.encoding_dim)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            total_loss = 0.0
            for batch_x, _ in dataloader:
                optimizer.zero_grad()
                output = self.model(batch_x)
                loss = criterion(output, batch_x)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if (epoch + 1) % 10 == 0:
                print(f"[AE] Epoch {epoch+1}/{epochs} Loss: {total_loss/len(dataloader):.4f}")

        self._calculate_threshold(X_tensor)
        print(f"[AE] Threshold: {self.threshold:.6f}")

    def _calculate_threshold(self, X_tensor):
        errors = self.model.reconstruction_error(X_tensor)
        self.threshold = np.percentile(errors, self.threshold_percentile)

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled)
        errors = self.model.reconstruction_error(X_tensor)
        is_fraud = errors > self.threshold
        fraud_score = errors / self.threshold
        return is_fraud, fraud_score

    def save(self, base_path):
        os.makedirs(os.path.dirname(base_path), exist_ok=True)
        torch.save(self.model.state_dict(), base_path + ".pth")
        joblib.dump(self.scaler, base_path + "_scaler.pkl")
        np.save(base_path + "_threshold.npy", self.threshold)

    def load(self, base_path):
        self.model = FraudAutoencoder(self.input_dim, self.encoding_dim)
        self.model.load_state_dict(torch.load(base_path + ".pth", map_location="cpu"))
        self.model.eval()
        self.scaler = joblib.load(base_path + "_scaler.pkl")
        self.threshold = np.load(base_path + "_threshold.npy", allow_pickle=True)