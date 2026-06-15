"""
lstm_predictor.py — LSTM Traffic Predictor
============================================
The Fortune Teller of your system.

Looks at last 10 traffic readings → predicts next value.

KEY IMPROVEMENT ADDED:
- 80/20 train/validation split
- Tracks both training AND validation loss
- Plots both curves so you can visually detect overfitting
- If validation loss diverges from training loss = overfit
- If both stay close together = good fit
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import os


class LSTMModel(nn.Module):
    """
    The actual neural network.

    Think of it as a brain with memory gates:
    - Input gate:  what new info to remember
    - Forget gate: what old info to throw away
    - Output gate: what to predict based on memory

    2 layers deep = can learn more complex patterns
    Dropout = randomly turns off 20% of neurons during
              training to prevent memorization (overfitting)
    """

    def __init__(self, input_size=1, hidden_size=64,
                 num_layers=2, dropout=0.2):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        prediction = self.fc(last_output)
        return prediction


class LSTMPredictor:
    """
    Wrapper around LSTMModel.
    Handles training, validation, saving, loading, predicting.

    NEW: train() now returns both train_losses AND val_losses
    so you can plot them and check for overfitting.
    """

    def __init__(self, sequence_length=10):
        self.sequence_length = sequence_length
        self.model = LSTMModel()
        self.is_trained = False
        self.data_min = 0
        self.data_max = 1000

        # Store losses for later plotting
        self.train_losses = []
        self.val_losses = []

    def _normalize(self, data):
        """Scale data to 0-1 range for better training."""
        return (data - self.data_min) / (
            self.data_max - self.data_min + 1e-8
        )

    def _denormalize(self, data):
        """Convert prediction back to real request numbers."""
        return data * (self.data_max - self.data_min) + self.data_min

    def _prepare_sequences(self, workload):
        """
        Convert flat traffic array into sequences.

        Example with sequence_length=3:
        Input:  [100, 120, 150, 200, 300]
        Output: X=[[100,120,150], [120,150,200]]
                y=[200, 300]

        This is what the LSTM trains on:
        - X = the last N readings (input)
        - y = what actually came next (answer)
        """
        normalized = self._normalize(workload)
        X, y = [], []

        for i in range(len(normalized) - self.sequence_length):
            X.append(normalized[i:i + self.sequence_length])
            y.append(normalized[i + self.sequence_length])

        X = torch.tensor(
            np.array(X), dtype=torch.float32
        ).unsqueeze(-1)
        y = torch.tensor(
            np.array(y), dtype=torch.float32
        ).unsqueeze(-1)

        return X, y

    def train(self, workload, epochs=50, batch_size=32,
              learning_rate=0.001, val_split=0.2):
        """
        Train LSTM with proper train/validation split.

        val_split=0.2 means:
        - 80% of data used for training (model learns from this)
        - 20% of data used for validation (model NEVER trains on this)

        During each epoch we check both:
        - training loss   = how well it predicts training data
        - validation loss = how well it predicts data it never saw

        If validation loss stays close to training loss = good fit
        If validation loss starts rising while training falls = overfit
        """
        print("🧠 Training LSTM Predictor...")
        print(f"   Data split: {int((1-val_split)*100)}% train / "
              f"{int(val_split*100)}% validation")

        # Set normalization range from full data
        self.data_min = float(np.min(workload))
        self.data_max = float(np.max(workload))

        # Prepare ALL sequences
        X, y = self._prepare_sequences(workload)
        total_samples = len(X)

        # SPLIT into train and validation
        # ─────────────────────────────────────────────
        # Example: 990 sequences total
        # val_split = 0.2
        # val_size  = 198  ← never trained on
        # train_size = 792 ← trained on
        # ─────────────────────────────────────────────
        val_size = int(total_samples * val_split)
        train_size = total_samples - val_size

        # Important: use LAST 20% for validation
        # because traffic data is sequential —
        # we want to validate on "future" data
        X_train, y_train = X[:train_size], y[:train_size]
        X_val, y_val = X[train_size:], y[train_size:]

        print(f"   Training sequences:   {train_size}")
        print(f"   Validation sequences: {val_size}")

        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )

        # Optimizer and loss function
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate
        )
        loss_fn = nn.MSELoss()

        # Training loop
        self.train_losses = []
        self.val_losses = []
        best_val_loss = float('inf')
        patience = 10  # stop if validation doesn't improve for 10 epochs
        patience_counter = 0

        for epoch in range(epochs):

            # ── Training phase ──
            self.model.train()
            epoch_train_loss = 0

            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                prediction = self.model(batch_X)
                loss = loss_fn(prediction, batch_y)
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item()

            avg_train_loss = epoch_train_loss / len(train_loader)

            # ── Validation phase ──
            # Model does NOT update weights here
            # Just checks how well it predicts unseen data
            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(X_val)
                avg_val_loss = loss_fn(val_pred, y_val).item()

            self.train_losses.append(avg_train_loss)
            self.val_losses.append(avg_val_loss)

            if (epoch + 1) % 10 == 0:
                gap = abs(avg_val_loss - avg_train_loss)
                status = "✅" if gap < 0.005 else "⚠️"
                print(f"  Epoch {epoch+1:3d}/{epochs} "
                      f"— Train: {avg_train_loss:.6f} "
                      f"| Val: {avg_val_loss:.6f} "
                      f"| Gap: {gap:.6f} {status}")

            # Early stopping — prevents overfitting
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best weights
                self._best_weights = {
                    k: v.clone()
                    for k, v in self.model.state_dict().items()
                }
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\n  Early stopping at epoch {epoch+1} "
                          f"— validation not improving")
                    break

        # Restore best weights
        if hasattr(self, '_best_weights'):
            self.model.load_state_dict(self._best_weights)

        self.is_trained = True

        # Final verdict
        final_train = self.train_losses[-1]
        final_val = self.val_losses[-1]
        gap = abs(final_val - final_train)

        print(f"\n✅ LSTM training complete!")
        print(f"   Final train loss: {final_train:.6f}")
        print(f"   Final val loss:   {final_val:.6f}")
        print(f"   Gap:              {gap:.6f}")

        if gap < 0.003:
            print(f"   Verdict: ✅ GOOD FIT — no overfitting detected")
        elif gap < 0.01:
            print(f"   Verdict: ⚠️  SLIGHT OVERFIT — acceptable for research")
        else:
            print(f"   Verdict: ❌ OVERFIT — consider more data or regularization")

        return self.train_losses, self.val_losses

    def plot_train_val_loss(self, save_path='results/lstm_train_val_loss.png'):
        """
        Plot training vs validation loss curves.

        This is THE graph that shows overfitting/underfitting.

        Good fit:   both lines go down together, stay close
        Overfit:    training keeps going down, validation goes UP
        Underfit:   both lines stay high, barely move
        """
        if not self.train_losses:
            print("No training history found. Train first.")
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(
            'LSTM Training vs Validation Loss\n'
            'Overfitting Detection',
            fontsize=13, fontweight='bold'
        )

        # Left plot — full training curve
        ax1 = axes[0]
        ax1.plot(self.train_losses, color='steelblue',
                 linewidth=2, label='Training Loss')
        ax1.plot(self.val_losses, color='crimson',
                 linewidth=2, linestyle='--',
                 label='Validation Loss')
        ax1.set_title('Full Training Curve')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('MSE Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Right plot — zoomed in on last 70% of training
        # (more interesting — shows convergence behavior)
        ax2 = axes[1]
        start = len(self.train_losses) // 3
        ax2.plot(self.train_losses[start:],
                 color='steelblue', linewidth=2,
                 label='Training Loss')
        ax2.plot(self.val_losses[start:],
                 color='crimson', linewidth=2,
                 linestyle='--', label='Validation Loss')
        ax2.set_title('Zoomed — Convergence Behavior')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MSE Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Add verdict annotation
        final_gap = abs(self.val_losses[-1] - self.train_losses[-1])
        if final_gap < 0.003:
            verdict = "✅ Good Fit — No Overfitting"
            color = 'green'
        elif final_gap < 0.01:
            verdict = "⚠️ Slight Overfit — Acceptable"
            color = 'orange'
        else:
            verdict = "❌ Overfit Detected"
            color = 'red'

        fig.text(0.5, 0.02, f'Verdict: {verdict}',
                 ha='center', fontsize=12,
                 color=color, fontweight='bold')

        plt.tight_layout(rect=[0, 0.05, 1, 1])
        os.makedirs('results', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        plt.close()
        print(f"✅ Train/val loss graph saved to {save_path}")

    def predict_next(self, recent_history):
        """
        Predict the next traffic value.
        Called in real time during simulation.

        Input:  last 10 traffic values
        Output: predicted next value
        """
        if not self.is_trained:
            return float(np.mean(recent_history))

        self.model.eval()
        with torch.no_grad():
            normalized = self._normalize(
                np.array(recent_history[-self.sequence_length:])
            )
            x = torch.tensor(normalized, dtype=torch.float32)
            x = x.unsqueeze(0).unsqueeze(-1)
            pred_normalized = self.model(x)
            prediction = self._denormalize(pred_normalized.item())

        return max(0, prediction)

    def evaluate_on_data(self, workload, label='Test'):
        """
        Evaluate prediction accuracy on any dataset.
        Used for cross-shard validation.

        Returns:
        - mae:  Mean Absolute Error (avg error in req/min)
        - mape: Mean Absolute Percentage Error (avg % error)
        - predictions: all predicted values
        - actuals: all actual values
        """
        self.model.eval()
        predictions = []
        actuals = []

        for i in range(self.sequence_length, len(workload)):
            history = workload[i-self.sequence_length:i]
            pred = self.predict_next(history)
            predictions.append(pred)
            actuals.append(float(workload[i]))

        predictions = np.array(predictions)
        actuals = np.array(actuals)

        mae = np.mean(np.abs(predictions - actuals))
        mape = np.mean(
            np.abs((predictions - actuals) /
                   (actuals + 1e-8)) * 100
        )

        print(f"\n📊 LSTM Evaluation on {label} data:")
        print(f"   Mean Absolute Error:  {mae:.2f} req/min")
        print(f"   Mean % Error (MAPE):  {mape:.2f}%")
        print(f"   Samples predicted:    {len(predictions)}")

        return {
            'mae': mae,
            'mape': mape,
            'predictions': predictions,
            'actuals': actuals,
            'label': label
        }

    def save(self, path='models/lstm_model.pth'):
        """Save trained model to file."""
        os.makedirs('models', exist_ok=True)
        torch.save({
            'model_state': self.model.state_dict(),
            'data_min': self.data_min,
            'data_max': self.data_max,
            'sequence_length': self.sequence_length,
            'is_trained': self.is_trained,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }, path)
        print(f"✅ LSTM model saved to {path}")

    def load(self, path='models/lstm_model.pth'):
        """Load trained model from file."""
        checkpoint = torch.load(path, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state'])
        self.data_min = checkpoint['data_min']
        self.data_max = checkpoint['data_max']
        self.sequence_length = checkpoint['sequence_length']
        self.is_trained = checkpoint['is_trained']
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.model.eval()
        print(f"✅ LSTM model loaded from {path}")

    def plot_predictions(self, workload,
                         title='LSTM Predictions'):
        """Plot predicted vs actual traffic."""
        self.model.eval()
        actuals = []
        predictions = []

        for i in range(self.sequence_length, len(workload)):
            history = workload[i-self.sequence_length:i]
            pred = self.predict_next(history)
            predictions.append(pred)
            actuals.append(workload[i])

        plt.figure(figsize=(14, 5))
        plt.plot(actuals, label='Actual Traffic',
                 color='steelblue', linewidth=1.5)
        plt.plot(predictions, label='LSTM Prediction',
                 color='crimson', linewidth=1.5,
                 linestyle='--', alpha=0.8)
        plt.title(title, fontsize=13, fontweight='bold')
        plt.xlabel('Time Steps')
        plt.ylabel('Requests/min')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        os.makedirs('results', exist_ok=True)
        filename = (
            f"results/lstm_"
            f"{title.lower().replace(' ', '_')}.png"
        )
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.show()
        plt.close()
        print(f"✅ Graph saved to {filename}")

        return actuals, predictions