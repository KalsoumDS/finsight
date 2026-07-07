"""
FinSight — ML Alpha Signals
Génération de signaux de trading avec :
  - XGBoost (signal classification : up/down/neutral)
  - LSTM PyTorch (prédiction de direction)
  - Walk-Forward Validation (pas de data leakage)
  - Métriques rigoureuses : accuracy, F1, precision/recall
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, classification_report, confusion_matrix
)
from sklearn.model_selection import TimeSeriesSplit
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False


# ── LSTM Architecture ────────────────────────────────────────────────────────
class LSTMSignalModel(nn.Module):
    """
    LSTM pour classification du signal de trading.
    Input  : séquence de features techniques
    Output : probabilité [down, neutral, up]
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        n_classes: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.bn = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, (h, _) = self.lstm(x)
        h_last = h[-1]                    # dernier layer
        h_last = self.bn(h_last)
        h_last = self.dropout(h_last)
        return self.fc(h_last)


# ── Label engineering ────────────────────────────────────────────────────────
def make_labels(
    returns: pd.Series,
    forward_horizon: int = 5,
    threshold: float = 0.005,
) -> pd.Series:
    """
    Crée les labels de trading à partir des rendements futurs.
    0 = Down  (rendement < -threshold)
    1 = Neutral
    2 = Up    (rendement > +threshold)

    IMPORTANT : utilise les rendements *futurs* (forward returns),
    décalés proprement pour éviter le look-ahead bias.
    """
    forward_ret = returns.shift(-forward_horizon).rolling(forward_horizon).sum()
    labels = pd.cut(
        forward_ret,
        bins=[-np.inf, -threshold, threshold, np.inf],
        labels=[0, 1, 2]
    ).astype(float)
    return labels


# ── Walk-Forward Validation ───────────────────────────────────────────────────
def walk_forward_split(
    n: int,
    n_splits: int = 5,
    min_train_size: int = 252,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Génère des splits temporels sans data leakage.
    Chaque fold : train sur le passé, test sur le futur immédiat.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits, max_train_size=None, test_size=None)
    indices = np.arange(n)
    splits = []
    for train_idx, test_idx in tscv.split(indices):
        if len(train_idx) >= min_train_size:
            splits.append((train_idx, test_idx))
    return splits


# ── XGBoost Signal Generator ─────────────────────────────────────────────────
class XGBoostSignalGenerator:
    """
    Génération de signaux avec XGBoost + walk-forward validation.
    Features : indicateurs techniques calculés dans data.py
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        n_splits: int = 5,
        forward_horizon: int = 5,
        threshold: float = 0.005,
    ):
        self.params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "use_label_encoder": False,
            "eval_metric": "mlogloss",
            "objective": "multi:softprob",
            "num_class": 3,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
        }
        self.n_splits = n_splits
        self.forward_horizon = forward_horizon
        self.threshold = threshold
        self.model = None
        self.feature_names: List[str] = []
        self.scaler = StandardScaler()
        self.results_: Dict = {}

    def fit(self, features: pd.DataFrame, returns: pd.Series) -> Dict:
        """
        Entraîne avec walk-forward validation.
        features : DataFrame de features techniques
        returns  : rendements de l'actif
        """
        if not XGB_AVAILABLE:
            raise ImportError("xgboost non installé. pip install xgboost")

        labels = make_labels(returns, self.forward_horizon, self.threshold)
        common_idx = features.index.intersection(labels.dropna().index)
        X = features.loc[common_idx].values
        y = labels.loc[common_idx].values.astype(int)
        self.feature_names = list(features.columns)

        splits = walk_forward_split(len(X), self.n_splits)
        all_preds, all_true, all_proba = [], [], []

        for train_idx, test_idx in splits:
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            model = xgb.XGBClassifier(**self.params)
            model.fit(
                X_train_s, y_train,
                eval_set=[(X_test_s, y_test)],
                verbose=False
            )
            preds = model.predict(X_test_s)
            proba = model.predict_proba(X_test_s)

            all_preds.extend(preds)
            all_true.extend(y_test)
            all_proba.extend(proba)

        # Entraînement final sur tout le dataset
        X_s = self.scaler.fit_transform(X)
        self.model = xgb.XGBClassifier(**self.params)
        self.model.fit(X_s, y, verbose=False)

        # Métriques agrégées
        all_preds = np.array(all_preds)
        all_true = np.array(all_true)
        all_proba = np.array(all_proba)

        self.results_ = {
            "accuracy": float(accuracy_score(all_true, all_preds)),
            "f1_macro": float(f1_score(all_true, all_preds, average="macro")),
            "f1_weighted": float(f1_score(all_true, all_preds, average="weighted")),
            "precision": float(precision_score(all_true, all_preds, average="macro")),
            "recall": float(recall_score(all_true, all_preds, average="macro")),
            "classification_report": classification_report(
                all_true, all_preds,
                target_names=["Down", "Neutral", "Up"],
                output_dict=True
            ),
            "confusion_matrix": confusion_matrix(all_true, all_preds),
            "feature_importance": dict(zip(
                self.feature_names,
                self.model.feature_importances_
            )),
            "n_samples": len(all_true),
            "n_splits": len(splits),
        }
        return self.results_

    def predict(self, features: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prédit les signaux sur de nouvelles données."""
        X = self.scaler.transform(features[self.feature_names].values)
        preds = self.model.predict(X)
        proba = self.model.predict_proba(X)
        return preds, proba

    def get_signals(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Retourne un DataFrame avec les signaux et probabilités.
        Signal : -1 (Down), 0 (Neutral), +1 (Up)
        """
        preds, proba = self.predict(features)
        signal_map = {0: -1, 1: 0, 2: 1}
        signals = pd.DataFrame({
            "signal": [signal_map[p] for p in preds],
            "prob_down": proba[:, 0],
            "prob_neutral": proba[:, 1],
            "prob_up": proba[:, 2],
            "confidence": proba.max(axis=1),
        }, index=features.index[-len(preds):])
        return signals


# ── LSTM Signal Generator ─────────────────────────────────────────────────────
class LSTMSignalGenerator:
    """
    Génération de signaux avec LSTM PyTorch.
    Utilise des séquences temporelles pour capturer les dépendances long-terme.
    """

    def __init__(
        self,
        sequence_length: int = 20,
        hidden_size: int = 64,
        num_layers: int = 2,
        n_epochs: int = 30,
        batch_size: int = 32,
        lr: float = 1e-3,
        forward_horizon: int = 5,
        threshold: float = 0.005,
        device: str = None,
    ):
        self.seq_len = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.forward_horizon = forward_horizon
        self.threshold = threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[LSTMSignalModel] = None
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self.train_losses: List[float] = []
        self.results_: Dict = {}

    def _create_sequences(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        Xs, ys = [], []
        for i in range(self.seq_len, len(X)):
            Xs.append(X[i - self.seq_len:i])
            ys.append(y[i])
        return np.array(Xs), np.array(ys)

    def fit(
        self,
        features: pd.DataFrame,
        returns: pd.Series,
        progress_callback=None,
    ) -> Dict:
        self.feature_names = list(features.columns)
        labels = make_labels(returns, self.forward_horizon, self.threshold)
        common_idx = features.index.intersection(labels.dropna().index)
        X_raw = features.loc[common_idx].values
        y_raw = labels.loc[common_idx].values.astype(int)

        # Normalisation
        X_scaled = self.scaler.fit_transform(X_raw)

        # Séquences
        X_seq, y_seq = self._create_sequences(X_scaled, y_raw)

        # Train/val split temporel (80/20)
        split = int(len(X_seq) * 0.8)
        X_train, X_val = X_seq[:split], X_seq[split:]
        y_train, y_val = y_seq[:split], y_seq[split:]

        # Datasets PyTorch
        train_ds = TensorDataset(
            torch.FloatTensor(X_train).to(self.device),
            torch.LongTensor(y_train).to(self.device)
        )
        val_ds = TensorDataset(
            torch.FloatTensor(X_val).to(self.device),
            torch.LongTensor(y_val).to(self.device)
        )
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=False)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)

        # Modèle
        self.model = LSTMSignalModel(
            input_size=X_scaled.shape[1],
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
        ).to(self.device)

        # Gestion du déséquilibre des classes
        class_counts = np.bincount(y_train)
        class_weights = torch.FloatTensor(
            1.0 / (class_counts + 1)
        ).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5
        )

        self.train_losses = []
        best_val_acc = 0.0
        best_state = None

        for epoch in range(self.n_epochs):
            # Train
            self.model.train()
            train_loss = 0.0
            for X_b, y_b in train_loader:
                optimizer.zero_grad()
                out = self.model(X_b)
                loss = criterion(out, y_b)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()

            avg_loss = train_loss / max(len(train_loader), 1)
            self.train_losses.append(avg_loss)

            # Validation
            self.model.eval()
            val_preds, val_true = [], []
            with torch.no_grad():
                for X_b, y_b in val_loader:
                    out = self.model(X_b)
                    preds = out.argmax(dim=1).cpu().numpy()
                    val_preds.extend(preds)
                    val_true.extend(y_b.cpu().numpy())

            val_acc = accuracy_score(val_true, val_preds)
            scheduler.step(avg_loss)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}

            if progress_callback:
                progress_callback(epoch + 1, self.n_epochs, avg_loss, val_acc)

        # Charger le meilleur modèle
        if best_state:
            self.model.load_state_dict(best_state)

        # Métriques finales sur validation
        self.model.eval()
        all_preds, all_true, all_proba = [], [], []
        with torch.no_grad():
            for X_b, y_b in val_loader:
                out = self.model(X_b)
                proba = torch.softmax(out, dim=1).cpu().numpy()
                preds = out.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_true.extend(y_b.cpu().numpy())
                all_proba.extend(proba)

        all_preds = np.array(all_preds)
        all_true = np.array(all_true)

        self.results_ = {
            "accuracy": float(accuracy_score(all_true, all_preds)),
            "f1_macro": float(f1_score(all_true, all_preds, average="macro")),
            "f1_weighted": float(f1_score(all_true, all_preds, average="weighted")),
            "classification_report": classification_report(
                all_true, all_preds,
                target_names=["Down", "Neutral", "Up"],
                output_dict=True
            ),
            "confusion_matrix": confusion_matrix(all_true, all_preds),
            "train_losses": self.train_losses,
            "best_val_accuracy": best_val_acc,
            "n_train": len(X_train),
            "n_val": len(X_val),
            "sequence_length": self.seq_len,
        }
        return self.results_

    def predict(self, features: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        if self.model is None:
            raise ValueError("Modèle non entraîné. Appeler fit() d'abord.")
        X = self.scaler.transform(features[self.feature_names].values)
        sequences = np.array([
            X[i - self.seq_len:i]
            for i in range(self.seq_len, len(X) + 1)
        ])
        if len(sequences) == 0:
            return np.array([]), np.array([])

        tensor = torch.FloatTensor(sequences).to(self.device)
        self.model.eval()
        with torch.no_grad():
            out = self.model(tensor)
            proba = torch.softmax(out, dim=1).cpu().numpy()
            preds = out.argmax(dim=1).cpu().numpy()

        return preds, proba
