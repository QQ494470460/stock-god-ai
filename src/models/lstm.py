# -*- coding: utf-8 -*-
"""
LSTM 深度学习模型
用于时间序列价格趋势预测
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List
from pathlib import Path
from loguru import logger

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from config.settings import config as app_config, MODEL_DIR


class StockLSTM(nn.Module):
    """股票预测LSTM模型"""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_classes: int = 3  # 涨/平/跌
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
    
    def attention_net(self, lstm_output: torch.Tensor) -> torch.Tensor:
        """注意力机制"""
        attn_weights = self.attention(lstm_output)  # [batch, seq_len, 1]
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = torch.sum(attn_weights * lstm_output, dim=1)  # [batch, hidden*2]
        return context
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, input_size]
        lstm_out, _ = self.lstm(x)  # [batch, seq_len, hidden*2]
        
        # Attention pooling
        context = self.attention_net(lstm_out)  # [batch, hidden*2]
        
        out = self.dropout(context)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out


class LSTMTrainer:
    """LSTM训练器"""
    
    def __init__(
        self,
        input_size: int,
        model_dir: Optional[Path] = None,
        device: Optional[str] = None
    ):
        self.input_size = input_size
        self.model_dir = Path(model_dir) if model_dir else MODEL_DIR
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        params = app_config.ml.lstm_params
        self.model = StockLSTM(
            input_size=input_size,
            hidden_size=params["hidden_size"],
            num_layers=params["num_layers"],
            dropout=params["dropout"]
        ).to(self.device)
        
        self.sequence_length = params["sequence_length"]
        self.batch_size = params["batch_size"]
        self.epochs = params["epochs"]
        self.learning_rate = params["learning_rate"]
        self.patience = params["patience"]
        
        logger.info(f"LSTM Trainer initialized on {self.device}")
    
    def prepare_sequences(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        sequence_length: int = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """将时间序列数据转为监督学习序列"""
        if sequence_length is None:
            sequence_length = self.sequence_length
        
        X, y = [], []
        for i in range(len(features) - sequence_length):
            X.append(features[i:i + sequence_length])
            y.append(labels[i + sequence_length])
        
        X = torch.FloatTensor(np.array(X))
        y = torch.LongTensor(np.array(y))
        
        # 将标签从 -1/0/1 转为 0/1/2
        y = y + 1
        
        return X, y
    
    def train(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        val_split: float = 0.2,
        save_best: bool = True
    ) -> dict:
        """训练LSTM模型"""
        logger.info(f"开始训练LSTM: {features.shape}")
        
        # 准备序列数据
        X, y = self.prepare_sequences(features, labels)
        
        # 划分训练/验证集
        split_idx = int(len(X) * (1 - val_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        logger.info(f"训练集: {len(X_train)}, 验证集: {len(X_val)}")
        
        train_loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=self.batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(X_val, y_val),
            batch_size=self.batch_size,
            shuffle=False
        )
        
        # 损失函数 & 优化器
        # 类别不平衡处理
        class_counts = torch.bincount(y_train, minlength=3)
        class_weights = 1.0 / (class_counts.float() + 1e-10)
        class_weights = class_weights / class_weights.sum() * 3
        class_weights = class_weights.to(self.device)
        
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
        
        history = {"train_loss": [], "val_loss": [], "val_acc": []}
        best_val_loss = float("inf")
        patience_counter = 0
        
        for epoch in range(self.epochs):
            # 训练
            self.model.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            history["train_loss"].append(train_loss)
            
            # 验证
            self.model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == batch_y).sum().item()
                    total += batch_y.size(0)
            
            val_loss /= len(val_loader)
            val_acc = correct / total
            
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                if save_best:
                    self.save("lstm_best.pt")
            else:
                patience_counter += 1
            
            if patience_counter >= self.patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch + 1}/{self.epochs} - "
                          f"Train Loss: {train_loss:.4f}, "
                          f"Val Loss: {val_loss:.4f}, "
                          f"Val Acc: {val_acc:.4f}")
        
        logger.info(f"LSTM训练完成，最佳验证准确率: {max(history['val_acc']):.4f}")
        return history
    
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """预测概率"""
        self.model.eval()
        
        # 需要sequence_length长度的序列
        if len(features) < self.sequence_length:
            raise ValueError(f"需要至少{self.sequence_length}条数据")
        
        seq = features[-self.sequence_length:].reshape(1, self.sequence_length, -1)
        seq = torch.FloatTensor(seq).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(seq)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
        
        return probs
    
    def save(self, filename: str = "lstm_model.pt"):
        """保存模型"""
        path = self.model_dir / filename
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "input_size": self.input_size,
            "sequence_length": self.sequence_length,
        }, path)
        logger.info(f"模型已保存到 {path}")
    
    def load(self, filename: str = "lstm_model.pt"):
        """加载模型"""
        path = self.model_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"模型文件不存在: {path}")
        
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.input_size = checkpoint["input_size"]
        self.sequence_length = checkpoint["sequence_length"]
        
        # 重建模型
        params = app_config.ml.lstm_params
        self.model = StockLSTM(
            input_size=self.input_size,
            hidden_size=params["hidden_size"],
            num_layers=params["num_layers"],
            dropout=params["dropout"]
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        logger.info(f"模型已从 {path} 加载")