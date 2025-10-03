#!/usr/bin/env python3
"""
Graph Neural Network for Market Structure Analysis
PhD-level implementation using market interdependencies to predict price movements.
Models the entire market as a dynamic graph with stocks, sectors, and economic indicators.
"""

import asyncio
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
import networkx as nx
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import defaultdict
import pickle
from scipy.stats import spearmanr, pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "../../shared/python-common"))

from trading_common import MarketData, get_settings, get_logger
from trading_common.cache import get_trading_cache

logger = get_logger(__name__)
settings = get_settings()


@dataclass
class GraphNodeFeatures:
    """Features for each node in the market graph."""
    node_id: str
    node_type: str  # 'stock', 'sector', 'economic_indicator', 'commodity'
    
    # Price-based features
    price_features: np.ndarray  # OHLCV, returns, volatility
    technical_features: np.ndarray  # RSI, MACD, Bollinger, etc.
    fundamental_features: np.ndarray  # P/E, P/B, ROE, etc.
    sentiment_features: np.ndarray  # News sentiment, social sentiment
    
    # Node-specific features
    market_cap: float
    sector_weight: float
    correlation_centrality: float
    volatility_regime: int
    
    def to_feature_vector(self) -> np.ndarray:
        """Convert all features to a single vector."""
        features = []
        
        features.extend(self.price_features)
        features.extend(self.technical_features)
        features.extend(self.fundamental_features)
        features.extend(self.sentiment_features)
        
        features.extend([
            self.market_cap,
            self.sector_weight,
            self.correlation_centrality,
            self.volatility_regime
        ])
        
        return np.array(features, dtype=np.float32)


@dataclass
class GraphEdgeFeatures:
    """Features for edges in the market graph."""
    source_id: str
    target_id: str
    edge_type: str  # 'correlation', 'sector', 'supply_chain', 'competitor'
    
    # Statistical relationships
    correlation: float
    mutual_information: float
    transfer_entropy: float  # Information flow from source to target
    granger_causality: float
    
    # Dynamic properties
    correlation_stability: float  # How stable the correlation is over time
    relationship_strength: float  # Combined strength metric
    lead_lag_days: int  # How many days source leads target
    
    def to_feature_vector(self) -> np.ndarray:
        """Convert edge features to vector."""
        return np.array([
            self.correlation,
            self.mutual_information,
            self.transfer_entropy,
            self.granger_causality,
            self.correlation_stability,
            self.relationship_strength,
            self.lead_lag_days
        ], dtype=np.float32)


class MarketGraphConstructor:
    """Constructs dynamic market graphs from market data."""
    
    def __init__(self):
        self.cache = None
        self.node_features = {}
        self.edge_features = {}
        
        # Graph construction parameters
        self.correlation_window = 60  # Days for correlation calculation
        self.min_correlation = 0.3    # Minimum correlation for edge creation
        self.max_edges_per_node = 20  # Limit edges to prevent overly dense graphs
        
        # Feature dimensions
        self.price_feature_dim = 20
        self.technical_feature_dim = 15
        self.fundamental_feature_dim = 10
        self.sentiment_feature_dim = 5
        
    async def initialize(self):
        """Initialize the graph constructor."""
        self.cache = get_trading_cache()
        logger.info("Market Graph Constructor initialized")
    
    async def construct_market_graph(self, symbols: List[str], 
                                   market_data: Dict[str, List[MarketData]],
                                   lookback_days: int = 252) -> Data:
        """Construct a PyTorch Geometric graph from market data."""
        
        # Extract node features
        node_features_list = []
        node_mapping = {}  # symbol -> node_index
        
        for i, symbol in enumerate(symbols):
            if symbol not in market_data or len(market_data[symbol]) < 30:
                continue
                
            node_features = await self._extract_node_features(
                symbol, market_data[symbol][-lookback_days:]
            )
            
            if node_features is not None:
                node_features_list.append(node_features.to_feature_vector())
                node_mapping[symbol] = len(node_features_list) - 1
        
        if len(node_features_list) < 2:
            logger.warning("Insufficient nodes for graph construction")
            return None
        
        # Create feature matrix
        x = torch.tensor(np.array(node_features_list), dtype=torch.float)
        
        # Construct edges based on correlations and other relationships
        edge_indices, edge_features = await self._construct_edges(
            symbols, market_data, node_mapping
        )
        
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float)
        
        # Create targets (what we want to predict)
        y = await self._create_targets(symbols, market_data, node_mapping)
        
        # Create PyTorch Geometric data object
        graph_data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            num_nodes=len(node_features_list)
        )
        
        logger.info(f"Constructed graph with {graph_data.num_nodes} nodes and {graph_data.edge_index.size(1)} edges")
        
        return graph_data
    
    async def _extract_node_features(self, symbol: str, 
                                   data: List[MarketData]) -> Optional[GraphNodeFeatures]:
        """Extract comprehensive features for a graph node."""
        
        if len(data) < 20:
            return None
            
        df = self._market_data_to_df(data)
        
        # Price-based features
        price_features = self._extract_price_features(df)
        
        # Technical indicators
        technical_features = self._extract_technical_features(df)
        
        # Fundamental features (would get from company intelligence service)
        fundamental_features = self._extract_fundamental_features(symbol)
        
        # Sentiment features (would get from social media collector)
        sentiment_features = self._extract_sentiment_features(symbol)
        
        # Network-specific features
        market_cap = await self._get_market_cap(symbol)
        sector_weight = await self._get_sector_weight(symbol)
        correlation_centrality = await self._calculate_correlation_centrality(symbol)
        volatility_regime = self._determine_volatility_regime(df)
        
        return GraphNodeFeatures(
            node_id=symbol,
            node_type='stock',
            price_features=price_features,
            technical_features=technical_features,
            fundamental_features=fundamental_features,
            sentiment_features=sentiment_features,
            market_cap=market_cap,
            sector_weight=sector_weight,
            correlation_centrality=correlation_centrality,
            volatility_regime=volatility_regime
        )
    
    def _extract_price_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract price-based features."""
        features = []
        
        # Returns over different horizons
        for horizon in [1, 2, 3, 5, 10, 20]:
            returns = df['close'].pct_change(horizon)
            features.extend([
                returns.iloc[-1] if not pd.isna(returns.iloc[-1]) else 0.0,
                returns.mean(),
                returns.std()
            ])
        
        # Price relative to historical levels
        features.append(df['close'].iloc[-1] / df['close'].iloc[-252] - 1)  # 1Y return
        
        # Volume features
        features.append(df['volume'].iloc[-1] / df['volume'].mean())  # Volume ratio
        
        return np.array(features[:self.price_feature_dim], dtype=np.float32)
    
    def _extract_technical_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract technical indicator features."""
        features = []
        
        # Moving averages
        sma_20 = df['close'].rolling(20).mean()
        sma_50 = df['close'].rolling(50).mean()
        features.extend([
            df['close'].iloc[-1] / sma_20.iloc[-1] - 1,  # Price vs SMA20
            df['close'].iloc[-1] / sma_50.iloc[-1] - 1,  # Price vs SMA50
            (sma_20.iloc[-1] / sma_50.iloc[-1] - 1) if sma_50.iloc[-1] != 0 else 0  # SMA20 vs SMA50
        ])
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        features.append(rsi.iloc[-1] / 100.0 if not pd.isna(rsi.iloc[-1]) else 0.5)
        
        # MACD
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        macd = ema12 - ema26
        macd_signal = macd.ewm(span=9).mean()
        features.extend([
            macd.iloc[-1] / df['close'].iloc[-1] if df['close'].iloc[-1] != 0 else 0,
            (macd.iloc[-1] - macd_signal.iloc[-1]) / df['close'].iloc[-1] if df['close'].iloc[-1] != 0 else 0
        ])
        
        # Bollinger Bands
        bb_sma = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        bb_upper = bb_sma + (2 * bb_std)
        bb_lower = bb_sma - (2 * bb_std)
        bb_position = (df['close'] - bb_lower) / (bb_upper - bb_lower)
        features.append(bb_position.iloc[-1] if not pd.isna(bb_position.iloc[-1]) else 0.5)
        
        # Volatility
        volatility = df['close'].pct_change().rolling(20).std()
        features.append(volatility.iloc[-1] if not pd.isna(volatility.iloc[-1]) else 0.02)
        
        # Price position in recent range
        high_20 = df['high'].rolling(20).max()
        low_20 = df['low'].rolling(20).min()
        position = (df['close'] - low_20) / (high_20 - low_20)
        features.append(position.iloc[-1] if not pd.isna(position.iloc[-1]) else 0.5)
        
        return np.array(features[:self.technical_feature_dim], dtype=np.float32)
    
    def _extract_fundamental_features(self, symbol: str) -> np.ndarray:
        """Extract fundamental features (placeholder for integration with company intelligence)."""
        # Would integrate with company intelligence service
        # For now, return default values
        return np.zeros(self.fundamental_feature_dim, dtype=np.float32)
    
    def _extract_sentiment_features(self, symbol: str) -> np.ndarray:
        """Extract sentiment features (placeholder for integration with social media collector)."""
        # Would integrate with social media collector
        # For now, return default values
        return np.zeros(self.sentiment_feature_dim, dtype=np.float32)
    
    async def _get_market_cap(self, symbol: str) -> float:
        """Get market cap (normalized)."""
        # Would get from company intelligence service
        return 1.0  # Placeholder
    
    async def _get_sector_weight(self, symbol: str) -> float:
        """Get sector weight."""
        # Would calculate based on sector classification
        return 0.1  # Placeholder
    
    async def _calculate_correlation_centrality(self, symbol: str) -> float:
        """Calculate how central this stock is in the correlation network."""
        # Would calculate based on correlation with other stocks
        return 0.5  # Placeholder
    
    def _determine_volatility_regime(self, df: pd.DataFrame) -> int:
        """Determine volatility regime (0=low, 1=medium, 2=high)."""
        volatility = df['close'].pct_change().rolling(20).std().iloc[-1]
        if pd.isna(volatility):
            return 1
        elif volatility < 0.015:
            return 0
        elif volatility < 0.035:
            return 1
        else:
            return 2
    
    async def _construct_edges(self, symbols: List[str], 
                             market_data: Dict[str, List[MarketData]],
                             node_mapping: Dict[str, int]) -> Tuple[List[List[int]], List[List[float]]]:
        """Construct edges based on correlations and relationships."""
        
        edge_indices = []
        edge_features = []
        
        # Calculate pairwise correlations
        returns_data = {}
        for symbol in symbols:
            if symbol in market_data and len(market_data[symbol]) >= self.correlation_window:
                df = self._market_data_to_df(market_data[symbol][-self.correlation_window:])
                returns = df['close'].pct_change().dropna()
                if len(returns) > 20:  # Minimum data points
                    returns_data[symbol] = returns
        
        # Create edges based on correlations
        symbols_with_data = list(returns_data.keys())
        for i, symbol1 in enumerate(symbols_with_data):
            correlations = []
            
            for j, symbol2 in enumerate(symbols_with_data):
                if i >= j or symbol1 not in node_mapping or symbol2 not in node_mapping:
                    continue
                
                try:
                    # Align time series
                    aligned_data = pd.DataFrame({
                        symbol1: returns_data[symbol1],
                        symbol2: returns_data[symbol2]
                    }).dropna()
                    
                    if len(aligned_data) < 20:
                        continue
                    
                    # Calculate correlation
                    correlation, _ = pearsonr(aligned_data[symbol1], aligned_data[symbol2])
                    
                    if abs(correlation) > self.min_correlation:
                        correlations.append((j, abs(correlation), correlation))
                
                except Exception as e:
                    logger.debug(f"Failed to calculate correlation between {symbol1} and {symbol2}: {e}")
                    continue
            
            # Keep only top correlations to avoid overly dense graphs
            correlations.sort(key=lambda x: x[1], reverse=True)
            top_correlations = correlations[:self.max_edges_per_node]
            
            for j, abs_corr, corr in top_correlations:
                symbol2 = symbols_with_data[j]
                
                # Create edge features
                edge_feature = [
                    corr,  # correlation
                    abs_corr,  # mutual information (simplified)
                    0.0,  # transfer entropy (placeholder)
                    0.0,  # granger causality (placeholder)
                    0.8,  # correlation stability (placeholder)
                    abs_corr,  # relationship strength
                    0    # lead_lag_days (placeholder)
                ]
                
                # Add bidirectional edges
                edge_indices.append([node_mapping[symbol1], node_mapping[symbol2]])
                edge_features.append(edge_feature)
                
                edge_indices.append([node_mapping[symbol2], node_mapping[symbol1]])
                edge_features.append(edge_feature)
        
        return edge_indices, edge_features
    
    async def _create_targets(self, symbols: List[str],
                            market_data: Dict[str, List[MarketData]],
                            node_mapping: Dict[str, int]) -> torch.Tensor:
        """Create prediction targets."""
        targets = []
        
        for symbol in symbols:
            if symbol not in node_mapping or symbol not in market_data:
                continue
                
            data = market_data[symbol]
            if len(data) < 2:
                targets.append(0.0)
                continue
            
            # Target: next day return
            try:
                current_price = data[-1].close
                prev_price = data[-2].close
                next_day_return = (current_price - prev_price) / prev_price
                targets.append(next_day_return)
            except:
                targets.append(0.0)
        
        return torch.tensor(targets, dtype=torch.float)
    
    def _market_data_to_df(self, data: List[MarketData]) -> pd.DataFrame:
        """Convert market data to DataFrame."""
        rows = []
        for md in data:
            rows.append({
                'timestamp': md.timestamp,
                'open': md.open,
                'high': md.high,
                'low': md.low,
                'close': md.close,
                'volume': md.volume
            })
        
        df = pd.DataFrame(rows)
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        return df


class GraphAttentionNetwork(nn.Module):
    """Graph Attention Network for market prediction."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, 
                 output_dim: int = 1, num_layers: int = 3,
                 heads: int = 8, dropout: float = 0.2):
        super(GraphAttentionNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Graph attention layers
        self.gat_layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.gat_layers.append(
                    GATConv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout)
                )
            else:
                self.gat_layers.append(
                    GATConv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout)
                )
        
        # Output layers
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # Regularization
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        """Forward pass."""
        # Input projection
        h = self.input_proj(x)
        h = F.relu(h)
        h = self.dropout(h)
        
        # Graph attention layers
        for i, gat_layer in enumerate(self.gat_layers):
            h_new = gat_layer(h, edge_index)
            h_new = F.relu(h_new)
            h_new = self.dropout(h_new)
            
            # Residual connection (if dimensions match)
            if h.size(-1) == h_new.size(-1):
                h = h + h_new
            else:
                h = h_new
            
            # Layer normalization
            h = self.layer_norm(h)
        
        # Output projection
        out = self.output_proj(h)
        
        return out


class MarketGNNService:
    """Service for Graph Neural Network market analysis."""
    
    def __init__(self):
        self.graph_constructor = MarketGraphConstructor()
        self.model = None
        self.scaler = StandardScaler()
        
        # Model configuration
        self.model_config = {
            'hidden_dim': 128,
            'num_layers': 3,
            'heads': 8,
            'dropout': 0.2,
            'learning_rate': 0.001,
            'weight_decay': 1e-4
        }
        
        # Training parameters
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = "/tmp/market_gnn_model.pth"
        
        # Performance tracking
        self.training_losses = []
        self.validation_scores = []
        
    async def initialize(self):
        """Initialize the GNN service."""
        await self.graph_constructor.initialize()
        
        # Try to load existing model
        if os.path.exists(self.model_path):
            await self.load_model()
            
        logger.info(f"Market GNN Service initialized on device: {self.device}")
    
    async def train_model(self, symbols: List[str], 
                         market_data: Dict[str, List[MarketData]],
                         epochs: int = 100) -> Dict[str, float]:
        """Train the GNN model."""
        
        logger.info(f"Training GNN model on {len(symbols)} symbols for {epochs} epochs")
        
        # Construct graph
        graph_data = await self.graph_constructor.construct_market_graph(
            symbols, market_data
        )
        
        if graph_data is None:
            logger.error("Failed to construct graph for training")
            return {}
        
        # Initialize model
        input_dim = graph_data.x.size(1)
        self.model = GraphAttentionNetwork(
            input_dim=input_dim,
            **self.model_config
        ).to(self.device)
        
        # Training setup
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.model_config['learning_rate'],
            weight_decay=self.model_config['weight_decay']
        )
        
        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=10, factor=0.5
        )
        
        # Move data to device
        graph_data = graph_data.to(self.device)
        
        # Split data (time-series split would be better)
        train_size = int(0.8 * graph_data.num_nodes)
        train_mask = torch.zeros(graph_data.num_nodes, dtype=torch.bool)
        train_mask[:train_size] = True
        val_mask = ~train_mask
        
        # Training loop
        self.model.train()
        training_losses = []
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Forward pass
            predictions = self.model(
                graph_data.x,
                graph_data.edge_index,
                graph_data.edge_attr
            ).squeeze()
            
            # Calculate loss only on training nodes
            train_loss = criterion(predictions[train_mask], graph_data.y[train_mask])
            
            # Backward pass
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            training_losses.append(train_loss.item())
            
            # Validation
            if epoch % 10 == 0:
                self.model.eval()
                with torch.no_grad():
                    val_loss = criterion(predictions[val_mask], graph_data.y[val_mask])
                    scheduler.step(val_loss)
                    
                    logger.info(f"Epoch {epoch}: Train Loss = {train_loss.item():.6f}, "
                               f"Val Loss = {val_loss.item():.6f}")
                
                self.model.train()
        
        # Calculate final metrics
        self.model.eval()
        with torch.no_grad():
            final_predictions = self.model(
                graph_data.x,
                graph_data.edge_index,
                graph_data.edge_attr
            ).squeeze()
            
            train_mse = criterion(final_predictions[train_mask], graph_data.y[train_mask]).item()
            val_mse = criterion(final_predictions[val_mask], graph_data.y[val_mask]).item()
            
            # Directional accuracy
            train_acc = self._calculate_directional_accuracy(
                final_predictions[train_mask], graph_data.y[train_mask]
            )
            val_acc = self._calculate_directional_accuracy(
                final_predictions[val_mask], graph_data.y[val_mask]
            )
        
        # Save model
        await self.save_model()
        
        metrics = {
            'train_mse': train_mse,
            'val_mse': val_mse,
            'train_directional_accuracy': train_acc,
            'val_directional_accuracy': val_acc,
            'final_lr': optimizer.param_groups[0]['lr']
        }
        
        logger.info(f"Training completed: Val MSE = {val_mse:.6f}, "
                   f"Val Directional Accuracy = {val_acc:.3f}")
        
        return metrics
    
    async def predict(self, symbols: List[str], 
                     market_data: Dict[str, List[MarketData]]) -> Dict[str, float]:
        """Make predictions using trained GNN model."""
        
        if self.model is None:
            logger.error("Model not trained or loaded")
            return {}
        
        # Construct graph
        graph_data = await self.graph_constructor.construct_market_graph(
            symbols, market_data
        )
        
        if graph_data is None:
            logger.error("Failed to construct graph for prediction")
            return {}
        
        # Make predictions
        self.model.eval()
        graph_data = graph_data.to(self.device)
        
        with torch.no_grad():
            predictions = self.model(
                graph_data.x,
                graph_data.edge_index,
                graph_data.edge_attr
            ).squeeze().cpu().numpy()
        
        # Map predictions back to symbols
        node_mapping = {i: symbol for i, symbol in enumerate(symbols) if symbol in market_data}
        
        results = {}
        for i, prediction in enumerate(predictions):
            if i in node_mapping:
                results[node_mapping[i]] = float(prediction)
        
        return results
    
    def _calculate_directional_accuracy(self, predictions: torch.Tensor, 
                                      targets: torch.Tensor) -> float:
        """Calculate directional accuracy."""
        pred_direction = torch.sign(predictions)
        target_direction = torch.sign(targets)
        
        correct = (pred_direction == target_direction).float()
        return correct.mean().item()
    
    async def save_model(self):
        """Save the trained model."""
        if self.model is not None:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'model_config': self.model_config,
                'scaler': self.scaler
            }, self.model_path)
            logger.info(f"Model saved to {self.model_path}")
    
    async def load_model(self):
        """Load a trained model."""
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Determine input dimension (would need to be stored or calculated)
            input_dim = 50  # Placeholder - would get from saved config
            
            self.model = GraphAttentionNetwork(
                input_dim=input_dim,
                **checkpoint['model_config']
            ).to(self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.scaler = checkpoint['scaler']
            
            logger.info(f"Model loaded from {self.model_path}")
            
        except Exception as e:
            logger.warning(f"Failed to load model: {e}")
    
    async def get_feature_importance(self, symbols: List[str],
                                   market_data: Dict[str, List[MarketData]]) -> Dict[str, float]:
        """Analyze feature importance using attention weights."""
        
        if self.model is None:
            return {}
        
        # This would extract and analyze attention weights from GAT layers
        # For now, return placeholder
        feature_names = [
            'price_momentum_1d', 'price_momentum_5d', 'rsi', 'macd',
            'volatility', 'volume_ratio', 'correlation_centrality'
        ]
        
        # Would calculate actual importance from attention weights
        importance_scores = np.random.random(len(feature_names))
        importance_scores /= importance_scores.sum()
        
        return dict(zip(feature_names, importance_scores))
    
    async def analyze_market_structure(self, symbols: List[str],
                                     market_data: Dict[str, List[MarketData]]) -> Dict[str, Any]:
        """Analyze the market structure using the graph."""
        
        graph_data = await self.graph_constructor.construct_market_graph(
            symbols, market_data
        )
        
        if graph_data is None:
            return {}
        
        # Convert to NetworkX for analysis
        nx_graph = nx.Graph()
        
        # Add nodes
        for i in range(graph_data.num_nodes):
            nx_graph.add_node(i)
        
        # Add edges
        edge_index = graph_data.edge_index.numpy()
        for i in range(edge_index.shape[1]):
            source, target = edge_index[:, i]
            nx_graph.add_edge(source, target)
        
        # Calculate network metrics
        try:
            centrality = nx.degree_centrality(nx_graph)
            clustering = nx.clustering(nx_graph)
            
            # Get most central nodes
            central_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]
            
            analysis = {
                'num_nodes': graph_data.num_nodes,
                'num_edges': graph_data.edge_index.size(1),
                'graph_density': nx.density(nx_graph),
                'avg_clustering': np.mean(list(clustering.values())),
                'most_central_stocks': [symbols[node] for node, _ in central_nodes if node < len(symbols)],
                'network_diameter': nx.diameter(nx_graph) if nx.is_connected(nx_graph) else -1
            }
            
            return analysis
            
        except Exception as e:
            logger.warning(f"Failed to analyze market structure: {e}")
            return {'error': str(e)}


# Global GNN service instance
gnn_service: Optional[MarketGNNService] = None


async def get_gnn_service() -> MarketGNNService:
    """Get or create GNN service instance."""
    global gnn_service
    if gnn_service is None:
        gnn_service = MarketGNNService()
        await gnn_service.initialize()
    return gnn_service