#!/usr/bin/env python3
"""
Transfer Entropy Analysis for Market Causality Detection
PhD-level implementation to detect information flow between assets and predict lead-lag relationships.
Uses transfer entropy to quantify causality and build predictive signals.
"""

import asyncio
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# Information theory and statistical imports
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import KBinsDiscretizer
import networkx as nx
from concurrent.futures import ThreadPoolExecutor
import threading

from trading_common import MarketData, get_settings, get_logger
from trading_common.cache import get_trading_cache

logger = get_logger(__name__)
settings = get_settings()


@dataclass
class TransferEntropyResult:
    """Transfer entropy calculation result."""
    source_symbol: str
    target_symbol: str
    transfer_entropy: float       # Information flow from source to target
    normalized_te: float          # Normalized TE (0-1)
    statistical_significance: float  # p-value
    confidence_interval: Tuple[float, float]  # 95% CI
    optimal_lag: int              # Optimal lag in days
    mutual_information: float     # Mutual information between series
    effective_transfer_entropy: float  # TE - mutual information
    
    @property
    def is_significant(self) -> bool:
        """Check if transfer entropy is statistically significant."""
        return self.statistical_significance < 0.05
    
    @property
    def causality_strength(self) -> str:
        """Categorize causality strength."""
        if self.normalized_te > 0.1:
            return "strong"
        elif self.normalized_te > 0.05:
            return "moderate"
        elif self.normalized_te > 0.01:
            return "weak"
        else:
            return "negligible"


@dataclass
class CausalityNetwork:
    """Network representation of market causality."""
    timestamp: datetime
    nodes: List[str]  # Asset symbols
    edges: List[TransferEntropyResult]  # Causal relationships
    
    # Network metrics
    network_density: float
    average_clustering: float
    most_influential: List[str]    # Assets that influence others most
    most_responsive: List[str]     # Assets most influenced by others
    
    # Regime indicators
    information_flow_intensity: float  # Overall information flow in network
    market_efficiency: float          # How efficiently information spreads
    systemic_risk_indicator: float    # Risk of contagion
    
    def get_influencers(self, symbol: str) -> List[TransferEntropyResult]:
        """Get assets that influence the given symbol."""
        return [edge for edge in self.edges if edge.target_symbol == symbol and edge.is_significant]
    
    def get_influenced(self, symbol: str) -> List[TransferEntropyResult]:
        """Get assets influenced by the given symbol."""
        return [edge for edge in self.edges if edge.source_symbol == symbol and edge.is_significant]
    
    def get_predictive_signals(self, symbol: str) -> Dict[str, float]:
        """Get predictive signals for a symbol based on its influencers."""
        signals = {}
        
        influencers = self.get_influencers(symbol)
        for result in influencers:
            # Weight signal by transfer entropy strength and significance
            signal_strength = result.normalized_te * (1 - result.statistical_significance)
            signals[result.source_symbol] = signal_strength
        
        return signals


class TransferEntropyCalculator:
    """Calculates transfer entropy between time series."""
    
    def __init__(self):
        self.discretization_bins = 8    # Number of bins for discretization
        self.max_lag = 10              # Maximum lag to test (days)
        self.embedding_dimension = 3    # Embedding dimension for reconstruction
        self.bootstrap_samples = 1000   # Bootstrap samples for significance testing
        
        # Optimization parameters
        self.parallel_workers = 4
        self.cache_results = True
        
    def calculate_transfer_entropy(self, source: pd.Series, target: pd.Series, 
                                 lag: int = 1) -> float:
        """
        Calculate transfer entropy from source to target with given lag.
        
        TE(X->Y) = H(Y_t+1 | Y_t^k) - H(Y_t+1 | Y_t^k, X_t^l)
        
        Where H is conditional entropy, k is embedding dimension, l is lag.
        """
        
        if len(source) != len(target):
            min_len = min(len(source), len(target))
            source = source.iloc[:min_len]
            target = target.iloc[:min_len]
        
        if len(source) < 50:  # Need minimum data
            return 0.0
        
        try:
            # Discretize continuous data
            source_discrete = self._discretize_series(source)
            target_discrete = self._discretize_series(target)
            
            # Create embedding vectors
            target_past = self._create_embedding(target_discrete, self.embedding_dimension)
            target_future = target_discrete[self.embedding_dimension + lag:]
            source_past = self._create_embedding_with_lag(source_discrete, lag, len(target_future))
            
            # Ensure all arrays have same length
            min_len = min(len(target_past), len(target_future), len(source_past))
            target_past = target_past[:min_len]
            target_future = target_future[:min_len]
            source_past = source_past[:min_len]
            
            if min_len < 20:
                return 0.0
            
            # Calculate conditional entropies
            # H(Y_t+1 | Y_t^k)
            h_y_given_y_past = self._conditional_entropy(target_future, target_past)
            
            # H(Y_t+1 | Y_t^k, X_t^l)  
            combined_past = self._combine_embeddings(target_past, source_past)
            h_y_given_xy_past = self._conditional_entropy(target_future, combined_past)
            
            # Transfer entropy
            te = h_y_given_y_past - h_y_given_xy_past
            
            return max(te, 0.0)  # TE should be non-negative
            
        except Exception as e:
            logger.debug(f"Transfer entropy calculation failed: {e}")
            return 0.0
    
    def _discretize_series(self, series: pd.Series) -> np.ndarray:
        """Discretize continuous time series."""
        discretizer = KBinsDiscretizer(
            n_bins=self.discretization_bins, 
            encode='ordinal', 
            strategy='quantile'
        )
        
        values = series.values.reshape(-1, 1)
        discretized = discretizer.fit_transform(values).flatten().astype(int)
        
        return discretized
    
    def _create_embedding(self, series: np.ndarray, dimension: int) -> np.ndarray:
        """Create embedding vectors for time series."""
        if len(series) < dimension:
            return np.array([])
        
        embedding = np.array([
            series[i:i+dimension] for i in range(len(series) - dimension + 1)
        ])
        
        return embedding
    
    def _create_embedding_with_lag(self, series: np.ndarray, lag: int, target_len: int) -> np.ndarray:
        """Create embedding with specified lag."""
        if len(series) < lag:
            return np.array([])
        
        # Take values at specified lag
        lagged_series = series[:-lag] if lag > 0 else series
        
        # Create single-dimension embedding (could be extended to multi-dim)
        embedding = lagged_series[-target_len:].reshape(-1, 1)
        
        return embedding
    
    def _combine_embeddings(self, emb1: np.ndarray, emb2: np.ndarray) -> np.ndarray:
        """Combine two embedding matrices."""
        if len(emb1) == 0 or len(emb2) == 0:
            return np.array([])
        
        min_len = min(len(emb1), len(emb2))
        combined = np.hstack([emb1[:min_len], emb2[:min_len]])
        
        return combined
    
    def _conditional_entropy(self, target: np.ndarray, condition: np.ndarray) -> float:
        """Calculate conditional entropy H(Y|X)."""
        
        if len(target) == 0 or len(condition) == 0:
            return 0.0
        
        # Convert embeddings to single values for entropy calculation
        if condition.ndim > 1:
            # Hash multi-dimensional conditions to single values
            condition_hashed = np.array([
                hash(tuple(row)) % 10000 for row in condition
            ])
        else:
            condition_hashed = condition
        
        # Calculate joint and marginal probabilities
        joint_counts = defaultdict(int)
        condition_counts = defaultdict(int)
        
        for t, c in zip(target, condition_hashed):
            joint_counts[(t, c)] += 1
            condition_counts[c] += 1
        
        total_count = len(target)
        
        # Calculate conditional entropy
        conditional_entropy = 0.0
        
        for (t, c), joint_count in joint_counts.items():
            if condition_counts[c] > 0:
                p_joint = joint_count / total_count
                p_conditional = joint_count / condition_counts[c]
                
                if p_conditional > 0:
                    conditional_entropy -= p_joint * np.log2(p_conditional)
        
        return conditional_entropy
    
    def find_optimal_lag(self, source: pd.Series, target: pd.Series) -> int:
        """Find optimal lag that maximizes transfer entropy."""
        
        best_lag = 1
        best_te = 0.0
        
        for lag in range(1, min(self.max_lag + 1, len(source) // 10)):
            te = self.calculate_transfer_entropy(source, target, lag)
            
            if te > best_te:
                best_te = te
                best_lag = lag
        
        return best_lag
    
    def bootstrap_significance(self, source: pd.Series, target: pd.Series, 
                             observed_te: float, lag: int = 1) -> Tuple[float, Tuple[float, float]]:
        """Test statistical significance using bootstrap."""
        
        bootstrap_tes = []
        
        # Generate bootstrap samples by shuffling source series
        source_values = source.values
        
        for _ in range(min(self.bootstrap_samples, 500)):  # Limit for performance
            # Shuffle source to break causal relationship
            shuffled_source = np.random.permutation(source_values)
            shuffled_series = pd.Series(shuffled_source, index=source.index)
            
            # Calculate TE on shuffled data
            bootstrap_te = self.calculate_transfer_entropy(shuffled_series, target, lag)
            bootstrap_tes.append(bootstrap_te)
        
        # Calculate p-value
        p_value = np.mean(np.array(bootstrap_tes) >= observed_te)
        
        # Calculate confidence interval
        ci_lower = np.percentile(bootstrap_tes, 2.5)
        ci_upper = np.percentile(bootstrap_tes, 97.5)
        
        return p_value, (ci_lower, ci_upper)


class MarketCausalityAnalyzer:
    """Analyzes causality relationships in financial markets."""
    
    def __init__(self):
        self.calculator = TransferEntropyCalculator()
        self.cache = None
        
        # Analysis parameters
        self.min_data_length = 60      # Minimum data points for analysis
        self.analysis_window = 252     # 1 year rolling window
        self.update_frequency = 7      # Update every 7 days
        
        # Network construction parameters
        self.min_te_threshold = 0.01   # Minimum TE for edge creation
        self.significance_threshold = 0.05  # Statistical significance threshold
        
        # Current state
        self.current_network = None
        self.causality_cache = {}      # Cache of recent TE calculations
        self.last_update = None
        
        # Performance tracking
        self.calculations_performed = 0
        self.significant_relationships = 0
        self.network_updates = 0
        
    async def initialize(self):
        """Initialize causality analyzer."""
        self.cache = get_trading_cache()
        logger.info("Market Causality Analyzer initialized")
    
    async def analyze_market_causality(self, symbols: List[str], 
                                     market_data: Dict[str, List[MarketData]]) -> CausalityNetwork:
        """Analyze causality relationships between market assets."""
        
        logger.info(f"Analyzing causality for {len(symbols)} symbols")
        
        # Prepare return series for all symbols
        return_series = {}
        for symbol in symbols:
            if symbol in market_data and len(market_data[symbol]) >= self.min_data_length:
                df = self._market_data_to_df(market_data[symbol])
                returns = df['close'].pct_change().dropna()
                
                if len(returns) >= self.min_data_length:
                    return_series[symbol] = returns[-self.analysis_window:]  # Use recent data
        
        valid_symbols = list(return_series.keys())
        logger.info(f"Valid symbols for causality analysis: {len(valid_symbols)}")
        
        if len(valid_symbols) < 2:
            return self._create_empty_network()
        
        # Calculate pairwise transfer entropies
        causality_results = await self._calculate_pairwise_causality(return_series)
        
        # Build causality network
        network = await self._build_causality_network(valid_symbols, causality_results)
        
        # Cache results
        await self._cache_network_results(network)
        
        self.current_network = network
        self.last_update = datetime.utcnow()
        self.network_updates += 1
        
        logger.info(f"Built causality network with {len(network.edges)} significant relationships")
        
        return network
    
    async def _calculate_pairwise_causality(self, return_series: Dict[str, pd.Series]) -> List[TransferEntropyResult]:
        """Calculate transfer entropy for all pairs of assets."""
        
        symbols = list(return_series.keys())
        results = []
        
        # Use thread pool for parallel computation
        with ThreadPoolExecutor(max_workers=self.calculator.parallel_workers) as executor:
            
            # Submit all pairwise calculations
            future_to_pair = {}
            for source_symbol in symbols:
                for target_symbol in symbols:
                    if source_symbol != target_symbol:
                        future = executor.submit(
                            self._calculate_causality_pair,
                            source_symbol, target_symbol, 
                            return_series[source_symbol], return_series[target_symbol]
                        )
                        future_to_pair[future] = (source_symbol, target_symbol)
            
            # Collect results
            for future in future_to_pair:
                try:
                    result = future.result(timeout=30)  # 30 second timeout per calculation
                    if result and result.transfer_entropy > 0:
                        results.append(result)
                        self.calculations_performed += 1
                        
                        if result.is_significant:
                            self.significant_relationships += 1
                            
                except Exception as e:
                    source, target = future_to_pair[future]
                    logger.debug(f"Causality calculation failed for {source}->{target}: {e}")
        
        # Sort by transfer entropy strength
        results.sort(key=lambda x: x.transfer_entropy, reverse=True)
        
        return results
    
    def _calculate_causality_pair(self, source_symbol: str, target_symbol: str,
                                source_series: pd.Series, target_series: pd.Series) -> Optional[TransferEntropyResult]:
        """Calculate transfer entropy for a specific pair."""
        
        try:
            # Find optimal lag
            optimal_lag = self.calculator.find_optimal_lag(source_series, target_series)
            
            # Calculate transfer entropy
            te = self.calculator.calculate_transfer_entropy(source_series, target_series, optimal_lag)
            
            if te <= 0:
                return None
            
            # Calculate mutual information for comparison
            # Align series for mutual information calculation
            aligned_source = source_series.values[:-optimal_lag] if optimal_lag > 0 else source_series.values
            aligned_target = target_series.values[optimal_lag:] if optimal_lag > 0 else target_series.values
            
            min_len = min(len(aligned_source), len(aligned_target))
            aligned_source = aligned_source[:min_len]
            aligned_target = aligned_target[:min_len]
            
            if len(aligned_source) > 10:
                mi = mutual_info_regression(
                    aligned_source.reshape(-1, 1), 
                    aligned_target, 
                    random_state=42
                )[0]
            else:
                mi = 0.0
            
            # Normalize TE (approximate normalization)
            normalized_te = min(te / (np.log2(self.calculator.discretization_bins) + 1e-6), 1.0)
            
            # Effective transfer entropy (TE beyond mutual information)
            effective_te = max(te - mi, 0.0)
            
            # Statistical significance testing (simplified for performance)
            if te > 0.01:  # Only test significance for substantial TE
                p_value, confidence_interval = self.calculator.bootstrap_significance(
                    source_series, target_series, te, optimal_lag
                )
            else:
                p_value = 1.0  # Not significant
                confidence_interval = (0.0, te)
            
            return TransferEntropyResult(
                source_symbol=source_symbol,
                target_symbol=target_symbol,
                transfer_entropy=te,
                normalized_te=normalized_te,
                statistical_significance=p_value,
                confidence_interval=confidence_interval,
                optimal_lag=optimal_lag,
                mutual_information=mi,
                effective_transfer_entropy=effective_te
            )
            
        except Exception as e:
            logger.debug(f"Failed to calculate causality for {source_symbol}->{target_symbol}: {e}")
            return None
    
    async def _build_causality_network(self, symbols: List[str], 
                                     causality_results: List[TransferEntropyResult]) -> CausalityNetwork:
        """Build causality network from transfer entropy results."""
        
        # Filter significant relationships
        significant_edges = [
            result for result in causality_results 
            if result.transfer_entropy >= self.min_te_threshold and 
               result.statistical_significance <= self.significance_threshold
        ]
        
        # Build NetworkX graph for analysis
        G = nx.DiGraph()
        G.add_nodes_from(symbols)
        
        for edge in significant_edges:
            G.add_edge(
                edge.source_symbol, 
                edge.target_symbol, 
                weight=edge.transfer_entropy,
                te=edge.transfer_entropy,
                lag=edge.optimal_lag
            )
        
        # Calculate network metrics
        network_density = nx.density(G)
        
        # Clustering (convert to undirected for clustering)
        G_undirected = G.to_undirected()
        try:
            avg_clustering = nx.average_clustering(G_undirected)
        except:
            avg_clustering = 0.0
        
        # Find most influential nodes (high out-degree)
        out_degrees = dict(G.out_degree(weight='weight'))
        most_influential = sorted(out_degrees.keys(), key=lambda x: out_degrees[x], reverse=True)[:5]
        
        # Find most responsive nodes (high in-degree)
        in_degrees = dict(G.in_degree(weight='weight'))
        most_responsive = sorted(in_degrees.keys(), key=lambda x: in_degrees[x], reverse=True)[:5]
        
        # Calculate regime indicators
        information_flow_intensity = np.mean([edge.transfer_entropy for edge in significant_edges]) if significant_edges else 0.0
        
        # Market efficiency (how well information spreads)
        if len(significant_edges) > 0:
            max_possible_edges = len(symbols) * (len(symbols) - 1)
            market_efficiency = len(significant_edges) / max_possible_edges
        else:
            market_efficiency = 0.0
        
        # Systemic risk indicator (based on network connectivity)
        try:
            # Average shortest path length (lower = higher systemic risk)
            if nx.is_weakly_connected(G):
                avg_path_length = nx.average_shortest_path_length(G)
                systemic_risk_indicator = max(1.0 - avg_path_length / len(symbols), 0.0)
            else:
                systemic_risk_indicator = 0.3  # Moderate risk for disconnected network
        except:
            systemic_risk_indicator = 0.3
        
        return CausalityNetwork(
            timestamp=datetime.utcnow(),
            nodes=symbols,
            edges=significant_edges,
            network_density=network_density,
            average_clustering=avg_clustering,
            most_influential=most_influential,
            most_responsive=most_responsive,
            information_flow_intensity=information_flow_intensity,
            market_efficiency=market_efficiency,
            systemic_risk_indicator=systemic_risk_indicator
        )
    
    async def get_causality_signals(self, target_symbol: str) -> Dict[str, float]:
        """Get causality-based trading signals for a target symbol."""
        
        if not self.current_network:
            return {}
        
        signals = {}
        
        # Get assets that influence the target
        influencers = self.current_network.get_influencers(target_symbol)
        
        for result in influencers:
            # Create signal strength based on transfer entropy and lag
            base_strength = result.normalized_te
            
            # Adjust for lag (shorter lags are more actionable)
            lag_adjustment = 1.0 / (1.0 + result.optimal_lag * 0.1)
            
            # Adjust for significance
            significance_adjustment = 1.0 - result.statistical_significance
            
            signal_strength = base_strength * lag_adjustment * significance_adjustment
            
            signals[result.source_symbol] = {
                'strength': signal_strength,
                'lag_days': result.optimal_lag,
                'causality_type': result.causality_strength
            }
        
        return signals
    
    async def predict_next_period_returns(self, symbols: List[str],
                                        current_returns: Dict[str, float]) -> Dict[str, float]:
        """Predict next period returns using causality relationships."""
        
        if not self.current_network:
            return {symbol: 0.0 for symbol in symbols}
        
        predictions = {}
        
        for target_symbol in symbols:
            prediction = 0.0
            total_weight = 0.0
            
            # Get influencers
            influencers = self.current_network.get_influencers(target_symbol)
            
            for result in influencers:
                source_symbol = result.source_symbol
                
                if source_symbol in current_returns:
                    # Weight by transfer entropy strength
                    weight = result.normalized_te
                    
                    # Simple linear prediction (could be enhanced with ML)
                    influence = current_returns[source_symbol] * weight
                    
                    prediction += influence
                    total_weight += weight
            
            # Normalize by total weight
            if total_weight > 0:
                prediction = prediction / total_weight
            
            predictions[target_symbol] = prediction
        
        return predictions
    
    async def detect_information_cascades(self) -> List[Dict[str, Any]]:
        """Detect information cascades in the market."""
        
        if not self.current_network:
            return []
        
        cascades = []
        
        # Find chains of causality (A->B->C)
        for node1 in self.current_network.nodes:
            # Get direct influences from node1
            node1_influences = self.current_network.get_influenced(node1)
            
            for edge1 in node1_influences:
                node2 = edge1.target_symbol
                
                # Get influences from node2
                node2_influences = self.current_network.get_influenced(node2)
                
                for edge2 in node2_influences:
                    node3 = edge2.target_symbol
                    
                    # Found cascade: node1 -> node2 -> node3
                    if node3 != node1:  # Avoid cycles
                        cascade_strength = min(edge1.normalized_te, edge2.normalized_te)
                        
                        if cascade_strength > 0.02:  # Minimum strength threshold
                            cascades.append({
                                'cascade': [node1, node2, node3],
                                'strength': cascade_strength,
                                'total_lag': edge1.optimal_lag + edge2.optimal_lag,
                                'edge1_te': edge1.transfer_entropy,
                                'edge2_te': edge2.transfer_entropy
                            })
        
        # Sort by strength
        cascades.sort(key=lambda x: x['strength'], reverse=True)
        
        return cascades[:10]  # Return top 10 cascades
    
    def _create_empty_network(self) -> CausalityNetwork:
        """Create empty network when insufficient data."""
        return CausalityNetwork(
            timestamp=datetime.utcnow(),
            nodes=[],
            edges=[],
            network_density=0.0,
            average_clustering=0.0,
            most_influential=[],
            most_responsive=[],
            information_flow_intensity=0.0,
            market_efficiency=0.0,
            systemic_risk_indicator=0.0
        )
    
    def _market_data_to_df(self, data: List[MarketData]) -> pd.DataFrame:
        """Convert market data to DataFrame."""
        rows = []
        for md in data:
            rows.append({
                'timestamp': md.timestamp,
                'close': md.close
            })
        
        df = pd.DataFrame(rows)
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        return df
    
    async def _cache_network_results(self, network: CausalityNetwork):
        """Cache network analysis results."""
        
        if self.cache:
            # Cache network summary
            network_summary = {
                'timestamp': network.timestamp.isoformat(),
                'num_nodes': len(network.nodes),
                'num_edges': len(network.edges),
                'network_density': network.network_density,
                'most_influential': network.most_influential,
                'most_responsive': network.most_responsive,
                'information_flow_intensity': network.information_flow_intensity,
                'market_efficiency': network.market_efficiency,
                'systemic_risk_indicator': network.systemic_risk_indicator
            }
            
            await self.cache.set_json("causality_network_summary", network_summary, ttl=3600)
            
            # Cache significant relationships
            significant_edges = [
                {
                    'source': edge.source_symbol,
                    'target': edge.target_symbol,
                    'transfer_entropy': edge.transfer_entropy,
                    'normalized_te': edge.normalized_te,
                    'optimal_lag': edge.optimal_lag,
                    'significance': edge.statistical_significance,
                    'causality_strength': edge.causality_strength
                }
                for edge in network.edges
            ]
            
            await self.cache.set_json("significant_causality_relationships", significant_edges, ttl=3600)
    
    async def get_service_statistics(self) -> Dict[str, Any]:
        """Get service performance statistics."""
        
        network_stats = {}
        if self.current_network:
            network_stats = {
                'network_nodes': len(self.current_network.nodes),
                'network_edges': len(self.current_network.edges),
                'network_density': self.current_network.network_density,
                'information_flow_intensity': self.current_network.information_flow_intensity,
                'market_efficiency': self.current_network.market_efficiency,
                'systemic_risk_indicator': self.current_network.systemic_risk_indicator
            }
        
        return {
            'calculations_performed': self.calculations_performed,
            'significant_relationships': self.significant_relationships,
            'network_updates': self.network_updates,
            'last_update': self.last_update.isoformat() if self.last_update else None,
            **network_stats
        }


# Global causality analyzer instance
causality_analyzer: Optional[MarketCausalityAnalyzer] = None


async def get_causality_analyzer() -> MarketCausalityAnalyzer:
    """Get or create causality analyzer instance."""
    global causality_analyzer
    if causality_analyzer is None:
        causality_analyzer = MarketCausalityAnalyzer()
        await causality_analyzer.initialize()
    return causality_analyzer