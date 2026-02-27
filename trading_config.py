"""
Configuration management for the Autonomous Generative Trading System.
Centralizes all environment variables, exchange configurations, and model parameters.
"""
import os
import json
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud.firestore_v1.client import Client as FirestoreClient

@dataclass
class TradingConfig:
    """Central configuration dataclass with validation"""
    
    # Exchange Configuration
    exchange_id: str = "binance"
    trading_pairs: list = field(default_factory=lambda: ["BTC/USDT", "ETH/USDT"])
    timeframe: str = "1h"
    initial_balance: float = 10000.0
    
    # Model Parameters
    state_size: int = 50
    action_size: int = 3  # [BUY, SELL, HOLD]
    hidden_layers: list = field(default_factory=lambda: [256, 128, 64])
    learning_rate: float = 0.001
    discount_factor: float = 0.95
    exploration_rate: float = 0.1
    
    # Risk Management
    max_position_size: float = 0.1  # 10% of portfolio
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.05
    max_daily_loss: float = 0.03
    
    # Firebase Configuration
    firebase_credential_path: Optional[str] = None
    firestore_collection: str = "trading_state"
    
    # Logging
    log_level: str = "INFO"
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        self._validate_config()
        self._initialize_firebase()
    
    def _validate_config(self) -> None:
        """Validate all configuration parameters"""
        valid_exchanges = ["binance", "coinbase", "kraken"]
        if self.exchange_id not in valid_exchanges:
            raise ValueError(f"Exchange must be one of {valid_exchanges}")
        
        if self.initial_balance <= 0:
            raise ValueError("Initial balance must be positive")
        
        if not 0 <= self.exploration_rate <= 1:
            raise ValueError("Exploration rate must be between 0 and 1")
        
        if self.max_position_size <= 0 or self.max_position_size > 1:
            raise ValueError("Max position size must be between 0 and 1")
    
    def _initialize_firebase(self) -> None:
        """Initialize Firebase connection"""
        try:
            if self.firebase_credential_path and os.path.exists(self.firebase_credential_path):
                cred = credentials.Certificate(self.firebase_credential_path)
                firebase_admin.initialize_app(cred)
            else:
                # Try environment variable or default credentials
                firebase_admin.initialize_app()
            
            self.db: FirestoreClient = firestore.client()
            print("Firebase Firestore initialized successfully")
            
        except Exception as e:
            print(f"Warning: Firebase initialization failed: {e}")
            print("System will run with local state management only")
            self.db = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization"""
        return {
            key: value for key, value in self.__dict__.items() 
            if not key.startswith('_') and key != 'db'
        }
    
    def save_to_firestore(self, document_id: str = "config") -> None:
        """Save configuration to Firestore"""
        if self.db:
            doc_ref = self.db.collection(self.firestore_collection).document(document_id)
            doc_ref.set(self.to_dict())
            print(f"Configuration saved to Firestore document: {document_id}")

def load_config_from_env() -> TradingConfig:
    """Load configuration from environment variables with fallbacks"""
    
    # Exchange settings
    exchange_id = os.getenv("TRADING_EXCHANGE", "binance")
    trading_pairs = os.getenv("TRADING_PAIRS", "BTC/USDT,ETH/USDT").split(",")
    
    # Model parameters
    state_size = int(os.getenv("STATE_SIZE", "50"))
    action_size = int(os.getenv("ACTION_SIZE", "3"))
    
    # Risk parameters
    max_position_size = float(os.getenv("MAX_POSITION_SIZE", "0.1"))
    
    # Firebase
    firebase_credential_path = os.getenv("FIREBASE_CREDENTIAL_PATH")
    
    return TradingConfig(
        exchange_id=exchange_id,
        trading_pairs=trading_pairs,
        state_size=state_size,
        action_size=action_size,
        max_position_size=max_position_size,
        firebase_credential_path=firebase_credential_path
    )