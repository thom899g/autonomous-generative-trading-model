"""
Real-time market data fetching with robust error handling and rate limiting.
Integrates with multiple exchanges via CCXT library.
"""
import ccxt
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import time
import logging
from dataclasses import dataclass
from trading_config import TradingConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MarketData:
    """Structured market data container"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    symbol: str
    exchange: str

class DataFetcher:
    """Robust market data fetcher with error handling and retry logic"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.exchange = self._initialize_exchange()
        self.rate_limit_delay = 1.0  # seconds between requests
        self.max_retries = 3
        self.request_count = 0
        self.last_request_time = time.time()
        
    def _initialize_exchange(self) -> ccxt.Exchange:
        """Initialize exchange connection with error handling"""
        try:
            exchange_class = getattr(ccxt, self.config.exchange_id)
            exchange = exchange_class({
                'enableRateLimit': True,
                'timeout': 30000,
                'verbose': False,
            })
            
            # Load markets to validate connection
            exchange.load_markets()
            logger.info(f"Successfully connected to {self.config.exchange_id}")
            return exchange
            
        except AttributeError:
            logger.error(f"Exchange {self.config.exchange_id} not found in CCXT")
            raise
        except Exception as e