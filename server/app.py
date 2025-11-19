#!/usr/bin/env python3
"""
FastAPI server for NinjaTrader integration (STEP 6).

This server exposes HTTP endpoints that NinjaTrader can call to:
1. Get trading signals from the trained model
2. Log trade executions for monitoring

Usage:
    python server/app.py

Then from NinjaTrader, POST to:
    http://localhost:8000/signal
    http://localhost:8000/trade_log
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from contextlib import asynccontextmanager
import numpy as np
import yaml
from datetime import datetime
from pathlib import Path
import logging

# Import our model components
import sys
sys.path.append(str(Path(__file__).parent.parent))

from model.data_utils import build_gamma
from model.trainer import build_segments
from model.clustering import assign_to_nearest_centroid, compute_centroids
from model.symplectic_model import estimate_global_kappa, estimate_kappa_per_cluster
from model.signal_api import SymplecticGlobalModel, SymplecticUltrametricModel

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize model state (will be loaded in lifespan)
model_state = None

# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events."""
    global model_state
    # Startup
    try:
        model_state = ModelState()
        model_state.load_model()
        logger.info("Server started successfully")
        yield
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    finally:
        # Shutdown (cleanup if needed)
        logger.info("Server shutting down")

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Ultrametric-Symplectic Trading API",
    description="REST API for NinjaTrader integration",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Pydantic Models (Request/Response schemas)
# ============================================================================

class Bar(BaseModel):
    """Single OHLCV bar."""
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float


class AccountSnapshot(BaseModel):
    """Optional account state from NinjaTrader."""
    account_id: str
    cash_value: float
    realized_pnl: float
    unrealized_pnl: float
    total_buying_power: float
    position_quantity: int
    position_avg_price: float


class SignalRequest(BaseModel):
    """Request for trading signal."""
    bars: List[Bar] = Field(..., min_items=10, description="Last K bars (minimum 10)")
    account: Optional[AccountSnapshot] = None
    instrument: str = Field(default="NQ", description="Instrument symbol")


class SignalResponse(BaseModel):
    """Trading signal response."""
    direction: int = Field(..., description="Direction: -1 (short), 0 (flat), 1 (long)")
    size_factor: float = Field(..., ge=0.0, le=1.0, description="Position size factor 0-1")
    model_used: str = Field(..., description="Which model generated signal")
    timestamp: str = Field(..., description="Server timestamp")
    metadata: Optional[Dict] = None


class TradeLog(BaseModel):
    """Trade execution log from NinjaTrader."""
    timestamp: str
    instrument: str
    side: str  # "Long" or "Short"
    quantity: int
    price: float
    realized_pnl: float
    strategy: str


class TradeLogResponse(BaseModel):
    """Acknowledgement of trade log."""
    status: str
    message: str
    timestamp: str


# ============================================================================
# Global Model State
# ============================================================================

class ModelState:
    """Holds loaded model state."""
    def __init__(self):
        self.config = None
        self.model = None
        self.model_type = "global"  # "global" or "hybrid"
        self.K = 10
        self.loaded = False

    def load_model(self, config_path: str = "configs/config.yaml", use_hybrid: bool = False):
        """Load trained model from config and data."""
        logger.info(f"Loading model (hybrid={use_hybrid})...")

        # Resolve config path relative to project root
        if not Path(config_path).exists():
            # Try relative to this file's parent directory (project root)
            project_root = Path(__file__).parent.parent
            config_path = project_root / config_path

        logger.info(f"Loading config from: {config_path}")

        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.K = self.config['segments']['K']
        encoding = self.config['symplectic']['encoding']

        if use_hybrid:
            # Load hybrid model (requires training data)
            # For now, we'll use global model in production
            # In a real deployment, you'd load pre-trained centroids and kappa values
            logger.warning("Hybrid model requires pre-trained state. Using global model instead.")
            use_hybrid = False

        if not use_hybrid:
            # Load global symplectic model
            # In production, you'd load pre-trained kappa from file
            # For now, use default
            kappa_global = 0.0100  # From training results
            self.model = SymplecticGlobalModel(self.config, kappa_global, encoding=encoding)
            self.model_type = "global"
            logger.info(f"Global symplectic model loaded (κ={kappa_global:.4f})")

        self.loaded = True
        logger.info("Model loaded successfully")


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "online",
        "service": "Ultrametric-Symplectic Trading API",
        "version": "1.0.0",
        "model_loaded": model_state.loaded,
        "model_type": model_state.model_type
    }


@app.get("/health")
async def health():
    """Detailed health check."""
    return {
        "status": "healthy" if model_state.loaded else "unhealthy",
        "model_loaded": model_state.loaded,
        "model_type": model_state.model_type,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/signal", response_model=SignalResponse)
async def get_signal(request: SignalRequest):
    """
    Generate trading signal from bar data.

    This is the main endpoint that NinjaTrader calls on each bar close.
    """
    try:
        if not model_state.loaded:
            raise HTTPException(status_code=503, detail="Model not loaded")

        K = model_state.K

        # Validate we have enough bars
        if len(request.bars) < K:
            raise HTTPException(
                status_code=400,
                detail=f"Need at least {K} bars, got {len(request.bars)}"
            )

        # Extract data from bars (use last K bars)
        bars = request.bars[-K:]
        closes = np.array([b.close for b in bars])
        volumes = np.array([b.volume for b in bars])

        # Compute log prices
        p = np.log(closes)

        # Normalize and smooth volume (simplified - no rolling mean)
        # In production, you'd maintain state or use a simpler normalization
        v = volumes / (np.mean(volumes) + 1e-8)

        # Apply simple EMA smoothing
        ema_period = model_state.config['volume']['ema_period']
        alpha = 2.0 / (ema_period + 1.0)
        v_smooth = np.zeros_like(v)
        v_smooth[0] = v[0]
        for i in range(1, len(v)):
            v_smooth[i] = alpha * v[i] + (1 - alpha) * v_smooth[i-1]

        # Build segment
        gamma = np.stack([p, v_smooth], axis=1)  # Shape (K, 2)

        # Get signal from model
        signal = model_state.model.get_signal(gamma)

        # Prepare response
        response = SignalResponse(
            direction=signal["direction"],
            size_factor=signal["size_factor"],
            model_used=model_state.model_type,
            timestamp=datetime.now().isoformat(),
            metadata={
                "bars_received": len(request.bars),
                "bars_used": K,
                "instrument": request.instrument,
                "last_close": float(closes[-1]),
                "last_volume": float(volumes[-1])
            }
        )

        logger.info(f"Signal generated: direction={response.direction}, size={response.size_factor:.2f}")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating signal: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.post("/trade_log", response_model=TradeLogResponse)
async def log_trade(trade: TradeLog):
    """
    Log trade execution from NinjaTrader.

    This endpoint allows NinjaTrader to report fills back to Python
    for monitoring and analysis.
    """
    try:
        logger.info(
            f"Trade logged: {trade.side} {trade.quantity} {trade.instrument} "
            f"@ {trade.price} (PnL: {trade.realized_pnl:.2f})"
        )

        # In production, you'd save this to a database or file
        # For now, just log it

        return TradeLogResponse(
            status="success",
            message="Trade logged successfully",
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        logger.error(f"Error logging trade: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.post("/reload_model")
async def reload_model():
    """Reload the model (useful for updates without restart)."""
    try:
        model_state.load_model()
        return {
            "status": "success",
            "message": "Model reloaded successfully",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error reloading model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reload model: {str(e)}")


# ============================================================================
# Run Server
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    print("=" * 80)
    print("Ultrametric-Symplectic Trading API Server")
    print("=" * 80)
    print("\nStarting server on http://localhost:8000")
    print("\nEndpoints:")
    print("  GET  /           - Health check")
    print("  GET  /health     - Detailed health status")
    print("  POST /signal     - Get trading signal")
    print("  POST /trade_log  - Log trade execution")
    print("  POST /reload_model - Reload model")
    print("\nDocs: http://localhost:8000/docs")
    print("=" * 80)
    print()

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
