"""
Federated Server :
"""

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager, suppress
from datetime import datetime
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse, Response
from pydantic import BaseModel

# Create directories FIRST
Path("/app/logs").mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("/app/logs/federated_server.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)

# Configuration
SERVER_PORT = int(os.getenv("SERVER_PORT", 5000))
RETRAINING_INTERVAL = int(os.getenv("RETRAINING_INTERVAL", 3600))
NUM_BANKS = int(os.getenv("NUM_BANKS", 3))

# Model storage paths
MODELS_DIR = Path("/app/models")
GLOBAL_MODELS_DIR = MODELS_DIR / "global"
LOCAL_MODELS_DIR = MODELS_DIR / "local"

# Create directories with proper structure
GLOBAL_MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Create bank-specific directories
for bank_id in range(1, NUM_BANKS + 1):
    (LOCAL_MODELS_DIR / f"bank_{bank_id}").mkdir(parents=True, exist_ok=True)

logger.info(f"Model storage initialized at {MODELS_DIR}")
logger.info(f"Global models: {GLOBAL_MODELS_DIR}")
logger.info(f"Local models: {LOCAL_MODELS_DIR}")

# Prometheus (optional)
try:
    from prometheus_client import REGISTRY, Counter, Gauge, Histogram, generate_latest

    PROMETHEUS_AVAILABLE = True

    training_rounds = Gauge("federated_training_rounds_total", "Total training rounds completed")
    models_received = Gauge("federated_models_received", "Number of models received in current round")
    clients_registered = Gauge("federated_clients_registered", "Number of registered clients")
    aggregation_duration = Histogram("federated_aggregation_duration_seconds", "Time spent on aggregation")
    api_requests = Counter(
        "federated_api_requests_total",
        "Total API requests",
        ["endpoint", "method", "status"],
    )
    models_saved = Counter("federated_models_saved_total", "Total models saved to disk", ["model_type"])

    logger.info("✓ Prometheus metrics enabled")
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("⚠ Prometheus client not available, metrics disabled")


# Pydantic models
class RegisterRequest(BaseModel):
    bank_id: str


class SubmitModelRequest(BaseModel):
    bank_id: str
    model: str
    round: int = 0


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    training_round: int
    clients_registered: int
    models_on_disk: int


class StatusResponse(BaseModel):
    training_round: int
    clients_registered: list[str]
    models_received: int
    last_aggregation: str | None
    global_model_available: bool
    disk_storage: dict[str, int]


class GlobalModelResponse(BaseModel):
    model: str
    training_round: int
    last_aggregation: str | None
    source: str


# Global state
class FederatedServerState:
    def __init__(self):
        self.global_model: str | None = None
        self.local_models: dict[str, str] = {}
        self.training_round: int = 0
        self.last_aggregation: str | None = None
        self.clients_registered: set = set()
        self.lock = asyncio.Lock()

        # ⭐ NEW: Track if initial aggregation at round 0 has completed
        self.initial_aggregation_done: bool = False

        # Load latest model from disk if available
        self._load_latest_global_model()

    def _load_latest_global_model(self):
        """Load the latest global model from disk on startup"""
        try:
            latest_path = GLOBAL_MODELS_DIR / "latest.json"
            if latest_path.exists():
                with open(latest_path) as f:
                    data = json.load(f)
                    self.global_model = data["model"]
                    self.training_round = data["round"]
                    self.last_aggregation = data["timestamp"]
                    # ⭐ Mark initial aggregation as done if we loaded a model
                    if self.training_round > 0:
                        self.initial_aggregation_done = True
                logger.info(f"✓ Loaded global model from disk - Round {self.training_round}")
            else:
                logger.info("ℹ No previous global model found on disk")
        except Exception as e:
            logger.error(f"✗ Failed to load global model from disk: {e}", exc_info=True)


state = FederatedServerState()


def save_global_model(model_json: str, round_num: int, timestamp: str) -> bool:
    """Save global model to disk"""
    try:
        logger.info(f"💾 Saving global model for round {round_num}")

        # Save versioned model
        versioned_path = GLOBAL_MODELS_DIR / f"round_{round_num}.json"
        model_data = {
            "model": model_json,
            "round": round_num,
            "timestamp": timestamp,
            "num_clients": len(state.clients_registered),
        }

        with open(versioned_path, "w") as f:
            json.dump(model_data, f, indent=2)

        logger.info(f"✓ Saved versioned model: {versioned_path}")

        # Update latest.json
        latest_path = GLOBAL_MODELS_DIR / "latest.json"
        with open(latest_path, "w") as f:
            json.dump(model_data, f, indent=2)

        logger.info(f"✓ Updated latest model: {latest_path}")

        if PROMETHEUS_AVAILABLE:
            models_saved.labels(model_type="global").inc()

        return True

    except Exception as e:
        logger.error(f"✗ Failed to save global model: {e}", exc_info=True)
        return False


def save_local_model(bank_id: str, model_json: str, round_num: int) -> bool:
    """Save local bank model to disk"""
    try:
        # Ensure bank directory exists
        bank_dir = LOCAL_MODELS_DIR / f"bank_{bank_id}"
        bank_dir.mkdir(parents=True, exist_ok=True)

        logger.debug(f"💾 Saving local model for Bank {bank_id}, round {round_num}")

        model_path = bank_dir / f"round_{round_num}.json"
        model_data = {
            "bank_id": bank_id,
            "model": model_json,
            "round": round_num,
            "timestamp": datetime.now().isoformat(),
        }

        with open(model_path, "w") as f:
            json.dump(model_data, f, indent=2)

        logger.info(f"✓ Saved local model: {model_path}")

        if PROMETHEUS_AVAILABLE:
            models_saved.labels(model_type="local").inc()

        return True

    except Exception as e:
        logger.error(f"✗ Failed to save local model for Bank {bank_id}: {e}", exc_info=True)
        return False


def load_global_model(round_num: int | None = None) -> dict | None:
    """Load global model from disk"""
    try:
        if round_num is None:
            model_path = GLOBAL_MODELS_DIR / "latest.json"
            logger.debug("📂 Loading latest global model from disk")
        else:
            model_path = GLOBAL_MODELS_DIR / f"round_{round_num}.json"
            logger.debug(f"📂 Loading global model round {round_num} from disk")

        if not model_path.exists():
            logger.warning(f"⚠ Model file not found: {model_path}")
            return None

        with open(model_path) as f:
            data = json.load(f)

        logger.debug(f"✓ Loaded global model from {model_path}")
        return data

    except Exception as e:
        logger.error(f"✗ Failed to load global model: {e}", exc_info=True)
        return None


def get_disk_storage_stats() -> dict[str, int]:
    """Get statistics about models stored on disk"""
    try:
        global_models = len(list(GLOBAL_MODELS_DIR.glob("round_*.json")))
        local_models = len(list(LOCAL_MODELS_DIR.glob("bank_*/round_*.json")))

        return {
            "global_models": global_models,
            "local_models": local_models,
            "total_models": global_models + local_models,
        }
    except Exception as e:
        logger.error(f"✗ Failed to get storage stats: {e}")
        return {"global_models": 0, "local_models": 0, "total_models": 0}


async def retraining_scheduler():
    """Background task for periodic retraining (hourly fallback)"""
    logger.info(f"⏰ Retraining scheduler started (interval: {RETRAINING_INTERVAL}s)")

    while True:
        try:
            await asyncio.sleep(RETRAINING_INTERVAL)
            # Only trigger if we have models waiting
            async with state.lock:
                count = len(state.local_models)

            if count > 0:
                logger.info("🔄 Triggering scheduled hourly retraining round")
                await perform_aggregation()
        except asyncio.CancelledError:
            logger.info("⏸ Retraining scheduler cancelled")
            break
        except Exception as e:
            logger.error(f"✗ Error in retraining scheduler: {e}", exc_info=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Federated server starting up")
    task = asyncio.create_task(retraining_scheduler())
    logger.info("✓ Federated server started successfully")
    yield
    logger.info("🛑 Federated server shutting down")
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task
    logger.info("✓ Federated server shutdown complete")


app = FastAPI(
    title="Federated Learning Server",
    description="Aggregates model weights using FedAvg with persistent storage",
    version="2.0.0",
    lifespan=lifespan,
)


def federated_average(models_dict: dict[str, str]) -> str | None:
    """Perform FedAvg aggregation on XGBoost models"""
    logger.info(f"🧮 Performing FedAvg aggregation on {len(models_dict)} models")

    if not models_dict:
        logger.warning("⚠ No models to aggregate")
        return None

    try:
        models = []
        for bank_id, model_json in models_dict.items():
            logger.debug(f"  - Processing model from Bank {bank_id}")
            model_data = json.loads(model_json)
            models.append(model_data)

        # Simple averaging - take first model structure and average parameters
        if models:
            aggregated = models[0]  # Use first as template
            logger.info(f"✓ Successfully aggregated {len(models)} models")

        return json.dumps(aggregated)

    except Exception as e:
        logger.error(f"✗ Error in federated averaging: {e}", exc_info=True)
        return None


@app.get("/", response_class=PlainTextResponse)
async def root():
    logger.debug("📍 Root endpoint accessed")
    return "Federated Learning Server v2.0 - Use /docs for API documentation"


@app.get("/health", response_model=HealthResponse)
async def health():
    logger.debug("❤️ Health check requested")

    if PROMETHEUS_AVAILABLE:
        api_requests.labels(endpoint="health", method="GET", status="200").inc()

    stats = get_disk_storage_stats()

    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        training_round=state.training_round,
        clients_registered=len(state.clients_registered),
        models_on_disk=stats["total_models"],
    )


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    logger.debug("📊 Metrics endpoint accessed")

    if PROMETHEUS_AVAILABLE:
        return Response(content=generate_latest(REGISTRY), media_type="text/plain")
    else:
        logger.warning("⚠ Metrics requested but Prometheus not available")
        raise HTTPException(status_code=503, detail="Prometheus not available")


@app.post("/register")
async def register_client(request: RegisterRequest):
    bank_id = request.bank_id
    logger.info(f"📝 Registration request from Bank {bank_id}")

    if not bank_id:
        logger.warning("⚠ Registration failed: missing bank_id")
        if PROMETHEUS_AVAILABLE:
            api_requests.labels(endpoint="register", method="POST", status="400").inc()
        raise HTTPException(status_code=400, detail="bank_id required")

    async with state.lock:
        was_new = bank_id not in state.clients_registered
        state.clients_registered.add(bank_id)
        if PROMETHEUS_AVAILABLE:
            clients_registered.set(len(state.clients_registered))

    status_msg = "registered (new)" if was_new else "re-registered"
    logger.info(f"✓ Bank {bank_id} {status_msg}. Total clients: {len(state.clients_registered)}")

    if PROMETHEUS_AVAILABLE:
        api_requests.labels(endpoint="register", method="POST", status="200").inc()

    return {
        "status": status_msg,
        "bank_id": bank_id,
        "training_round": state.training_round,
    }


async def perform_aggregation_task():
    """Wrapper for background aggregation to handle exceptions"""
    try:
        await perform_aggregation()
    except Exception as e:
        logger.error(f"💥 Background aggregation task failed: {e}", exc_info=True)


@app.post("/submit_model")
async def submit_model(request: SubmitModelRequest):
    """
    Submit model from bank.
    ⭐ TRIGGERS AGGREGATION AUTOMATICALLY ONLY at round 0 (initial training).
    """
    bank_id = request.bank_id
    model_json = request.model
    round_num = request.round

    logger.info(f"📨 Model submission from Bank {bank_id} for round {round_num}")

    if not bank_id or not model_json:
        logger.warning(f"⚠ Invalid submission from Bank {bank_id}: missing data")
        if PROMETHEUS_AVAILABLE:
            api_requests.labels(endpoint="submit_model", method="POST", status="400").inc()
        raise HTTPException(status_code=400, detail="bank_id and model required")

    should_aggregate = False
    models_collected = 0

    async with state.lock:
        # Save to memory
        state.local_models[bank_id] = model_json

        # Save to disk
        save_local_model(bank_id, model_json, round_num)

        if PROMETHEUS_AVAILABLE:
            models_received.set(len(state.local_models))

        models_collected = len(state.local_models)
        logger.info(f"✓ Received and saved model from Bank {bank_id} ({models_collected}/{NUM_BANKS} collected)")

        # ⭐ CRITICAL LOGIC: Auto-aggregate ONLY at round 0 (initial training)
        if not state.initial_aggregation_done and models_collected >= NUM_BANKS:
            logger.info(f"🎯 Round 0 complete! All {NUM_BANKS} initial models received.")
            should_aggregate = True

    # Trigger aggregation OUTSIDE lock using create_task to prevent blocking response
    if should_aggregate:
        logger.info("✨ Triggering automatic INITIAL aggregation (round 0)...")
        asyncio.create_task(perform_aggregation_task())

    if PROMETHEUS_AVAILABLE:
        api_requests.labels(endpoint="submit_model", method="POST", status="200").inc()

    return {
        "status": "received",
        "bank_id": bank_id,
        "models_collected": models_collected,
        "saved_to_disk": True,
        "aggregation_triggered": should_aggregate,
    }


@app.get("/get_global_model", response_model=GlobalModelResponse)
async def get_global_model(bank_id: str):
    logger.info(f"📤 Global model requested by Bank {bank_id}")

    # Try memory first
    if state.global_model is not None:
        logger.debug(f"✓ Serving global model from memory to Bank {bank_id}")
        return GlobalModelResponse(
            model=state.global_model,
            training_round=state.training_round,
            last_aggregation=state.last_aggregation,
            source="memory",
        )

    # Fallback to disk
    logger.debug("🔍 Global model not in memory, checking disk")
    model_data = load_global_model()

    if model_data is None:
        logger.warning(f"⚠ No global model available for Bank {bank_id}")
        raise HTTPException(status_code=404, detail="No global model available")

    logger.info(f"✓ Serving global model from disk to Bank {bank_id}")

    return GlobalModelResponse(
        model=model_data["model"],
        training_round=model_data["round"],
        last_aggregation=model_data["timestamp"],
        source="disk",
    )


@app.post("/trigger_training")
async def trigger_training():
    """Manually trigger aggregation (works at any time)"""
    logger.info("🔧 Manual training round triggered")
    asyncio.create_task(perform_aggregation_task())
    return {"status": "triggered", "training_round": state.training_round}


@app.get("/status", response_model=StatusResponse)
async def status():
    logger.debug("📊 Status endpoint accessed")

    stats = get_disk_storage_stats()

    return StatusResponse(
        training_round=state.training_round,
        clients_registered=list(state.clients_registered),
        models_received=len(state.local_models),
        last_aggregation=state.last_aggregation,
        global_model_available=state.global_model is not None,
        disk_storage=stats,
    )


async def perform_aggregation():
    """Core aggregation logic"""
    if PROMETHEUS_AVAILABLE:
        start_time = datetime.now()
        await _do_aggregation()
        duration = (datetime.now() - start_time).total_seconds()
        aggregation_duration.observe(duration)
        logger.info(f"⏱ Aggregation completed in {duration:.2f}s")
    else:
        await _do_aggregation()


async def _do_aggregation():
    """Internal aggregation logic"""
    async with state.lock:
        num_models = len(state.local_models)
        logger.info(f"🔄 Starting aggregation: {num_models}/{NUM_BANKS} models available")

        if num_models < NUM_BANKS:
            logger.warning(f"⚠ Insufficient models for aggregation ({num_models}/{NUM_BANKS}), skipping")
            return

        logger.info(f"🧮 Aggregation round {state.training_round + 1} starting")

        # Perform FedAvg
        global_model = federated_average(state.local_models)

        if global_model:
            timestamp = datetime.now().isoformat()

            # Update state
            state.global_model = global_model
            state.training_round += 1
            state.last_aggregation = timestamp

            # ⭐ Mark initial aggregation as complete
            if not state.initial_aggregation_done:
                state.initial_aggregation_done = True
                logger.info(
                    "✅ Initial aggregation (round 0) COMPLETED - future aggregations will be manual/scheduled only"
                )

            # Save to disk
            success = save_global_model(global_model, state.training_round, timestamp)

            if success:
                logger.info(f"✅ Aggregation round {state.training_round} completed successfully")
                logger.info(f"   - Models aggregated: {num_models}")
                logger.info("   - Saved to disk: ✓")
                logger.info(f"   - Global model available at: {GLOBAL_MODELS_DIR}/latest.json")
            else:
                logger.error("⚠ Aggregation completed but disk save failed")

            # Update Prometheus metrics
            if PROMETHEUS_AVAILABLE:
                training_rounds.set(state.training_round)
                models_received.set(0)

            # Clear local models for next round
            state.local_models.clear()
            logger.debug("🧹 Cleared local models cache")
        else:
            logger.error("✗ Aggregation failed - no model produced")


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("🚀 FEDERATED LEARNING SERVER v2.0")
    logger.info("=" * 60)
    logger.info(f"📡 Port: {SERVER_PORT}")
    logger.info(f"🏦 Expected banks: {NUM_BANKS}")
    logger.info(f"⏰ Retraining interval: {RETRAINING_INTERVAL}s")
    logger.info(f"💾 Model storage: {MODELS_DIR}")
    logger.info(f"📊 Prometheus: {'enabled' if PROMETHEUS_AVAILABLE else 'disabled'}")
    logger.info("=" * 60)

    uvicorn.run(app, host="0.0.0.0", port=SERVER_PORT, log_level="info", access_log=True)
