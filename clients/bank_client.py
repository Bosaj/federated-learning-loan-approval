"""
Bank Client :
"""

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from kafka import KafkaConsumer

# Create logs directory FIRST
Path("/app/logs").mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("/app/logs/bank_client.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)


class BankClient:
    def __init__(self, bank_id):
        self.bank_id = str(bank_id)
        self.data_path = f"/app/data/bank{bank_id}/bank{bank_id}_dataset.csv"
        self.server_url = f"http://{os.getenv('SERVER_HOST', 'federated-server')}:{os.getenv('SERVER_PORT', '5000')}"
        self.kafka_bootstrap = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
        self.kafka_topic = f"bank{bank_id}_txn"

        # Model storage paths
        self.models_dir = Path("/app/models")
        self.local_models_dir = self.models_dir / "local" / f"bank_{self.bank_id}"
        self.global_models_dir = self.models_dir / "global"

        # ⭐ Metrics storage path (for Streamlit)
        self.metrics_path = Path(f"/app/logs/bank{bank_id}_metrics.json")
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)

        # Create directories
        self.local_models_dir.mkdir(parents=True, exist_ok=True)

        self.local_model = None
        self.global_model = None
        self.training_round = 0

        self.metrics = {
            "bank_id": self.bank_id,
            "accuracy": 0.0,
            "auc": 0.0,
            "f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "predictions_made": 0,
            "loans_approved": 0,
            "loans_denied": 0,
            "total_transactions": 0,
            "dataset_size": 0,
            "last_updated": datetime.now().isoformat(),
        }

        # Track transactions for batching
        self.transaction_batch = []
        self.batch_size = 100

        # Track retraining timing
        self.last_training_time = time.time()

        logger.info("=" * 60)
        logger.info(f"🏦 Bank {bank_id} Client Initialized (Loan Approval)")
        logger.info("=" * 60)
        logger.info(f"📁 Data path: {self.data_path}")
        logger.info(f"🌐 Server URL: {self.server_url}")
        logger.info(f"📨 Kafka topic: {self.kafka_topic}")
        logger.info(f"💾 Local models: {self.local_models_dir}")
        logger.info(f"📊 Metrics file: {self.metrics_path}")
        logger.info("=" * 60)

    def save_metrics(self):
        """Save metrics to JSON for Streamlit"""
        try:
            self.metrics["last_updated"] = datetime.now().isoformat()
            with open(self.metrics_path, "w") as f:
                json.dump(self.metrics, f, indent=2)
        except Exception as e:
            logger.error(f"✗ Failed to save metrics: {e}")

    def register_with_server(self):
        """Register this bank with federated server"""
        logger.info(f"📝 Registering Bank {self.bank_id} with federated server")

        try:
            response = requests.post(
                f"{self.server_url}/register",
                json={"bank_id": self.bank_id},
                timeout=10,
            )
            if response.status_code == 200:
                logger.info(f"✓ Bank {self.bank_id} successfully registered")
                return True
            else:
                logger.error(f"✗ Registration failed: {response.text}")
                return False
        except Exception as e:
            logger.error(f"✗ Registration error: {e}", exc_info=True)
            return False

    def load_data(self):
        """Load local data partition"""
        logger.debug(f"📂 Loading data from {self.data_path}")

        try:
            if not os.path.exists(self.data_path):
                logger.warning(f"⚠ Data file not found: {self.data_path}")
                return None

            df = pd.read_csv(self.data_path)
            logger.info(f"✓ Loaded {len(df)} loan applications for Bank {self.bank_id}")

            # Update metrics
            self.metrics["dataset_size"] = len(df)
            self.save_metrics()

            if "loan_status" in df.columns:
                approved_count = df["loan_status"].sum()
                approval_rate = (approved_count / len(df)) * 100
                logger.info(f"   - Approved loans: {approved_count} ({approval_rate:.2f}%)")

            return df
        except Exception as e:
            logger.error(f"✗ Error loading data: {e}", exc_info=True)
            return None

    def save_local_model(self, model, round_num):
        """Save local model to disk"""
        try:
            logger.debug(f"💾 Saving local model for round {round_num}")

            model_path = self.local_models_dir / f"round_{round_num}.json"

            # Serialize model
            model_json = model.get_booster().save_raw(raw_format="json").decode("utf-8")

            model_data = {
                "bank_id": self.bank_id,
                "model": model_json,
                "round": round_num,
                "timestamp": datetime.now().isoformat(),
                "metrics": self.metrics.copy(),
            }

            with open(model_path, "w") as f:
                json.dump(model_data, f, indent=2)

            # Update latest
            latest_path = self.local_models_dir / "latest.json"
            with open(latest_path, "w") as f:
                json.dump(model_data, f, indent=2)

            logger.info(f"✓ Local model saved: {model_path}")
            return True

        except Exception as e:
            logger.error(f"✗ Failed to save local model: {e}", exc_info=True)
            return False

    def train_local_model(self):
        """Train XGBoost model for loan approval prediction"""
        logger.info(f"🎯 Starting local model training for Bank {self.bank_id}")
        start_time = time.time()

        try:
            df = self.load_data()
            if df is None or len(df) < 10:
                logger.warning("⚠ Insufficient data for training")
                return None

            # Remove duplicates
            original_size = len(df)
            df = df.drop_duplicates()
            duplicates_removed = original_size - len(df)
            if duplicates_removed > 0:
                logger.info(f"🧹 Removed {duplicates_removed} duplicate loan applications")

            # Prepare features and labels
            X = df.drop("loan_status", axis=1) if "loan_status" in df.columns else df
            y = df["loan_status"] if "loan_status" in df.columns else None

            if y is None:
                logger.error("✗ No 'loan_status' column found in data")
                return None

            # ⭐ CRITICAL: Remove rows with NaN in loan_status
            nan_count = y.isna().sum()
            if nan_count > 0:
                logger.warning(f"⚠ Found {nan_count} rows with missing loan_status - removing them")
                valid_indices = ~y.isna()
                X = X[valid_indices]
                y = y[valid_indices]
                logger.info(f"✓ Cleaned dataset: {len(y)} valid samples remaining")

            # Log class distribution
            class_counts = y.value_counts()
            logger.info("📊 Loan Status Distribution:")
            logger.info(f"   - Denied (0): {class_counts.get(0, 0)}")
            logger.info(f"   - Approved (1): {class_counts.get(1, 0)}")

            # Train/test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.82, random_state=58)

            logger.info("📊 Data Split:")
            logger.info(f"   - Training samples: {len(X_train)}")
            logger.info(f"   - Test samples: {len(X_test)}")

            # Training parameters
            params = {
                "n_estimators": 150,
                "learning_rate": 0.1,
                "max_depth": 5,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "eval_metric": "logloss",
                "random_state": 42,
                "n_jobs": -1,
                "tree_method": "hist",
            }

            logger.debug(f"🔧 Training with params: {params}")

            self.local_model = xgb.XGBClassifier(**params)
            self.local_model.fit(
                X_train,
                y_train,
                eval_set=[(X_test, y_test)],
                early_stopping_rounds=20,
                verbose=False,
            )

            training_time = time.time() - start_time
            logger.info(f"⏱ Training completed in {training_time:.2f}s")

            # Evaluate
            y_pred_proba = self.local_model.predict_proba(X_test)[:, 1]
            y_pred = self.local_model.predict(X_test)

            self.metrics["accuracy"] = float(accuracy_score(y_test, y_pred))
            self.metrics["auc"] = float(roc_auc_score(y_test, y_pred_proba))
            self.metrics["f1"] = float(f1_score(y_test, y_pred))
            self.metrics["precision"] = float(precision_score(y_test, y_pred, zero_division=0))
            self.metrics["recall"] = float(recall_score(y_test, y_pred, zero_division=0))

            logger.info("📈 Model Performance on Test Set:")
            logger.info(f"   - Accuracy: {self.metrics['accuracy']:.4f}")
            logger.info(f"   - AUC: {self.metrics['auc']:.4f}")
            logger.info(f"   - F1 Score: {self.metrics['f1']:.4f}")
            logger.info(f"   - Precision: {self.metrics['precision']:.4f}")
            logger.info(f"   - Recall: {self.metrics['recall']:.4f}")

            # Save model and metrics
            self.save_local_model(self.local_model, self.training_round)
            self.save_metrics()

            # Reset timing
            self.last_training_time = time.time()

            return self.local_model

        except Exception as e:
            logger.error(f"✗ Training error: {e}", exc_info=True)
            return None

    def submit_model_to_server(self):
        """Send local model to federated server"""
        logger.info("📤 Submitting model to federated server")

        try:
            if self.local_model is None:
                logger.warning("⚠ No local model to submit")
                return False

            # Serialize model
            model_json = self.local_model.get_booster().save_raw(raw_format="json").decode("utf-8")

            logger.debug(f"📦 Model size: {len(model_json)} bytes")

            response = requests.post(
                f"{self.server_url}/submit_model",
                json={
                    "bank_id": self.bank_id,
                    "model": model_json,
                    "round": self.training_round,
                },
                timeout=30,
            )

            if response.status_code == 200:
                data = response.json()
                logger.info("✓ Model submitted successfully")
                logger.info(f"   - Models collected: {data.get('models_collected', '?')}")

                if data.get("aggregation_triggered"):
                    logger.info("   - ✨ Server triggered aggregation!")

                return True
            else:
                logger.error(f"✗ Model submission failed: {response.text}")
                return False

        except Exception as e:
            logger.error(f"✗ Model submission error: {e}", exc_info=True)
            return False

    def fetch_global_model(self):
        """Fetch global model from server"""
        logger.info("📥 Fetching global model from server")

        max_retries = 3
        retry_delay = 5

        for attempt in range(max_retries):
            try:
                response = requests.get(
                    f"{self.server_url}/get_global_model",
                    params={"bank_id": self.bank_id},
                    timeout=30,
                )

                if response.status_code == 200:
                    data = response.json()
                    model_json = data["model"]

                    # Load as Booster
                    booster = xgb.Booster()
                    booster.load_model(bytearray(model_json, "utf-8"))

                    self.global_model = booster
                    self.training_round = data["training_round"]
                    source = data.get("source", "unknown")

                    logger.info(f"✓ Loaded global model (round {self.training_round}, source: {source})")
                    return True
                else:
                    logger.warning(
                        f"⚠ Global model not available (attempt {attempt + 1}/{max_retries}): {response.text}"
                    )

                    if attempt < max_retries - 1:
                        logger.info(f"⏳ Retrying in {retry_delay}s...")
                        time.sleep(retry_delay)

            except Exception as e:
                logger.error(f"✗ Error fetching global model (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)

        logger.warning("⚠ Failed to fetch global model after all retries")
        return False

    def predict_loan_approval(self, application):
        """
        ⭐ UPDATED: Predict loan approval using available model
        - Extracts and stores the actual loan_status for validation
        - Removes loan_status before prediction (so model doesn't see it)
        - Returns prediction, probability, and actual status
        """
        try:
            # Use global model if available, otherwise local
            model = self.global_model if self.global_model else self.local_model

            if model is None:
                logger.warning("⚠ No model available for prediction")
                return None, 0.0, None

            # ⭐ CRITICAL: Extract actual loan_status BEFORE creating features
            actual_status = application.get("loan_status", None)

            # Prepare features (create copy to avoid modifying original)
            features = pd.DataFrame([application.copy()])

            # ⭐ Remove loan_status from features (model shouldn't see it)
            if "loan_status" in features.columns:
                features = features.drop("loan_status", axis=1)

            # Predict
            if isinstance(model, xgb.Booster):
                dmatrix = xgb.DMatrix(features)
                pred_proba = model.predict(dmatrix)[0]
            else:
                pred_proba = model.predict_proba(features)[0, 1]

            pred_status = 1 if pred_proba > 0.5 else 0

            # Update metrics
            self.metrics["predictions_made"] += 1
            self.metrics["total_transactions"] += 1

            if pred_status == 1:
                self.metrics["loans_approved"] += 1
            else:
                self.metrics["loans_denied"] += 1

            # Save metrics every 10 predictions
            if self.metrics["predictions_made"] % 10 == 0:
                self.save_metrics()

            # Log predictions with validation
            decision = "APPROVED" if pred_status == 1 else "DENIED"
            logger.info(f"📋 Loan Decision: {decision} (confidence: {pred_proba:.4f})")

            if actual_status is not None:
                actual_decision = "APPROVED" if actual_status == 1 else "DENIED"
                logger.info(f"   - Actual: {actual_decision}")
                if actual_status != pred_status:
                    logger.warning("   ⚠ MISMATCH!")

            return pred_status, pred_proba, actual_status

        except Exception as e:
            logger.error(f"✗ Prediction error: {e}", exc_info=True)
            return None, 0.0, None

    def append_transactions_to_dataset(self):
        """
        ⭐ UPDATED: Append batch of transactions to dataset
        Transactions now include loan_status (kept from Kafka message)
        """
        try:
            if not self.transaction_batch:
                return

            logger.info(f"📥 Appending {len(self.transaction_batch)} transactions to dataset")

            # Load current dataset
            df_existing = pd.read_csv(self.data_path) if os.path.exists(self.data_path) else pd.DataFrame()
            original_size = len(df_existing)

            # Create dataframe from batch (includes loan_status now!)
            df_new = pd.DataFrame(self.transaction_batch)

            # Verify loan_status is present
            if "loan_status" in df_new.columns:
                logger.info(f"✓ Batch contains loan_status for all {len(df_new)} transactions")
            else:
                logger.error("✗ CRITICAL: Batch missing loan_status column!")
                return

            # Combine
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)

            # Remove duplicates
            df_combined = df_combined.drop_duplicates()
            final_size = len(df_combined)

            # Save back
            df_combined.to_csv(self.data_path, index=False)

            # Update metrics
            self.metrics["dataset_size"] = final_size
            self.save_metrics()

            logger.info("✅ Dataset updated:")
            logger.info(f"   - Original size: {original_size}")
            logger.info(f"   - Added: {len(self.transaction_batch)}")
            logger.info(f"   - After deduplication: {final_size}")
            logger.info(f"   - Net increase: {final_size - original_size}")

            # Clear batch
            self.transaction_batch = []

        except Exception as e:
            logger.error(f"✗ Error appending transactions: {e}", exc_info=True)

    def consume_kafka_stream(self):
        """Consume Kafka stream, predict, and batch for retraining"""
        logger.info(f"🎧 Starting Kafka consumer for topic: {self.kafka_topic}")
        logger.info("⚡ Rate: 1 transaction per second")
        logger.info(f"📦 Batch size: {self.batch_size} transactions")

        retry_count = 0
        max_retries = 5
        retrain_interval = int(os.getenv("RETRAINING_INTERVAL", 3600))

        while retry_count < max_retries:
            try:
                consumer = KafkaConsumer(
                    self.kafka_topic,
                    bootstrap_servers=self.kafka_bootstrap,
                    value_deserializer=lambda m: json.loads(m.decode("utf-8")),
                    auto_offset_reset="latest",
                    enable_auto_commit=True,
                    group_id=f"bank{self.bank_id}_consumer",
                )

                logger.info("✓ Kafka consumer connected successfully")
                retry_count = 0

                for message in consumer:
                    application = message.value

                    # ⭐ Predict (loan_status will be extracted and removed from features)
                    pred_status, _pred_proba, _actual_status = self.predict_loan_approval(application)

                    if pred_status is not None:
                        # ⭐ Add COMPLETE application to batch (includes loan_status)
                        self.transaction_batch.append(application)

                        # Process batch when full
                        if len(self.transaction_batch) >= self.batch_size:
                            logger.info(f"🎯 Batch complete ({self.batch_size} transactions)")
                            self.append_transactions_to_dataset()

                        # Log every 20 applications
                        if self.metrics["predictions_made"] % 20 == 0:
                            logger.info(f"📊 Bank {self.bank_id} Stats:")
                            logger.info(f"   - Total predictions: {self.metrics['predictions_made']}")
                            logger.info(f"   - Loans approved: {self.metrics['loans_approved']}")
                            logger.info(f"   - Loans denied: {self.metrics['loans_denied']}")
                            approval_rate = self.metrics["loans_approved"] / self.metrics["predictions_made"] * 100
                            logger.info(f"   - Approval rate: {approval_rate:.2f}%")
                            logger.info(f"   - Batch progress: {len(self.transaction_batch)}/{self.batch_size}")

                        # Check hourly retraining
                        time_since_last_training = time.time() - self.last_training_time
                        if time_since_last_training >= retrain_interval:
                            logger.info(f"⏰ Retraining interval reached ({retrain_interval}s = 1 hour)")
                            self.trigger_retraining()

                        # Rate control: 1 transaction per second
                        time.sleep(1.0)

            except Exception as e:
                retry_count += 1
                logger.error(f"✗ Kafka consumer error (attempt {retry_count}/{max_retries}): {e}")
                if retry_count < max_retries:
                    logger.info("⏳ Retrying in 5 seconds...")
                    time.sleep(5)
                else:
                    logger.error("✗ Max retries reached, giving up")
                    raise

    def trigger_retraining(self):
        """Trigger hourly retraining"""
        try:
            logger.info(f"🔄 Bank {self.bank_id} triggering automatic retraining round {self.training_round + 1}")

            # Train local model
            self.train_local_model()

            # Submit to server
            self.submit_model_to_server()

            logger.info("⏳ Models submitted - server will aggregate when all banks submit")

            # Try to fetch updated global model
            time.sleep(5)
            success = self.fetch_global_model()

            if success:
                logger.info(f"✅ Retraining round {self.training_round} completed successfully")
            else:
                logger.info("ℹ️ Global model not yet available - will use current model")

        except Exception as e:
            logger.error(f"✗ Retraining error: {e}", exc_info=True)

    def run(self):
        """Main run loop"""
        logger.info("🚀 Starting Bank Client with Smart Startup")

        # Register with server
        self.register_with_server()

        # Try to fetch global model
        logger.info("🔍 Attempting to fetch global model from server...")
        success = self.fetch_global_model()

        if success:
            logger.info("✅ Global model loaded from server")
            logger.info(f"   - Ready to process loan applications with round {self.training_round} model")
        else:
            # No global model available - need initial training
            logger.info("🆕 FRESH START: No global model available")
            logger.info("   - Performing initial training...")

            # Initial training
            self.train_local_model()
            self.submit_model_to_server()

            # Wait for aggregation
            logger.info("⏳ Waiting for initial aggregation (15s)")
            time.sleep(15)

            # Try to fetch global model
            success = self.fetch_global_model()

            if not success:
                logger.warning("⚠ Will use local model until global is available")

        # Start consuming loan applications
        logger.info("=" * 60)
        logger.info("🎧 Starting loan application stream consumption")
        logger.info("=" * 60)

        # Verify we have at least one model
        if self.global_model is None and self.local_model is None:
            logger.error("✗ CRITICAL: No model available for predictions!")
            logger.error("   - Cannot start processing loan applications")
            return

        model_type = "global" if self.global_model else "local"
        logger.info(f"✓ Ready with {model_type} model (round {self.training_round})")
        logger.info(f"✓ Automatic retraining every {os.getenv('RETRAINING_INTERVAL', 3600)}s (1 hour)")

        # Start Kafka consumer (blocking)
        self.consume_kafka_stream()


if __name__ == "__main__":
    bank_id = int(os.getenv("BANK_ID", "1"))

    logger.info("=" * 60)
    logger.info("🏦 BANK CLIENT STARTING - LOAN APPROVAL SYSTEM")
    logger.info("=" * 60)

    client = BankClient(bank_id)
    client.run()
