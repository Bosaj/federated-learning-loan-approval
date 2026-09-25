"""
Kafka Producer - Streams loan applications from real testing dataset
"""

import json
import logging
import os
import time

import pandas as pd
import requests
from kafka.errors import NoBrokersAvailable

from kafka import KafkaProducer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class LoanApplicationProducer:
    def __init__(self, bank_id, kafka_bootstrap="kafka:9092"):
        self.bank_id = bank_id
        self.kafka_bootstrap = kafka_bootstrap
        self.topic = f"bank{bank_id}_txn"
        self.producer = None
        self.server_url = f"http://{os.getenv('SERVER_HOST', 'federated-server')}:{os.getenv('SERVER_PORT', '5000')}"

        # ⭐ NEW: Path to real-time testing dataset
        self.dataset_path = "/app/data/kafka/real_time_testing_dataset.csv"

        self.connect()

    def connect(self):
        """Connect to Kafka broker with retry logic"""
        max_retries = 30
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                self.producer = KafkaProducer(
                    bootstrap_servers=self.kafka_bootstrap,
                    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                    acks="all",
                    retries=3,
                )
                logger.info(f"✓ Producer for Bank {self.bank_id} connected to Kafka")
                return
            except NoBrokersAvailable:
                logger.warning(f"⏳ Kafka not ready, retrying... ({attempt + 1}/{max_retries})")
                time.sleep(retry_delay)

        raise Exception("Failed to connect to Kafka after multiple retries")

    def wait_for_initial_training(self):
        """
        Wait for federated server to complete initial training
        Checks if global model is available before starting application stream
        """
        logger.info("⏰ Waiting for initial federated training to complete...")
        logger.info("   (Banks need to train → submit models → server aggregates → banks fetch global model)")

        max_wait_time = 180  # 3 minutes max
        check_interval = 5  # Check every 5 seconds
        elapsed = 0

        while elapsed < max_wait_time:
            try:
                response = requests.get(f"{self.server_url}/status", timeout=5)

                if response.status_code == 200:
                    data = response.json()
                    training_round = data.get("training_round", 0)
                    global_model_available = data.get("global_model_available", False)

                    logger.info(f"   📊 Server status: Round {training_round}, Global model: {global_model_available}")

                    # Check if initial training is complete
                    if training_round >= 1 and global_model_available:
                        logger.info("✅ Initial training complete! Starting loan application stream...")
                        return True

                    logger.info(f"   ⏳ Not ready yet, checking again in {check_interval}s...")

            except Exception as e:
                logger.debug(f"   ⚠ Could not reach server: {e}")

            time.sleep(check_interval)
            elapsed += check_interval

        logger.warning("⚠ Max wait time reached, starting anyway...")
        return False

    def load_testing_dataset(self):
        """⭐ NEW: Load real-time testing dataset from CSV"""
        try:
            logger.info(f"📂 Loading testing dataset from {self.dataset_path}")

            if not os.path.exists(self.dataset_path):
                logger.error(f"✗ Dataset not found: {self.dataset_path}")
                return None

            df = pd.read_csv(self.dataset_path)
            logger.info(f"✓ Loaded {len(df)} loan applications from dataset")

            # Log dataset info
            if "loan_status" in df.columns:
                approved = df["loan_status"].sum()
                logger.info(f"   - Approved: {approved} ({(approved / len(df) * 100):.1f}%)")
                logger.info(f"   - Denied: {len(df) - approved} ({((len(df) - approved) / len(df) * 100):.1f}%)")

            logger.info(f"   - Features: {', '.join(df.columns.tolist())}")

            return df

        except Exception as e:
            logger.error(f"✗ Error loading dataset: {e}", exc_info=True)
            return None

    def produce_stream(self, max_applications=100, rate=1.0):
        """
        ⭐ UPDATED: Produce stream of loan applications from real dataset
        NOW KEEPS loan_status in the message (banks will handle it properly)

        Args:
            max_applications: Number of applications to send (default: 100)
            rate: Applications per second (default: 1.0)
        """
        logger.info(f"🚀 Starting loan application stream for Bank {self.bank_id}")
        logger.info(f"   - Max applications: {max_applications}")
        logger.info(f"   - Rate: {rate} applications/s")
        logger.info(f"   - Topic: {self.topic}")
        logger.info("   - ⭐ Sending WITH loan_status (for validation & retraining)")

        # Load dataset
        df = self.load_testing_dataset()
        if df is None:
            logger.error("✗ Cannot start producer without dataset")
            return

        # Shuffle to randomize which applications each bank gets
        df = df.sample(frac=1, random_state=self.bank_id).reset_index(drop=True)

        # Limit to max_applications
        df = df.head(max_applications)

        logger.info(f"📤 Will send {len(df)} loan applications")

        count = 0
        start_time = time.time()

        try:
            for _idx, row in df.iterrows():
                # Convert row to dictionary
                application = row.to_dict()

                # ⭐ CHANGED: Keep loan_status in the message
                # Banks will remove it before prediction but keep it for validation/retraining

                # Send to Kafka (with loan_status included)
                self.producer.send(self.topic, value=application)

                count += 1

                # Log every 20 applications
                if count % 20 == 0:
                    elapsed = time.time() - start_time
                    actual_rate = count / elapsed if elapsed > 0 else 0

                    logger.info(f"📊 Bank {self.bank_id} - Sent {count}/{len(df)} applications")
                    logger.info(f"   - Actual rate: {actual_rate:.2f} apps/s")

                # Control rate
                time.sleep(1.0 / rate)

            # Final stats
            elapsed = time.time() - start_time
            logger.info(f"✅ Completed sending {count} loan applications in {elapsed:.1f}s")
            logger.info(f"   - Average rate: {(count / elapsed):.2f} apps/s")

        except KeyboardInterrupt:
            logger.info("⏸ Producer stopped by user")
        except Exception as e:
            logger.error(f"✗ Producer error: {e}", exc_info=True)
        finally:
            if self.producer:
                self.producer.flush()
                self.producer.close()
                logger.info(f"✓ Producer for Bank {self.bank_id} closed")


def create_topics():
    """Create Kafka topics for all banks"""
    from kafka.admin import KafkaAdminClient, NewTopic

    logger.info("📝 Creating Kafka topics...")

    try:
        admin_client = KafkaAdminClient(bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092"))

        topics = []
        for bank_id in range(1, 4):
            topics.append(NewTopic(name=f"bank{bank_id}_txn", num_partitions=1, replication_factor=1))

        admin_client.create_topics(new_topics=topics, validate_only=False)
        logger.info("✓ Kafka topics created successfully")
        admin_client.close()
    except Exception as e:
        logger.info(f"ℹ Topics may already exist: {e}")


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("🏦 KAFKA LOAN APPLICATION PRODUCER")
    logger.info("=" * 60)

    # Get configuration from environment
    bank_id = int(os.getenv("BANK_ID", "1"))
    kafka_bootstrap = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
    rate = float(os.getenv("TXN_RATE", "1.0"))
    max_apps = int(os.getenv("MAX_APPLICATIONS", "100"))  # ⭐ NEW: configurable

    logger.info("📋 Configuration:")
    logger.info(f"   - Bank ID: {bank_id}")
    logger.info(f"   - Kafka: {kafka_bootstrap}")
    logger.info(f"   - Application rate: {rate} apps/s")
    logger.info(f"   - Max applications: {max_apps}")
    logger.info("=" * 60)

    # Wait for Kafka to be ready FIRST
    logger.info("⏳ Waiting for Kafka to be ready (10s)...")
    time.sleep(10)

    # Create topics (idempotent)
    try:
        create_topics()
    except Exception as e:
        logger.warning(f"⚠ Could not create topics: {e}")

    # Initialize producer
    producer = LoanApplicationProducer(bank_id, kafka_bootstrap)

    # Wait for initial training to complete
    producer.wait_for_initial_training()

    # Start producing loan applications (sends max_apps then stops)
    producer.produce_stream(max_applications=max_apps, rate=rate)
