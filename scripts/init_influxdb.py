#!/usr/bin/env python3
"""
InfluxDB initialization script for CryptoRL.

This script sets up the InfluxDB organization, bucket, and token needed for the CryptoRL system.
"""

import requests
import time
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loguru import logger

# InfluxDB configuration
INFLUXDB_URL = "http://localhost:8086"
INFLUXDB_USERNAME = "admin"
INFLUXDB_PASSWORD = "cryptorl_admin_2024"
INFLUXDB_ORG = "cryptorl"
INFLUXDB_BUCKET = "market_data"
INFLUXDB_TOKEN = "cryptorl_token_2024"

def wait_for_influxdb():
    """Wait for InfluxDB to be ready."""
    logger.info("Waiting for InfluxDB to be ready...")
    max_retries = 30
    
    for i in range(max_retries):
        try:
            response = requests.get(f"{INFLUXDB_URL}/health", timeout=10)
            if response.status_code == 200:
                logger.info("InfluxDB is ready!")
                return True
        except requests.exceptions.RequestException:
            pass
        
        if i < max_retries - 1:
            logger.info(f"InfluxDB not ready, retrying in 2 seconds... ({i+1}/{max_retries})")
            time.sleep(2)
    
    logger.error("InfluxDB failed to start")
    return False

def setup_influxdb():
    """Set up InfluxDB with organization, bucket, and token."""
    
    if not wait_for_influxdb():
        return False
    
    # Setup request payload
    setup_data = {
        "username": INFLUXDB_USERNAME,
        "password": INFLUXDB_PASSWORD,
        "org": INFLUXDB_ORG,
        "bucket": INFLUXDB_BUCKET,
        "token": INFLUXDB_TOKEN
    }
    
    try:
        logger.info("Setting up InfluxDB organization and bucket...")
        response = requests.post(f"{INFLUXDB_URL}/api/v2/setup", json=setup_data, timeout=30)
        
        if response.status_code == 201:
            logger.success("InfluxDB setup completed successfully!")
            
            # Extract and display the token
            data = response.json()
            token = data.get("auth", {}).get("token")
            if token:
                logger.info(f"Admin token: {token}")
                logger.info("Please update your .env file with this token")
            
            return True
        elif response.status_code == 422:
            logger.warning("InfluxDB appears to already be set up")
            
            # Try to get existing token
            try:
                auth_response = requests.post(f"{INFLUXDB_URL}/api/v2/signin", 
                                            auth=(INFLUXDB_USERNAME, INFLUXDB_PASSWORD),
                                            timeout=10)
                if auth_response.status_code == 204:
                    logger.info("Authentication successful with existing credentials")
                    return True
            except Exception as e:
                logger.error(f"Authentication failed: {e}")
                
        else:
            logger.error(f"InfluxDB setup failed: {response.status_code} - {response.text}")
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to connect to InfluxDB: {e}")
    
    return False

def main():
    """Main initialization function."""
    logger.info("Starting InfluxDB initialization...")
    
    if setup_influxdb():
        logger.success("InfluxDB initialization completed!")
        
        # Update .env file with the correct token
        env_path = Path(__file__).parent.parent / ".env"
        if env_path.exists():
            logger.info("Please ensure your .env file contains:")
            logger.info(f"INFLUXDB_URL={INFLUXDB_URL}")
            logger.info(f"INFLUXDB_TOKEN={INFLUXDB_TOKEN}")
            logger.info(f"INFLUXDB_ORG={INFLUXDB_ORG}")
            logger.info(f"INFLUXDB_BUCKET={INFLUXDB_BUCKET}")
    else:
        logger.error("InfluxDB initialization failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()