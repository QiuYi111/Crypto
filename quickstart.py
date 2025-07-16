#!/usr/bin/env python3
"""
CryptoRL Agent Quick Start Script

This script provides a streamlined setup process for the CryptoRL Agent.
It handles environment setup, configuration validation, and initial deployment.
"""

import os
import sys
import json
import subprocess
import platform
from pathlib import Path
from typing import Dict, Any, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CryptoRLQuickStart:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.env_file = self.project_root / ".env"
        self.config_file = self.project_root / "config" / "settings.json"
        
    def check_python_version(self) -> bool:
        """Check if Python version is compatible."""
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            logger.error("Python 3.8+ is required")
            return False
        logger.info(f"Python {version.major}.{version.minor}.{version.micro} detected")
        return True
    
    def check_uv_installation(self) -> bool:
        """Check if uv is installed."""
        try:
            subprocess.run(["uv", "--version"], capture_output=True, check=True)
            logger.info("uv is installed")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("uv not found. Installing uv...")
            return self.install_uv()
    
    def install_uv(self) -> bool:
        """Install uv package manager."""
        try:
            if platform.system() == "Windows":
                subprocess.run([
                    "powershell", "-ExecutionPolicy", "ByPass", "-c",
                    "irm https://astral.sh/uv/install.ps1 | iex"
                ], check=True)
            else:
                subprocess.run([
                    "curl", "-LsSf", "https://astral.sh/uv/install.sh", "|", "sh"
                ], shell=True, check=True)
            logger.info("uv installed successfully")
            return True
        except subprocess.CalledProcessError:
            logger.error("Failed to install uv")
            return False
    
    def setup_environment(self) -> bool:
        """Set up Python environment with dependencies."""
        try:
            logger.info("Setting up Python environment...")
            
            # Create virtual environment
            subprocess.run(["uv", "venv"], cwd=self.project_root, check=True)
            
            # Install dependencies
            subprocess.run(["uv", "pip", "install", "-e", "."], 
                         cwd=self.project_root, check=True)
            
            # Install development dependencies
            subprocess.run(["uv", "pip", "install", "-e", ".[dev]"], 
                         cwd=self.project_root, check=True)
            
            logger.info("Environment setup complete")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Environment setup failed: {e}")
            return False
    
    def create_env_template(self) -> Dict[str, str]:
        """Create .env template with required variables."""
        return {
            "BINANCE_API_KEY": "your_binance_api_key_here",
            "BINANCE_SECRET_KEY": "your_binance_secret_key_here",
            "BINANCE_TESTNET": "true",
            "DATABASE_URL": "postgresql://cryptorl:cryptorl@postgresql:5432/cryptorl",
            "INFLUXDB_URL": "http://influxdb:8086",
            "INFLUXDB_TOKEN": "cryptorl_token_2024",
            "INFLUXDB_BUCKET": "market_data",
            "INFLUXDB_ORG": "cryptorl",
            "LLM_PROVIDER": "deepseek",
            "DEEPSEEK_API_KEY": "your_deepseek_api_key_here",
            "LOG_LEVEL": "INFO",
            "ENVIRONMENT": "development",
            "REDIS_URL": "redis://redis:6379/0"
        }
    
    def setup_configuration(self) -> bool:
        """Set up configuration files."""
        try:
            # Create .env file if it doesn't exist
            if not self.env_file.exists():
                logger.info("Creating .env file...")
                template = self.create_env_template()
                with open(self.env_file, 'w') as f:
                    for key, value in template.items():
                        f.write(f"{key}={value}\n")
                logger.info("✅ .env file created. Please update with your actual credentials.")
            
            # Ensure config directory exists
            config_dir = self.project_root / "config"
            config_dir.mkdir(exist_ok=True)
            
            return True
            
        except Exception as e:
            logger.error(f"Configuration setup failed: {e}")
            return False
    
    def validate_configuration(self) -> bool:
        """Validate critical configuration."""
        required_vars = [
            "BINANCE_API_KEY",
            "BINANCE_SECRET_KEY",
            "DATABASE_URL",
            "INFLUXDB_URL",
            "LLM_PROVIDER"
        ]
        
        if not self.env_file.exists():
            logger.error("❌ .env file not found")
            return False
        
        # Load environment variables
        with open(self.env_file) as f:
            env_content = f.read()
        
        missing_vars = []
        for var in required_vars:
            if var not in env_content or f"{var}=your_" in env_content:
                missing_vars.append(var)
        
        if missing_vars:
            logger.error(f"❌ Missing required environment variables: {missing_vars}")
            return False
        
        logger.info("✅ Configuration validated")
        return True
    
    def run_tests(self) -> bool:
        """Run basic tests to verify installation."""
        try:
            logger.info("Running basic tests...")
            
            # Clean environment to avoid mamba conflicts
            clean_env = dict(os.environ)
            clean_env.pop('PYTHONHOME', None)
            clean_env.pop('PYTHONPATH', None)
            clean_env['PATH'] = f"{self.project_root}/.venv/bin:{clean_env.get('PATH', '')}"
            
            # Test 1: Basic import
            result = subprocess.run([
                str(self.project_root / ".venv" / "bin" / "python"), "-c", 
                "import sys; sys.path.insert(0, 'src'); import cryptorl; print('✅ Package imported successfully')"
            ], cwd=self.project_root, capture_output=True, text=True, env=clean_env)
            
            if result.returncode != 0:
                logger.error(f"❌ Import test failed: {result.stderr}")
                return False
            
            logger.info("✅ Basic import test passed")
            
            # Test 2: Settings validation
            result = subprocess.run([
                str(self.project_root / ".venv" / "bin" / "python"), "-c", 
                "import sys; sys.path.insert(0, 'src'); from cryptorl.config import settings; print('✅ Settings loaded successfully')"
            ], cwd=self.project_root, capture_output=True, text=True, env=clean_env)
            
            if result.returncode != 0:
                logger.error(f"❌ Settings test failed: {result.stderr}")
                return False
            
            logger.info("✅ Settings validation passed")
            
            # Test 3: Core dependencies
            result = subprocess.run([
                str(self.project_root / ".venv" / "bin" / "python"), "-c", 
                "import sys; sys.path.insert(0, 'src'); import torch; import numpy as np; import pandas as pd; print('✅ Core dependencies working')"
            ], cwd=self.project_root, capture_output=True, text=True, env=clean_env)
            
            if result.returncode != 0:
                logger.error(f"❌ Dependencies test failed: {result.stderr}")
                return False
            
            logger.info("✅ Core dependencies test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Test failed: {e}")
            return False
    
    def start_dashboard(self) -> bool:
        """Start the monitoring dashboard."""
        try:
            logger.info("Starting monitoring dashboard...")
            clean_env = dict(os.environ)
            clean_env.pop('PYTHONHOME', None)
            clean_env.pop('PYTHONPATH', None)
            clean_env['PATH'] = f"{self.project_root}/.venv/bin:{clean_env.get('PATH', '')}"
            subprocess.run([
                str(self.project_root / ".venv" / "bin" / "python"), "-m", "streamlit", "run", 
                "src/cryptorl/monitoring/dashboard.py", "--server.port=8501"
            ], cwd=self.project_root, env=clean_env)
            return True
        except KeyboardInterrupt:
            logger.info("Dashboard stopped by user")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to start dashboard: {e}")
            return False

    def start_docker_services(self) -> bool:
        """Start Docker services (databases)."""
        try:
            logger.info("Starting Docker services...")
            # Try docker-compose first, then fallback to docker compose
            try:
                subprocess.run([
                    "docker-compose", "up", "-d", "influxdb", "postgresql", "redis"
                ], cwd=self.project_root, check=True)
            except FileNotFoundError:
                subprocess.run([
                    "docker", "compose", "up", "-d", "influxdb", "postgresql", "redis"
                ], cwd=self.project_root, check=True)
            logger.info("✅ Docker services started")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to start Docker services: {e}")
            return False
        except FileNotFoundError:
            logger.error("❌ Docker not found. Please install Docker and Docker Compose first")
            return False

    def stop_docker_services(self) -> bool:
        """Stop Docker services."""
        try:
            logger.info("Stopping Docker services...")
            try:
                subprocess.run([
                    "docker-compose", "down"
                ], cwd=self.project_root, check=True)
            except FileNotFoundError:
                subprocess.run([
                    "docker", "compose", "down"
                ], cwd=self.project_root, check=True)
            logger.info("✅ Docker services stopped")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to stop Docker services: {e}")
            return False
        except FileNotFoundError:
            logger.error("❌ Docker not found. Please install Docker and Docker Compose first")
            return False
    
    def display_help(self) -> None:
        """Display help information."""
        help_text = """
🚀 CryptoRL Agent Quick Start Guide

📋 Setup Steps:
1. Run this script: python quickstart.py --setup
2. Update .env file with your API credentials
3. Run tests: python quickstart.py --test
4. Start dashboard: python quickstart.py --dashboard

🔧 Available Commands:
  --setup       Complete environment setup + start Docker services
  --test        Run validation tests
  --dashboard   Start monitoring dashboard
  --validate    Check configuration
  --docker-up   Start Docker services (databases)
  --docker-down Stop Docker services
  --help        Show this help message

📊 After Setup:
- Dashboard: http://localhost:8501
- InfluxDB: http://localhost:8086 (admin/cryptorl_admin_2024)
- PostgreSQL: localhost:5432 (cryptorl/cryptorl)
- Redis: localhost:6379
- Logs: logs/cryptorl.log

🐳 Docker Services:
- influxdb: Time-series database for market data
- postgresql: Relational database for account/orders
- redis: Caching and session management

🆘 Support:
- Check logs/cryptorl.log for detailed errors
- Verify .env file contains valid credentials
- Ensure Docker is running: docker-compose ps
"""
        print(help_text)
    
    def check_docker_installation(self) -> bool:
        """Check if Docker is installed."""
        try:
            subprocess.run(["docker", "--version"], capture_output=True, check=True)
            logger.info("✅ Docker is installed")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("❌ Docker not found. Please install Docker first")
            logger.info("Installation: https://docs.docker.com/get-docker/")
            return False

    def interactive_setup(self) -> bool:
        """Interactive setup wizard."""
        print("🎯 CryptoRL Agent Interactive Setup")
        print("=" * 40)
        
        # Python version check
        if not self.check_python_version():
            return False
        
        # Docker check
        if not self.check_docker_installation():
            return False
        
        # UV installation
        if not self.check_uv_installation():
            return False
        
        # Environment setup
        if not self.setup_environment():
            return False
        
        # Configuration setup
        if not self.setup_configuration():
            return False
        
        print("\n✅ Setup complete!")
        print("\nNext steps:")
        print("1. Edit .env file with your actual API credentials")
        print("2. Run: python quickstart.py --validate")
        print("3. Run: python quickstart.py --test")
        print("4. Run: python quickstart.py --dashboard")
        print("5. To stop services: python quickstart.py --docker-down")
        
        return True
    
    def run(self) -> None:
        """Main execution method."""
        if len(sys.argv) == 1 or "--help" in sys.argv:
            self.display_help()
            return
        
        if "--setup" in sys.argv:
            success = self.interactive_setup()
            if success:
                success = self.start_docker_services()
            sys.exit(0 if success else 1)
        
        if "--validate" in sys.argv:
            success = self.validate_configuration()
            sys.exit(0 if success else 1)
        
        if "--test" in sys.argv:
            success = self.run_tests()
            sys.exit(0 if success else 1)
        
        if "--dashboard" in sys.argv:
            success = self.start_dashboard()
            sys.exit(0 if success else 1)
        
        if "--docker-up" in sys.argv:
            success = self.start_docker_services()
            sys.exit(0 if success else 1)
        
        if "--docker-down" in sys.argv:
            success = self.stop_docker_services()
            sys.exit(0 if success else 1)

if __name__ == "__main__":
    quickstart = CryptoRLQuickStart()
    quickstart.run()