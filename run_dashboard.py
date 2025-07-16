#!/usr/bin/env python3
"""Fixed dashboard runner with proper error handling."""

import sys
import os
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    import streamlit as st
    logger.info("✅ Streamlit imported successfully")
    
    from cryptorl.config.settings import Settings
    from cryptorl.monitoring.dashboard import MonitoringDashboard
    
    def launch_fixed_dashboard():
        """Launch the dashboard with proper error handling."""
        try:
            settings = Settings()
            logger.info(f"✅ Settings loaded: {settings.project_name}")
            
            dashboard = MonitoringDashboard(settings)
            dashboard.run_dashboard()
            
        except Exception as e:
            logger.error(f"❌ Dashboard error: {e}")
            # Fallback to simple dashboard
            logger.info("🔄 Falling back to simple dashboard...")
            os.system("uv run streamlit run simple_dashboard.py --server.port=8501")
    
    if __name__ == "__main__":
        launch_fixed_dashboard()

except ImportError as e:
    logger.error(f"❌ Import error: {e}")
    logger.info("🔄 Using simple dashboard as fallback...")
    os.system("uv run streamlit run simple_dashboard.py --server.port=8501")