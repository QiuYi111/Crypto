"""Factory for creating LLM clients based on configuration."""

from typing import Union
from loguru import logger

from ..config.settings import Settings
from .llm_client import LLMClient
from .deepseek_client import DeepSeekClient


class LLMFactory:
    """Factory class for creating appropriate LLM clients."""
    
    @staticmethod
    def create_client(settings: Settings) -> Union[LLMClient, DeepSeekClient]:
        """Create the appropriate LLM client based on configuration."""
        
        provider = settings.llm_provider.lower()
        
        if provider == "deepseek":
            if not settings.deepseek_api_key:
                logger.warning("DeepSeek API key not provided, falling back to local LLM")
                return LLMClient(settings)
            
            logger.info("Creating DeepSeek API client")
            return DeepSeekClient(settings)
            
        elif provider == "local":
            logger.info("Creating local LLM client")
            return LLMClient(settings)
            
        else:
            logger.warning(f"Unknown LLM provider: {provider}, defaulting to local")
            return LLMClient(settings)
    
    @staticmethod
    def get_available_providers() -> list[str]:
        """Get list of available LLM providers."""
        return ["local", "deepseek"]