"""Configuration loading and management."""

import os
from dotenv import load_dotenv
from typing import Dict, Optional


def load_config(env_path: Optional[str] = None) -> Dict[str, str]:
    """
    Load configuration from environment variables.

    Args:
        env_path (str, optional): Path to .env file. If None, looks in current directory.

    Returns:
        Dict[str, str]: Dictionary containing configuration values
    """
    # Load environment variables
    load_dotenv(env_path)

    # Return configuration dictionary
    return {
        'db_host': os.getenv('DB_HOST'),
        'db_port': os.getenv('DB_PORT'),
        'db_user': os.getenv('DB_USER'),
        'db_password': os.getenv('DB_PASSWORD'),
        'db_name': os.getenv('DB_NAME'),
    }