"""Database connection management module."""

import os
import psycopg2
from dotenv import load_dotenv


class DatabaseConnector:
    """
    Manages connections to PostgreSQL database.

    This class handles creating and closing database connections using
    environment variables for configuration.
    """

    def __init__(self, env_path=None):
        """
        Initialize the database connector.

        Args:
            env_path (str, optional): Path to .env file. If None, looks in current directory.
        """
        # Load environment variables
        load_dotenv(env_path)

        # Get database parameters from environment variables
        self.db_params = {
            'host': os.getenv('DB_HOST'),
            'port': os.getenv('DB_PORT'),
            'user': os.getenv('DB_USER'),
            'password': os.getenv('DB_PASSWORD'),
            'database': os.getenv('DB_NAME')
        }

        self.conn = None
        self.cur = None

    def connect(self):
        """
        Establish connection to the database.

        Returns:
            tuple: (connection, cursor) for database operations

        Raises:
            psycopg2.Error: If connection fails
        """
        try:
            self.conn = psycopg2.connect(**self.db_params)
            self.cur = self.conn.cursor()
            return self.conn, self.cur
        except psycopg2.Error as e:
            raise ConnectionError(f"Failed to connect to database: {e}")

    def close(self):
        """Close database connection and cursor safely."""
        if self.cur:
            self.cur.close()
        if self.conn:
            self.conn.close()

    def __enter__(self):
        """Context manager entry point."""
        return self.connect()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit point."""
        self.close()