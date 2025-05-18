"""Module for fetching and processing data from database."""

import os
from typing import Optional

import pandas as pd

from src.data.db_connector import DatabaseConnector


class EvalDataFetcher:
    """
    Fetches evaluation data from PostgreSQL database.

    This class is responsible for retrieving evaluation text pairs and assessment
    results from the database, along with adding reference texts from external sources.
    """

    def __init__(self, db_connector: Optional[DatabaseConnector] = None):
        """
        Initialize the data fetcher.

        Args:
            db_connector (DatabaseConnector, optional): Database connector object.
                If None, creates a new connector.
        """
        self.db_connector = db_connector or DatabaseConnector()

    def fetch_evaluation_data(self) -> pd.DataFrame:
        """
        Fetch evaluation data by joining required tables.

        Returns:
            pd.DataFrame: DataFrame containing evaluation data with columns:
                          'score', 'item_id', 'createdBy_id', 'sourceText',
                          'targetText', 'targetID'

        Raises:
            ConnectionError: If database connection fails
            RuntimeError: If query execution fails
        """
        try:
            with self.db_connector as (conn, cur):
                # Query to join the tables and fetch required columns
                query = """
                        SELECT dr.score, \
                               dr.item_id, \
                               dr."createdBy_id", \
                               tp."sourceText", \
                               tp."targetText", \
                               tp."targetID"
                        FROM "EvalData_directassessmentresult" dr
                                 JOIN "EvalData_textpair" tp
                                      ON dr.item_id = tp.id \
                        """

                # Execute query
                cur.execute(query)
                rows = cur.fetchall()

                # Create DataFrame
                columns = ['score', 'item_id', 'createdBy_id', 'sourceText', 'targetText', 'targetID']
                df = pd.DataFrame(rows, columns=columns)

                return df

        except Exception as e:
            raise RuntimeError(f"Failed to fetch evaluation data: {e}")

    def add_reference_text(self, df: pd.DataFrame, reference_file: str) -> pd.DataFrame:
        """
        Add reference text to the DataFrame from an external source.

        Args:
            df (pd.DataFrame): DataFrame containing evaluation data
            reference_file (str): Path to the reference file (CSV)

        Returns:
            pd.DataFrame: DataFrame with added reference text column

        Raises:
            FileNotFoundError: If reference file doesn't exist
            KeyError: If source text doesn't have a matching reference
        """
        if not os.path.exists(reference_file):
            raise FileNotFoundError(f"Reference file not found: {reference_file}")

        # Load reference data
        reference_df = pd.read_csv(reference_file)

        # Create source-to-reference mapping
        source_ref_map = dict(zip(reference_df['en'], reference_df['ka']))

        # Add reference text column
        try:
            df_copy = df.copy()
            df_copy['referenceText'] = df_copy['sourceText'].apply(lambda x: source_ref_map[x])
            return df_copy
        except KeyError as e:
            raise KeyError(f"Source text not found in reference data: {e}")

    def get_complete_data(self, reference_file: str) -> pd.DataFrame:
        """
        Fetch evaluation data and add reference text in one operation.

        Args:
            reference_file (str): Path to the reference file (CSV)

        Returns:
            pd.DataFrame: Complete DataFrame with all required columns
        """
        # Fetch evaluation data
        eval_data = self.fetch_evaluation_data()

        # Add reference text
        complete_data = self.add_reference_text(eval_data, reference_file)

        return complete_data