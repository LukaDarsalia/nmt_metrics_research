"""Tests for the EvalDataFetcher class."""

import unittest
from unittest.mock import patch, MagicMock

import pandas as pd

from src.data.data_fetcher import EvalDataFetcher
from src.data.db_connector import DatabaseConnector


class TestEvalDataFetcher(unittest.TestCase):
    """Test cases for EvalDataFetcher class."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock the database connector
        self.mock_db_connector = MagicMock(spec=DatabaseConnector)
        self.mock_conn = MagicMock()
        self.mock_cursor = MagicMock()

        # Configure the mock to return connection and cursor
        self.mock_db_connector.__enter__.return_value = (self.mock_conn, self.mock_cursor)

        # Create fetcher with mock connector
        self.fetcher = EvalDataFetcher(self.mock_db_connector)

    def test_fetch_evaluation_data(self):
        """Test fetching evaluation data from database."""
        # Mock data to be returned by the cursor
        mock_rows = [
            (4.5, 1, 101, "Source text 1", "Target text 1", "target1"),
            (3.2, 2, 102, "Source text 2", "Target text 2", "target2"),
        ]
        self.mock_cursor.fetchall.return_value = mock_rows

        # Call the method
        result = self.fetcher.fetch_evaluation_data()

        # Verify the result
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)
        self.assertEqual(list(result.columns),
                         ['score', 'item_id', 'createdBy_id', 'sourceText', 'targetText', 'targetID'])

        # Verify the query was executed
        self.mock_cursor.execute.assert_called_once()

    @patch('pandas.read_csv')
    def test_add_reference_text(self, mock_read_csv):
        """Test adding reference text to DataFrame."""
        # Mock the CSV reading
        mock_ref_df = pd.DataFrame({
            'en': ["Source text 1", "Source text 2"],
            'ka': ["Reference 1", "Reference 2"]
        })
        mock_read_csv.return_value = mock_ref_df

        # Mock os.path.exists
        with patch('os.path.exists', return_value=True):
            # Create test DataFrame
            test_df = pd.DataFrame({
                'sourceText': ["Source text 1", "Source text 2"],
                'targetText': ["Target text 1", "Target text 2"]
            })

            # Call the method
            result = self.fetcher.add_reference_text(test_df, "dummy_path.csv")

            # Verify the result
            self.assertIn('referenceText', result.columns)
            self.assertEqual(result['referenceText'].tolist(), ["Reference 1", "Reference 2"])

    def test_get_complete_data(self):
        """Test fetching complete data with reference text in one step."""
        # Set up mocks for the individual methods
        mock_eval_data = pd.DataFrame({
            'sourceText': ["Source text 1", "Source text 2"],
            'targetText': ["Target text 1", "Target text 2"]
        })

        mock_complete_data = mock_eval_data.copy()
        mock_complete_data['referenceText'] = ["Reference 1", "Reference 2"]

        # Create patches for the methods
        with patch.object(self.fetcher, 'fetch_evaluation_data',
                          return_value=mock_eval_data) as mock_fetch:
            with patch.object(self.fetcher, 'add_reference_text',
                              return_value=mock_complete_data) as mock_add:
                # Call the method
                result = self.fetcher.get_complete_data("dummy_path.csv")

                # Verify method calls
                mock_fetch.assert_called_once()
                mock_add.assert_called_once_with(mock_eval_data, "dummy_path.csv")

                # Verify result
                self.assertIs(result, mock_complete_data)


if __name__ == '__main__':
    unittest.main()