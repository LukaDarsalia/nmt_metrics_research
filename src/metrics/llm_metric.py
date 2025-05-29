"""
LLM-based metric implementation for NMT evaluation using Anthropic Batch API.

This module provides LLM-based evaluation using the Anthropic Batch API with support for
both reference-based and reference-free evaluation modes.
Can be run standalone from command line.
"""

import argparse
import json
import os
import re
import sys
import time
from typing import List, Optional, Literal, Dict, Any
from datetime import datetime

sys.stdout.reconfigure(encoding='utf-8')
sys.stdin.reconfigure(encoding='utf-8')

# Add parent directory to path to import base_metric
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.metrics.base_metric import BaseMetric
from src.utils.llm_prompts import LLMPrompts

from dotenv import load_dotenv
load_dotenv()  # Load .env file from current directory or parent directories


class LLMMetric(BaseMetric):
    """LLM-based metric implementation using Anthropic Batch API."""

    def __init__(
        self,
        evaluation_mode: Literal["reference_based", "reference_free"] = "reference_based",
        model_name: str = "claude-sonnet-4-20250514",
        api_key: Optional[str] = None,
        batch_size: Optional[int] = None,  # Now defaults to None (will use DataFrame size)
        max_retries: int = 3,
        env_path: Optional[str] = None,
        progress_file: Optional[str] = None
    ):
        """
        Initialize the LLM metric.

        Args:
            evaluation_mode: Type of evaluation - "reference_based" or "reference_free"
            model_name: Anthropic model to use (e.g., "claude-sonnet-4-20250514")
            api_key: Anthropic API key (if None, reads from ANTHROPIC_API_KEY env var)
            batch_size: Number of samples to evaluate (None = all data, max 100000)
            max_retries: Maximum number of retries for failed requests
            env_path: Path to .env file (if None, searches in current/parent directories)
            progress_file: Path to save/load progress (for resuming interrupted batches)
        """
        # Load environment variables from specified .env file
        if env_path:
            try:
                from dotenv import load_dotenv
                load_dotenv(env_path)
                print(f"✅ Loaded environment variables from: {env_path}")
            except ImportError:
                print("Warning: python-dotenv not installed, cannot load custom .env file")
            except Exception as e:
                print(f"Warning: Failed to load .env file {env_path}: {e}")

        mode_name = f"LLM-{evaluation_mode.replace('_', '-').title()}"
        super().__init__(mode_name)

        self.evaluation_mode = evaluation_mode
        self.model_name = model_name or os.getenv("LLM_MODEL", "claude-sonnet-4-20250514")
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        # batch_size will be set in calculate_scores() based on DataFrame size
        self.batch_size = batch_size  # Can be None initially
        self.max_retries = max_retries or int(os.getenv("LLM_MAX_RETRIES", "3"))
        self.env_path = env_path
        self.progress_file = progress_file or f"llm_batch_progress_{evaluation_mode}.json"

        self._client = None
        self._prompts = LLMPrompts()

    def get_dependencies(self) -> List[str]:
        """Return required packages for LLM metric."""
        return ["anthropic", "python-dotenv", "requests"]

    def _initialize_client(self) -> None:
        """Initialize Anthropic client lazily."""
        if self._client is None:
            if not self.api_key:
                raise RuntimeError(
                    "Anthropic API key not found. Please:\n"
                    "1. Set ANTHROPIC_API_KEY environment variable, or\n"
                    "2. Create a .env file with ANTHROPIC_API_KEY=your_key, or\n"
                    "3. Pass --api-key parameter, or\n"
                    "4. Use --env-path to specify a custom .env file location"
                )

            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.api_key)
                print(f"✅ Initialized Anthropic client with model: {self.model_name}")
            except ImportError as e:
                raise RuntimeError(
                    f"Failed to import anthropic. Install with: pip install anthropic\n"
                    f"Error: {e}"
                )

    def _get_prompt_template(self) -> str:
        """
        Get appropriate prompt template based on evaluation mode.

        Returns:
            str: Prompt template for the current evaluation mode
        """
        if self.evaluation_mode == "reference_based":
            return self._prompts.get_reference_based_prompt()
        elif self.evaluation_mode == "reference_free":
            return self._prompts.get_reference_free_prompt()
        else:
            raise ValueError(f"Unsupported evaluation mode: {self.evaluation_mode}")

    def _format_prompt(
        self,
        source: str,
        hypothesis: str,
        reference: Optional[str] = None
    ) -> str:
        """
        Format prompt with actual text data.

        Args:
            source: Source text
            hypothesis: Machine translation to evaluate
            reference: Reference translation (required for reference_based mode)

        Returns:
            str: Formatted prompt ready for LLM

        Raises:
            ValueError: If reference is None in reference_based mode
        """
        template = self._get_prompt_template()

        if self.evaluation_mode == "reference_based":
            if reference is None:
                raise ValueError("Reference text required for reference-based evaluation")
            return template.format(source=source, hypothesis=hypothesis, reference=reference)
        else:
            return template.format(source=source, hypothesis=hypothesis)

    def _extract_score(self, response_text: str) -> float:
        """
        Extract numerical score from LLM response.

        Args:
            response_text: Raw response from LLM

        Returns:
            float: Extracted score (0-100 scale)
        """
        # Clean the response
        cleaned = response_text.strip()

        # Try to find a number between 0 and 100
        patterns = [
            r'\b(\d{1,3}(?:\.\d+)?)\b',  # Any number
            r'Score:\s*(\d{1,3}(?:\.\d+)?)',  # "Score: XX"
            r'(\d{1,3}(?:\.\d+)?)(?:\s*(?:/100|%)?)',  # Number with optional /100 or %
        ]

        for pattern in patterns:
            match = re.search(pattern, cleaned)
            if match:
                try:
                    score = float(match.group(1))
                    # Ensure score is in valid range
                    if 0 <= score <= 100:
                        return score
                    elif score > 100:
                        # If score is > 100, it might be on a different scale
                        return min(100.0, score)
                except ValueError:
                    continue

        # If no valid score found, log the response and return 0
        print(f"Warning: Could not extract valid score from response: '{cleaned}'. Using 0.0")
        return 0.0

    def _save_progress(self, progress_data: Dict[str, Any]) -> None:
        """
        Save progress to file for resuming interrupted batches.

        Args:
            progress_data: Dictionary containing progress information
        """
        try:
            with open(self.progress_file, 'w') as f:
                json.dump(progress_data, f, indent=2)
            print(f"💾 Progress saved to {self.progress_file}")
        except Exception as e:
            print(f"Warning: Failed to save progress: {e}")

    def _load_progress(self) -> Optional[Dict[str, Any]]:
        """
        Load progress from file to resume interrupted batches.

        Returns:
            Progress data if file exists, None otherwise
        """
        try:
            if os.path.exists(self.progress_file):
                with open(self.progress_file, 'r') as f:
                    progress_data = json.load(f)
                print(f"📁 Loaded progress from {self.progress_file}")
                return progress_data
        except Exception as e:
            print(f"Warning: Failed to load progress: {e}")
        return None

    def _create_batch_requests(
        self,
        hypotheses: List[str],
        sources: List[str],
        references: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Create batch requests for all evaluation pairs.

        Args:
            hypotheses: List of machine translation outputs
            sources: List of source texts
            references: List of reference translations (optional)

        Returns:
            List of batch request dictionaries
        """
        requests = []

        # Choose examples based on evaluation mode
        if self.evaluation_mode == "reference_based":
            examples_section = self._prompts.get_reference_based_examples_section()
        else:
            examples_section = self._prompts.get_examples_section()

        # Get prompt configuration
        config = self._prompts.get_prompt_config()

        for i, (hyp, src) in enumerate(zip(hypotheses, sources)):
            ref = references[i] if references else None

            try:
                # Format the evaluation prompt
                evaluation_prompt = self._format_prompt(src, hyp, ref)

                # Prepare message content with examples and evaluation prompt
                message_content = [
                    {
                        "type": "text",
                        "text": examples_section
                    },
                    {
                        "type": "text",
                        "text": evaluation_prompt
                    }
                ]

                # Create batch request
                request = {
                    "custom_id": f"eval_{self.evaluation_mode}_{i}",
                    "params": {
                        "model": config["model"],
                        "max_tokens": config["max_tokens"],
                        "temperature": config["temperature"],
                        "system": self._prompts.get_system_prompt(),
                        "messages": [
                            {
                                "role": "user",
                                "content": message_content
                            }
                        ]
                    }
                }
                requests.append(request)

            except Exception as e:
                print(f"Warning: Failed to create request for sample {i}: {e}")
                # Create a dummy request that will return 0
                dummy_request = {
                    "custom_id": f"eval_{self.evaluation_mode}_{i}",
                    "params": {
                        "model": config["model"],
                        "max_tokens": 10,
                        "temperature": 0,
                        "system": "Return 0",
                        "messages": [{"role": "user", "content": "0"}]
                    }
                }
                requests.append(dummy_request)

        return requests

    def _create_batch(self, requests: List[Dict[str, Any]]) -> str:
        """
        Create a batch with the given requests.

        Args:
            requests: List of batch requests

        Returns:
            Batch ID for tracking
        """
        try:
            # Use the correct beta API endpoint
            batch = self._client.messages.batches.create(requests=requests)
            print(f"✅ Created batch {batch.id} with {len(requests)} requests")
            return batch.id
        except Exception as e:
            raise RuntimeError(f"Failed to create batch: {e}")

    def _wait_for_batch_completion(self, batch_id: str, check_interval: int = 30) -> Dict[str, Any]:
        """
        Wait for batch to complete, with progress updates.

        Args:
            batch_id: Batch ID to monitor
            check_interval: Seconds between status checks

        Returns:
            Completed batch information
        """
        print(f"⏳ Waiting for batch {batch_id} to complete...")

        while True:
            try:
                batch = self._client.messages.batches.retrieve(batch_id)
                status = batch.processing_status  # Correct attribute name from documentation

                # Print progress update - using correct attribute names
                request_counts = batch.request_counts
                completed = getattr(request_counts, 'succeeded', 0)
                failed = getattr(request_counts, 'errored', 0)
                processing = getattr(request_counts, 'processing', 0)
                canceled = getattr(request_counts, 'canceled', 0)
                expired = getattr(request_counts, 'expired', 0)
                total = processing + completed + failed + canceled + expired

                if total > 0:
                    progress_pct = (completed / total) * 100
                    print(f"📊 Batch {batch_id} - Status: {status} | Progress: {completed}/{total} ({progress_pct:.1f}%) | Failed: {failed} | Canceled: {canceled} | Expired: {expired}")
                else:
                    print(f"📊 Batch {batch_id} - Status: {status}")

                # Save progress
                progress_data = {
                    "batch_id": batch_id,
                    "status": status,
                    "completed": completed,
                    "failed": failed,
                    "canceled": canceled,
                    "expired": expired,
                    "total": total,
                    "last_updated": datetime.now().isoformat()
                }
                self._save_progress(progress_data)

                # FIXED: Based on official documentation - only "ended" is completion status
                if status == "ended":
                    print(f"✅ Batch {batch_id} completed!")
                    return batch
                elif status in ["failed", "expired", "canceled"]:
                    print(f"❌ Batch {batch_id} finished with status: {status}")
                    return batch
                # Continue waiting for "in_progress" and other statuses

                time.sleep(check_interval)

            except KeyboardInterrupt:
                print(f"\n⚠️ Interrupted! Batch {batch_id} is still processing. You can resume later.")
                raise
            except Exception as e:
                print(f"Error checking batch status: {e}")
                time.sleep(check_interval)

    def _get_batch_results(self, batch_id: str) -> Dict[str, float]:
        """
        Retrieve and parse batch results using the official Anthropic SDK method.

        Args:
            batch_id: Batch ID to get results for

        Returns:
            Dictionary mapping custom_id to extracted scores
        """
        try:
            # Get batch information first
            batch = self._client.messages.batches.retrieve(batch_id)

            if not hasattr(batch, 'results_url') or not batch.results_url:
                raise RuntimeError("Batch results not available")

            print(f"📥 Retrieving results for batch {batch_id}...")

            # Use the SDK's built-in results method if available
            results = {}

            try:
                # Method 1: Try using SDK's results method (preferred)
                if hasattr(self._client.messages.batches, 'results'):
                    result_stream = self._client.messages.batches.results(batch_id)
                    for entry in result_stream:
                        custom_id = entry.custom_id
                        if entry.result.type == "succeeded":
                            response_text = entry.result.message.content[0].text
                            score = self._extract_score(response_text)
                            results[custom_id] = score
                        else:
                            error_info = getattr(entry.result, 'error', 'Unknown error')
                            print(f"Warning: Request {custom_id} failed: {error_info}")
                            results[custom_id] = 0.0
                else:
                    # Method 2: Download results from URL directly
                    import requests
                    response = requests.get(batch.results_url)
                    response.raise_for_status()

                    # Parse JSONL format (each line is a JSON object)
                    for line in response.text.strip().split('\n'):
                        if line.strip():
                            result = json.loads(line)
                            custom_id = result['custom_id']

                            if result['result']['type'] == 'succeeded':
                                response_text = result['result']['message']['content'][0]['text']
                                score = self._extract_score(response_text)
                                results[custom_id] = score
                            else:
                                print(f"Warning: Request {custom_id} failed: {result['result'].get('error', 'Unknown error')}")
                                results[custom_id] = 0.0

            except Exception as e:
                print(f"Error retrieving results: {e}")
                return {}

            print(f"✅ Retrieved {len(results)} results from batch {batch_id}")
            return results

        except Exception as e:
            print(f"Error retrieving batch results: {e}")
            return {}

    def _process_batch(
        self,
        hypotheses: List[str],
        sources: List[str],
        references: Optional[List[str]] = None
    ) -> List[float]:
        """
        Process evaluation pairs using batch API (simplified - single batch).

        Args:
            hypotheses: List of machine translation outputs
            sources: List of source texts
            references: List of reference translations (optional)

        Returns:
            List of evaluation scores
        """
        num_samples = len(hypotheses)
        all_scores = [0.0] * num_samples  # Initialize with zeros

        # Check for existing progress
        progress = self._load_progress()
        if progress and progress.get("status") == "in_progress":
            batch_id = progress.get("batch_id")
            if batch_id:
                print(f"🔄 Resuming batch {batch_id}...")
                try:
                    batch = self._wait_for_batch_completion(batch_id)
                    if batch.processing_status == "ended":
                        results = self._get_batch_results(batch_id)
                        # Parse results and update scores
                        for custom_id, score in results.items():
                            try:
                                index = int(custom_id.split('_')[-1])
                                if 0 <= index < num_samples:
                                    all_scores[index] = score
                            except (ValueError, IndexError):
                                print(f"Warning: Could not parse custom_id {custom_id}")

                        # Clean up progress file
                        if os.path.exists(self.progress_file):
                            os.remove(self.progress_file)

                        return all_scores
                except Exception as e:
                    print(f"Error resuming batch: {e}")
                    print("Starting fresh batch...")

        # Create new batch requests
        print(f"📦 Creating batch requests for {num_samples} samples...")
        requests = self._create_batch_requests(hypotheses, sources, references)

        try:
            # Create single batch (simplified)
            batch_id = self._create_batch(requests)

            # Wait for completion
            batch = self._wait_for_batch_completion(batch_id)

            if batch.processing_status == "ended":
                # Get results
                results = self._get_batch_results(batch_id)

                # Update scores
                for custom_id, score in results.items():
                    try:
                        index = int(custom_id.split('_')[-1])
                        if 0 <= index < num_samples:
                            all_scores[index] = score
                    except (ValueError, IndexError):
                        print(f"Warning: Could not parse custom_id {custom_id}")
            else:
                print(f"Warning: Batch failed with status {batch.processing_status}")
                # All scores remain 0.0

        except Exception as e:
            print(f"Error processing batch: {e}")
            # All scores remain 0.0

        # Clean up progress file
        if os.path.exists(self.progress_file):
            os.remove(self.progress_file)

        return all_scores

    def calculate_scores(
        self,
        hypotheses: List[str],
        references: List[str],
        sources: List[str] = None
    ) -> List[float]:
        """
        Calculate LLM-based scores for hypothesis-reference pairs using Batch API.

        Args:
            hypotheses: List of machine translation outputs
            references: List of reference translations (required for reference_based mode)
            sources: List of source texts (required for all modes)

        Returns:
            List of LLM quality scores (0-100 scale)

        Raises:
            ValueError: If required inputs are missing based on evaluation mode
        """
        # Validate inputs based on evaluation mode
        if sources is None:
            raise ValueError("Source texts are required for LLM evaluation")

        if self.evaluation_mode == "reference_based" and references is None:
            raise ValueError("Reference texts are required for reference-based evaluation")

        self.validate_inputs(hypotheses, references, sources)
        self._initialize_client()

        # Set batch_size to DataFrame size if not specified
        original_data_size = len(hypotheses)
        if self.batch_size is None:
            # Use all data by default, but respect Anthropic's limit
            self.batch_size = min(original_data_size, 100000)
            print(f"📊 Using DataFrame size as batch size: {self.batch_size} samples")
        else:
            # User specified a batch_size, respect it but apply limits
            self.batch_size = min(self.batch_size, 100000, original_data_size)
            if self.batch_size < original_data_size:
                print(f"📊 Using specified batch size: {self.batch_size} samples (out of {original_data_size})")

        # Apply batch_size limit to data
        num_samples = self.batch_size
        if num_samples < original_data_size:
            hypotheses = hypotheses[:num_samples]
            sources = sources[:num_samples]
            if references:
                references = references[:num_samples]

        print(f"Starting {self.metric_name} batch evaluation of {num_samples} samples...")
        print(f"Mode: {self.evaluation_mode}")
        print(f"Model: {self.model_name}")
        print(f"Batch size: {self.batch_size}")

        # Process using simplified batch API
        scores = self._process_batch(hypotheses, sources, references)

        # Extend scores to original length if needed (for consistency with other metrics)
        if len(scores) < original_data_size:
            avg_score = sum(scores) / len(scores) if scores else 0.0
            scores.extend([avg_score] * (original_data_size - len(scores)))
            print(f"Extended {len(scores)} LLM scores to match original data size using average: {avg_score:.2f}")

        return scores

    def run_from_cli(self) -> int:
        """
        Run the metric from command line interface.
        Overridden to add LLM-specific parameters.

        Returns:
            int: Exit code (0 for success, 1 for failure)
        """
        parser = argparse.ArgumentParser(
            description=f'Calculate {self.metric_name} scores for NMT evaluation using Batch API'
        )
        parser.add_argument(
            '--input', '-i',
            required=True,
            help='Path to input CSV file containing evaluation data'
        )
        parser.add_argument(
            '--output', '-o',
            required=True,
            help='Path to output JSON file for metric scores'
        )
        parser.add_argument(
            '--mode', '-m',
            choices=['reference_based', 'reference_free'],
            default='reference_based',
            help='Evaluation mode (default: reference_based)'
        )
        parser.add_argument(
            '--model',
            default='claude-sonnet-4-20250514',
            help='Anthropic model to use (default: claude-sonnet-4-20250514)'
        )
        parser.add_argument(
            '--api-key',
            help='Anthropic API key (default: from ANTHROPIC_API_KEY env var)'
        )
        parser.add_argument(
            '--batch-size', '-b',
            type=int,
            default=None,
            help='Number of samples to evaluate (default: all data, max: 100000)'
        )
        parser.add_argument(
            '--max-retries',
            type=int,
            default=3,
            help='Maximum number of retries for failed requests (default: 3)'
        )
        parser.add_argument(
            '--env-path',
            help='Path to .env file containing API keys and configuration'
        )
        parser.add_argument(
            '--progress-file',
            help='Path to save/load progress for resuming interrupted batches'
        )

        args = parser.parse_args()

        # Update instance parameters
        self.evaluation_mode = args.mode
        self.model_name = args.model
        self.env_path = args.env_path
        self.batch_size = args.batch_size  # Can be None - will be set in calculate_scores()

        if args.progress_file:
            self.progress_file = args.progress_file

        # Load .env file if specified
        if self.env_path:
            try:
                from dotenv import load_dotenv
                load_dotenv(self.env_path)
                print(f"✅ Loaded environment variables from: {self.env_path}")
            except ImportError:
                print("Warning: python-dotenv not installed, cannot load custom .env file")
            except Exception as e:
                print(f"Warning: Failed to load .env file {self.env_path}: {e}")

        self.api_key = args.api_key or os.getenv("ANTHROPIC_API_KEY")
        self.max_retries = args.max_retries

        # Update metric name based on mode
        mode_name = f"LLM-{self.evaluation_mode.replace('_', '-').title()}"
        self.metric_name = mode_name

        try:
            print(f"🚀 Running {self.metric_name} metric evaluation using Batch API...")
            print(f"  Mode: {self.evaluation_mode}")
            print(f"  Model: {self.model_name}")
            if self.batch_size:
                print(f"  Batch size: {self.batch_size}")
            else:
                print(f"  Batch size: Auto (will use all data)")

            # Load data
            print(f"Loading data from {args.input}...")
            data = self.load_data(args.input)
            print(f"Loaded {len(data)} evaluation pairs")

            # Extract text components
            hypotheses = data['targetText'].tolist()
            sources = data['sourceText'].tolist()
            references = data['referenceText'].tolist() if self.evaluation_mode == "reference_based" else None

            # Calculate scores
            print(f"Calculating {self.metric_name} scores...")
            scores = self.calculate_scores(hypotheses, references, sources)

            # Save results
            self.save_scores(scores, args.output)

            # Print summary statistics
            import numpy as np
            print(f"\n{self.metric_name} Score Statistics:")
            print(f"  Mean: {np.mean(scores):.4f}")
            print(f"  Std:  {np.std(scores):.4f}")
            print(f"  Min:  {np.min(scores):.4f}")
            print(f"  Max:  {np.max(scores):.4f}")

            return 0

        except Exception as e:
            print(f"❌ Error running {self.metric_name}: {e}")
            import traceback
            traceback.print_exc()
            return 1


def main():
    """Main function for CLI usage."""
    metric = LLMMetric()
    return metric.run_from_cli()


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
