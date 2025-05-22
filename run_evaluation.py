"""
Script to evaluate NMT metrics on Georgian dataset.

This script runs the evaluation of various machine translation metrics
(BLEU, CHRF++, TER, ROUGE, COMET) on a Georgian dataset and compares
them with human evaluation scores using correlation measures.
"""

import sys
import os

def main():
    """Run the evaluation script with default parameters."""
    try:
        # Add the project root to the Python path
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        
        # Import the evaluation module
        from src.evaluate_metrics import main as evaluate_main
        
        # Run the evaluation
        evaluate_main()
        
    except ImportError as e:
        print(f"Error importing modules: {e}")
        print("Make sure you have installed all required packages:")
        print("pip install -r requirements.txt")
        print("Also, make sure to download NLTK data: nltk.download('punkt')")
        return 1
    
    except Exception as e:
        print(f"Error running evaluation: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())