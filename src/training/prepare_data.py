import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description='Prepare train and validation datasets.')
    parser.add_argument('--input_train', required=True, help='Input CSV file for training data.')
    parser.add_argument('--input_valid', required=True, help='Input CSV file for validation data.')
    parser.add_argument('--output_train', required=True, help='Output CSV file for training data.')
    parser.add_argument('--output_valid', required=True, help='Output CSV file for validation data.')
    args = parser.parse_args()

    df = pd.read_csv(args.input_train)

    df['lp'] = 'en-ka'
    df = df.rename(columns={'sourceText': 'src', 'targetText': 'mt', 'referenceText': 'ref'})
    mean = df['llm_reference_based_score_noisy'].mean()
    std = df['llm_reference_based_score_noisy'].std()

    df['score'] = (df['llm_reference_based_score_noisy'] - mean) / std
    df = df.rename(columns={'llm_reference_based_score_noisy': 'raw'})
    df = df.drop(['llm_reference_based_score'], axis=1)
    df['annotators'] = 0
    df['domain'] = 'corp_dict'
    df.to_csv(args.output_train, index=False)

    valid_df = pd.read_csv(args.input_valid)
    valid_df['lp'] = 'en-ka'
    valid_df = valid_df.rename(columns={'sourceText': 'src', 'targetText': 'mt', 'referenceText': 'ref'})
    valid_df = pd.concat([g for _, g in valid_df.groupby('createdBy_id') if len(g) > 5])
    valid_df = valid_df.rename(columns={'score': 'raw'})
    mean = valid_df['raw'].mean()
    std = valid_df['raw'].std()

    valid_df['score'] = (valid_df['raw'] - mean) / std
    valid_df.to_csv(args.output_valid, index=False)

if __name__ == '__main__':
    main()