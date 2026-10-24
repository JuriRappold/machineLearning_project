"""Change PJME, PJMW, and PJM_Load to PJM.

Read data/all_data.csv and save a separate copy as data/all_data_relabelled.csv.
"""

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
COMPANY_LABELS = {'PJME': 'PJM', 'PJMW': 'PJM', 'PJM_Load': 'PJM'}


def relabel_companies(input_path=None, output_path=None):
    """Save the new labels and return the number of rows changed."""
    input_path = Path(input_path) if input_path is not None else PROJECT_ROOT / 'data' / 'all_data.csv'
    output_path = Path(output_path) if output_path is not None else PROJECT_ROOT / 'data' / 'all_data_relabelled.csv'
    if input_path.resolve() == output_path.resolve() or (
        output_path.exists() and input_path.samefile(output_path)
    ):
        raise ValueError('Input and output must be different files; the input will not be overwritten.')

    # Keep the original cell formatting when reading the CSV.
    data = pd.read_csv(input_path, dtype=str, keep_default_na=False)
    if 'Company' not in data.columns:
        raise ValueError('The input CSV must contain a Company column.')
    before = data['Company'].value_counts(sort=False)
    changed_rows = int(data['Company'].isin(COMPANY_LABELS).sum())
    data['Company'] = data['Company'].replace(COMPANY_LABELS)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)
    print(f'Input (unchanged): {input_path}')
    print(f'Output: {output_path}')
    print(f'Rows relabelled: {changed_rows:,}')
    print(f'Total rows retained: {len(data):,}')
    print('\nCompany counts before:')
    print(before.to_string())
    print('\nCompany counts after:')
    print(data['Company'].value_counts(sort=False).to_string())
    return changed_rows


def relabel():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=PROJECT_ROOT / 'data' / 'all_data.csv')
    parser.add_argument('--output', type=Path, default=PROJECT_ROOT / 'data' / 'all_data_relabelled.csv')
    args = parser.parse_args()
    try:
        relabel_companies(args.input, args.output)
    except (ValueError, OSError) as error:
        parser.exit(1, f'Error: {error}\n')


if __name__ == '__main__':
    relabel()
