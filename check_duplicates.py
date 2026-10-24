"""Check for duplicate rows in data/cleaned.

Ignore saved index columns (Unnamed:*). All other values must match for a row
to count as a duplicate. This script only reads the files.
"""

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_INPUT_DIR = Path(__file__).resolve().parent / 'data' / 'cleaned_two'


def count_duplicate_rows(data):
    """Count repeated rows, ignoring saved CSV indices."""
    values = data.loc[:, ~data.columns.astype(str).str.startswith('Unnamed:')]
    return int(values.duplicated(keep='first').sum())


def check_duplicates(input_dir=DEFAULT_INPUT_DIR):
    """Print row counts and duplicate counts for each CSV."""
    input_dir = Path(input_dir)
    paths = sorted(input_dir.glob('*.csv'))
    if not paths:
        raise ValueError(f'No CSV files found in {input_dir}')

    report = {}
    print(f'Checking CSV files in {input_dir}')
    for path in paths:
        data = pd.read_csv(path)
        duplicates = count_duplicate_rows(data)
        report[path.name] = {'rows': len(data), 'duplicates': duplicates}
        print(f'  {path.name}: {len(data):,} rows; {duplicates:,} duplicate rows')

    total_rows = sum(item['rows'] for item in report.values())
    total_duplicates = sum(item['duplicates'] for item in report.values())
    files_with_duplicates = sum(item['duplicates'] > 0 for item in report.values())
    print(f'\nFiles checked: {len(report):,}')
    print(f'Total rows: {total_rows:,}')
    print(f'Total duplicate rows within files: {total_duplicates:,}')
    print(f'Files with duplicates: {files_with_duplicates:,}')
    print('Duplicates found.' if total_duplicates else 'No duplicate rows found.')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=DEFAULT_INPUT_DIR)
    args = parser.parse_args()
    try:
        report = check_duplicates(args.input_dir)
    except (ValueError, OSError) as error:
        parser.exit(2, f'Error: {error}\n')
    if any(item['duplicates'] for item in report.values()):
        parser.exit(1)


if __name__ == '__main__':
    main()
