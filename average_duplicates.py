"""Average repeated readings at the same date and time within each company.

Read data/all_data_outliers_removed.csv.
Save data/all_data_outliers_removed_and_duplicates_averaged.csv.
Keep one row per Company, Date, Time, using the mean MW of that group.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = PROJECT_ROOT / 'data' / 'all_data_outliers_removed.csv'
DEFAULT_OUTPUT = PROJECT_ROOT / 'data' / 'all_data_outliers_removed_and_duplicates_averaged.csv'
ORIGINAL_DATA = PROJECT_ROOT / 'data' / 'all_data.csv'
GROUP_COLUMNS = ['Date', 'Time', 'Company']


def average_duplicates(input_path=DEFAULT_INPUT, output_path=DEFAULT_OUTPUT):
    """Save one mean MW reading for each company's timestamp."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    if input_path.resolve() == output_path.resolve() or (
        output_path.exists() and input_path.samefile(output_path)
    ):
        raise ValueError('Input and output must be different files.')
    if output_path.resolve() == ORIGINAL_DATA.resolve() or (
        output_path.exists() and ORIGINAL_DATA.exists() and output_path.samefile(ORIGINAL_DATA)
    ):
        raise ValueError('The output must not overwrite data/all_data.csv.')

    data = pd.read_csv(input_path)
    if set(data.columns) != {'Date', 'Time', 'Company', 'MW'}:
        raise ValueError('Expected columns: Date, Time, Company, MW.')
    if data[GROUP_COLUMNS].isna().any().any() or data['Company'].astype(str).str.strip().eq('').any():
        raise ValueError('Date, Time, and Company must not be missing.')
    timestamps = pd.to_datetime(
        data['Date'].astype(str) + ' ' + data['Time'].astype(str),
        format='%Y-%m-%d %H:%M:%S', errors='raise',
    )
    if timestamps.isna().any():
        raise ValueError('Date and Time must not be missing.')
    data['Date'] = timestamps.dt.strftime('%Y-%m-%d')
    data['Time'] = timestamps.dt.strftime('%H:%M:%S')
    data['MW'] = pd.to_numeric(data['MW'], errors='raise')
    if not np.isfinite(data['MW']).all():
        raise ValueError('MW values must be finite and not missing.')

    duplicate_mask = data.duplicated(subset=GROUP_COLUMNS, keep=False)
    duplicate_groups = len(data.loc[duplicate_mask, GROUP_COLUMNS].drop_duplicates())
    extra_rows = int(data.duplicated(subset=GROUP_COLUMNS, keep='first').sum())
    # Group by company as well as timestamp so different companies stay separate.
    result = data.groupby(GROUP_COLUMNS, as_index=False, sort=False)['MW'].mean()
    result = result.loc[:, data.columns]
    if len(result) != len(data) - extra_rows or result.duplicated(subset=GROUP_COLUMNS).any():
        raise RuntimeError('Averaged row counts or duplicate checks failed.')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f'Input: {input_path}')
    print(f'Rows before: {len(data):,}')
    print(f'Duplicate timestamp groups within companies: {duplicate_groups:,}')
    print(f'Rows in duplicate groups: {int(duplicate_mask.sum()):,}')
    print(f'Extra rows combined: {extra_rows:,}')
    print(f'Rows after: {len(result):,}')
    print('Duplicate company/date/time rows remaining: 0')
    print(f'Output: {output_path}')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        average_duplicates(args.input, args.output)
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f'Error: {error}\n')


if __name__ == '__main__':
    main()
