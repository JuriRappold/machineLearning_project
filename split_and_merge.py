"""Split Datetime, add company names, and combine the cleaned CSVs.

Input: data/cleaned/*.csv
Output: data/split/date_time_<filename>.csv and data/all_data.csv

Check for duplicate rows before saving. Repeated timestamps with different MW
values are kept. The output columns are Date, Time, Company, MW. Company names
come from the filename, e.g. AEP_hourly_CLEANED.csv becomes AEP.
"""

import argparse
from pathlib import Path

import pandas as pd

from check_duplicates import count_duplicate_rows


PROJECT_ROOT = Path(__file__).resolve().parent


def split_and_merge(input_dir=None, merged_path=None):
    """Save the split and combined files, then check their row counts."""
    input_dir = Path(input_dir) if input_dir is not None else PROJECT_ROOT / 'data' / 'cleaned'
    #split_dir = Path(split_dir) if split_dir is not None else PROJECT_ROOT / 'data' / 'split'
    merged_path = Path(merged_path) if merged_path is not None else PROJECT_ROOT / 'data' / 'all_data.csv'
    input_paths = sorted(input_dir.glob('*.csv'))
    if not input_paths:
        raise ValueError(f'No CSV files found in {input_dir}')
    #if input_dir.resolve() == split_dir.resolve():
    #    raise ValueError('Input and split directories must be different.')
    if merged_path.resolve().parent == input_dir.resolve():
        raise ValueError('The merged file must not be saved in the input directory.')
    #split_paths = [split_dir / f'date_time_{path.name}' for path in input_paths]
    #if merged_path.resolve() in {path.resolve() for path in split_paths}:
    #    raise ValueError('The merged filename must not overwrite a split file.')

    frames = []
    duplicate_files = []
    input_counts = {}
    print('Input row counts and duplicate checks (excluding saved CSV index columns):')
    for path in input_paths:
        data = pd.read_csv(path)
        data = data.loc[:, ~data.columns.str.startswith('Unnamed:')]
        duplicate_count, dupli_rows = count_duplicate_rows(data)
        input_counts[path.name] = len(data)
        print(f'  {path.name}: {len(data):,} rows; {duplicate_count:,} duplicate rows')
        if duplicate_count:
            duplicate_files.append(path.name)
        frames.append(data)

    input_total = sum(input_counts.values())
    print(f'Total input rows: {input_total:,}')
    if duplicate_files:
        raise ValueError('Duplicates found; no outputs written. Check: ' + ', '.join(duplicate_files))

    # Check all files before saving.
    for path, data in zip(input_paths, frames):
        if 'Datetime' not in data:
            raise ValueError(f'{path.name}: missing Datetime column; no outputs written.')
        value_columns = [column for column in data.columns if column.endswith('MW')]
        if len(value_columns) != 1 or len(data.columns) != 2:
            raise ValueError(
                f'{path.name}: expected only Datetime and one MW column; no outputs written.'
            )
        company = path.stem.removesuffix('_hourly_CLEANED')
        if not company:
            raise ValueError(f'{path.name}: missing company name in the filename; no outputs written.')
        try:
            timestamps = pd.to_datetime(data['Datetime'], errors='raise')
        except (ValueError, TypeError) as error:
            raise ValueError(f'{path.name}: invalid Datetime values; no outputs written.') from error
        if timestamps.isna().any():
            raise ValueError(f'{path.name}: missing Datetime values; no outputs written.')
        if ((timestamps.dt.microsecond != 0) | (timestamps.dt.nanosecond != 0)).any():
            raise ValueError(f'{path.name}: subsecond timestamps would lose precision; no outputs written.')
        data.insert(0, 'Company', company)


    input_total = sum(len(data) for data in frames)
    print(f'Total rows before merging: {input_total:,}')

    # Stack the rows without dropping any.
    print(len(frames) == 12)
    merged = pd.concat(frames, ignore_index=True, sort=False)
    if len(merged) != input_total:
        raise RuntimeError('Merged row count does not match the input/ totals.')
    merged_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(merged_path, index=False)
    final_total = len(pd.read_csv(merged_path, usecols=['Datetime']))
    if final_total != input_total:
        raise RuntimeError('The saved merged CSV row count does not match the input total.')

    print(f'\nMerged file: {merged_path}')
    print(f'Total input rows:         {input_total:,}')
    print(f'Final merged rows:       {final_total:,}')
    print('Row totals match. No rows were dropped.')
    return {
        'input_counts': input_counts,
        'input_total': input_total,
        'merged_total': final_total,
    }


def merge():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=PROJECT_ROOT / 'data' / 'cleaned')
    parser.add_argument('--merged-path', type=Path, default=PROJECT_ROOT / 'data' / 'all_data.csv')
    args = parser.parse_args()
    try:
        split_and_merge(args.input_dir, args.merged_path)
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f'Error: {error}\n')


if __name__ == '__main__':
    merge()
