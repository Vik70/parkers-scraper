#!/usr/bin/env python3
"""
Debug version of the specs agent to see what's happening.
"""

import pandas as pd

# Read the data
df = pd.read_excel('database_2.0_enriched.xlsx', sheet_name='Sheet2')

print(f'Total cars: {len(df)}')

# Check a few rows with missing values
missing_mpg = df[df["Min. of mpg low helper"] == 0]
print(f'Cars with missing MPG: {len(missing_mpg)}')

if len(missing_mpg) > 0:
    sample_row = missing_mpg.iloc[0]
    print(f'Sample missing MPG row:')
    print(f'  Make: {sample_row["Make"]}')
    print(f'  Model: {sample_row["Model"]}')
    print(f'  MPG: {sample_row["Min. of mpg low helper"]} (type: {type(sample_row["Min. of mpg low helper"])})')
    print(f'  Insurance: {sample_row["Min. of Insurance group"]} (type: {type(sample_row["Min. of Insurance group"])})')
    print(f'  0-60: {sample_row["Min. of 0-60 mph (secs)"]} (type: {type(sample_row["Min. of 0-60 mph (secs)"])})')
    
    # Test the condition logic
    needs_mpg = sample_row["Min. of mpg low helper"] == 0
    needs_insurance = sample_row["Min. of Insurance group"] == 0
    needs_060 = sample_row["Min. of 0-60 mph (secs)"] == 0
    
    print(f'  needs_mpg: {needs_mpg}')
    print(f'  needs_insurance: {needs_insurance}')
    print(f'  needs_060: {needs_060}')
    print(f'  Should call LLM: {needs_mpg or needs_insurance or needs_060}')

# Check if there are any NaN values
print(f'\nNaN values:')
print(f'  MPG NaN: {df["Min. of mpg low helper"].isna().sum()}')
print(f'  Insurance NaN: {df["Min. of Insurance group"].isna().sum()}')
print(f'  0-60 NaN: {df["Min. of 0-60 mph (secs)"].isna().sum()}')
