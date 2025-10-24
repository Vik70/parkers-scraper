import pandas as pd
import numpy as np

# Read the data
df = pd.read_excel('database_2.0_enriched.xlsx', sheet_name='Sheet2')

print(f'Total cars: {len(df)}')

# Check for various types of missing/empty values
def check_column(col_name):
    print(f"\n{col_name}:")
    print(f"  Total rows: {len(df)}")
    print(f"  NaN values: {df[col_name].isna().sum()}")
    print(f"  Empty strings: {(df[col_name] == '').sum()}")
    print(f"  Zero values: {(df[col_name] == 0).sum()}")
    print(f"  Negative values: {(df[col_name] < 0).sum()}")
    print(f"  Unique values: {df[col_name].nunique()}")
    print(f"  Sample values: {df[col_name].head(10).tolist()}")

check_column("Min. of mpg low helper")
check_column("Min. of Insurance group") 
check_column("Min. of 0-60 mph (secs)")

# Check if there are any obviously wrong values
print(f"\nMPG values range: {df['Min. of mpg low helper'].min()} - {df['Min. of mpg low helper'].max()}")
print(f"Insurance group range: {df['Min. of Insurance group'].min()} - {df['Min. of Insurance group'].max()}")
print(f"0-60 range: {df['Min. of 0-60 mph (secs)'].min()} - {df['Min. of 0-60 mph (secs)'].max()}")
