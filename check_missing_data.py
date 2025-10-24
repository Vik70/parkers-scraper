import pandas as pd

# Read the data
df = pd.read_excel('database_2.0_enriched.xlsx', sheet_name='Sheet2')

print(f'Total cars: {len(df)}')
print(f'Missing mpg low: {df["Min. of mpg low helper"].isna().sum()}')
print(f'Missing insurance group: {df["Min. of Insurance group"].isna().sum()}')
print(f'Missing 0-60: {df["Min. of 0-60 mph (secs)"].isna().sum()}')

# Show some sample data
print("\nSample columns:")
print(df[['Make', 'Model', 'Series (production years start-end)', 'Min. of mpg low helper', 'Min. of Insurance group', 'Min. of 0-60 mph (secs)']].head())
