import pandas as pd

# Check the final results
df = pd.read_excel('database_2.0_specs_final.xlsx')

print('Final Results:')
print(f'Total cars: {len(df)}')
print(f'Missing MPG: {(df["Min. of mpg low helper"] == 0).sum()}')
print(f'Missing Insurance: {(df["Min. of Insurance group"] == 0).sum()}')
print(f'Missing 0-60: {(df["Min. of 0-60 mph (secs)"] == 0).sum()}')

print(f'\nValue ranges:')
print(f'MPG range: {df["Min. of mpg low helper"].min()} - {df["Min. of mpg low helper"].max()}')
print(f'Insurance range: {df["Min. of Insurance group"].min()} - {df["Min. of Insurance group"].max()}')
print(f'0-60 range: {df["Min. of 0-60 mph (secs)"].min()} - {df["Min. of 0-60 mph (secs)"].max()}')

print('\nSample of filled values:')
sample = df[df["Min. of mpg low helper"] > 0].head(5)
print(sample[['Make', 'Model', 'Min. of mpg low helper', 'Min. of Insurance group', 'Min. of 0-60 mph (secs)']].to_string())
