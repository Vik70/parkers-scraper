import pandas as pd

# Check the final results
df = pd.read_excel('database_2.0_specs_complete_fixed.xlsx')

print('Sample results:')
print(df[['Make', 'Model', 'Min. of mpg low helper', 'Min. of Insurance group', 'Min. of 0-60 mph (secs)']].head(10))

print(f'\nMissing MPG: {(df["Min. of mpg low helper"] == 0).sum()}')
print(f'Missing Insurance: {(df["Min. of Insurance group"] == 0).sum()}')
print(f'Missing 0-60: {(df["Min. of 0-60 mph (secs)"] == 0).sum()}')

# Check if any values were actually updated
print(f'\nMPG values range: {df["Min. of mpg low helper"].min()} - {df["Min. of mpg low helper"].max()}')
print(f'Insurance group range: {df["Min. of Insurance group"].min()} - {df["Min. of Insurance group"].max()}')
print(f'0-60 range: {df["Min. of 0-60 mph (secs)"].min()} - {df["Min. of 0-60 mph (secs)"].max()}')

# Let's check a few specific rows that should have missing values
print('\nChecking rows with missing values:')
missing_mpg = df[df["Min. of mpg low helper"] == 0]
if len(missing_mpg) > 0:
    print(f'First missing MPG row: {missing_mpg[["Make", "Model", "Min. of mpg low helper", "Min. of Insurance group", "Min. of 0-60 mph (secs)"]].iloc[0].to_dict()}')
