import pandas as pd
import numpy as np
import pickle
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv('laptop_data.csv')

# Show available columns for debug
print("🔎 Columns in dataset:", df.columns.tolist())

# Clean RAM and Weight
df['Ram'] = df['Ram'].str.replace('GB', '').astype(int)
df['Weight'] = df['Weight'].str.replace('kg', '').astype(float)

# Extract Touchscreen and IPS info from ScreenResolution
df['Touchscreen'] = df['ScreenResolution'].apply(lambda x: 1 if 'Touchscreen' in x else 0)
df['IPS'] = df['ScreenResolution'].apply(lambda x: 1 if 'IPS' in x else 0)

# Extract resolution
res = df['ScreenResolution'].str.extract(r'(\d+)x(\d+)')
df['X_res'] = res[0].astype(float)
df['Y_res'] = res[1].astype(float)

# Calculate PPI
df['PPI'] = ((df['X_res']**2 + df['Y_res']**2) ** 0.5) / df['Inches']

# Drop processed columns
df.drop(columns=['ScreenResolution', 'Inches', 'X_res', 'Y_res'], inplace=True)

# Extract CPU brand
df['Cpu brand'] = df['Cpu'].apply(lambda x: x.split()[0])
df.drop(columns=['Cpu'], inplace=True)

# Extract GPU brand
df['Gpu brand'] = df['Gpu'].apply(lambda x: x.split()[0])
df.drop(columns=['Gpu'], inplace=True)

# Rename OS column
df['os'] = df['OpSys']
df.drop(columns=['OpSys'], inplace=True)

# Rename or create HDD/SSD if needed
if 'HDD' not in df.columns:
    df['HDD'] = 0
if 'SSD' not in df.columns:
    df['SSD'] = 0

# Drop missing
df.dropna(inplace=True)

# Log-transform target
df['Price'] = np.log(df['Price'])

# Split
X = df.drop(columns=['Price'])
y = df['Price']

# Save df for Streamlit app
with open('df.pkl', 'wb') as f:
    pickle.dump(X, f)

# Define preprocessing
cat_cols = ['Company', 'TypeName', 'Cpu brand', 'Gpu brand', 'os']
num_cols = ['Ram', 'Weight', 'Touchscreen', 'IPS', 'PPI', 'HDD', 'SSD']

# Build pipeline
ct = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
    ('num', StandardScaler(), num_cols)
])

pipe = Pipeline([
    ('transform', ct),
    ('model', LinearRegression())
])

# Fit and save
pipe.fit(X, y)

with open('pipe.pkl', 'wb') as f:
    pickle.dump(pipe, f)

print("✅ Model and df saved successfully.")
