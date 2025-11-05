import pandas as pd
import os

path = os.path.join('data', 'UMBRALES POR MES REGION.xlsx')
print('Path exists:', os.path.exists(path))
try:
    df = pd.read_excel(path, engine='openpyxl')
    print('Read OK. Columns:')
    print(list(df.columns))
    print('\nPreview (first 8 rows):')
    print(df.head(8).to_string(index=False))
    # print unique region values (up to 20)
    cols_lower = [c.strip().lower() for c in df.columns]
    print('\nColumns lowered:', cols_lower)
    # try detect region/mes/q33/q66
    lower_map = {c.strip().lower(): c for c in df.columns}
    for key in ['region','mes','q33_casos','q66_casos']:
        print(f"Has '{key}':", key in lower_map)
        if key in lower_map:
            print(f"Sample values for {key}:")
            print(df[lower_map[key]].dropna().unique()[:10])
except Exception as e:
    print('Error reading file:', e)
