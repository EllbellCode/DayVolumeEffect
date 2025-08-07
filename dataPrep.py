import pandas as pd
import numpy as np
from pathlib import Path

def convertVol(volume_str):
    if pd.isna(volume_str):
        return None

    suffixes = {'k': 1e3, 'K': 1e3, 'm': 1e6, 'M': 1e6, 'b': 1e9, 'B': 1e9}
    if isinstance(volume_str, str) and volume_str[-1] in suffixes:
        return float(volume_str[:-1]) * suffixes[volume_str[-1]]

    return float(volume_str)

def standardize(data_dir):

    data_dir = Path(data_dir)
    output_dir = data_dir / 'Verified'
    output_dir.mkdir(exist_ok=True)

    for file in data_dir.glob("*.csv"):
        print(f"Processing {file.name}")
        df = pd.read_csv(file)

        # Convert date
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors='coerce')

        # Clean numeric columns
        for col in ['Price', 'Open', 'High', 'Low']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace(',', '', regex=False).astype(float)

        # Handle Volume
        if 'Vol.' in df.columns:
            df['Vol.'] = df['Vol.'].apply(convertVol)
            df['Vol.'] = df['Vol.'].fillna(999_999_999_999).round()

        # Save
        out_path = output_dir / f"Verif_{file.name}"
        df.to_csv(out_path, index=False)
        print(f"Saved to {out_path}")

def checkMissing(df, start, end):

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    expected_dates = pd.date_range(start=start, end=end).date
    actual_dates = df['Date'].dt.date.dropna().unique()

    return sorted(set(expected_dates) - set(actual_dates))

def interpolate(missing_dict, data_dir):
    data_dir = Path(data_dir)

    for file_name, missing_dates in missing_dict.items():
        path = data_dir / file_name
        df = pd.read_csv(path)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

        for miss_date in missing_dates:
            miss_dt = pd.to_datetime(miss_date)
            before = df[df['Date'] == miss_dt - pd.Timedelta(days=1)]
            after = df[df['Date'] == miss_dt + pd.Timedelta(days=1)]

            if not before.empty and not after.empty:
                new_row = {
                    'Date': miss_dt,
                    'Price': (before['Price'].values[0] + after['Price'].values[0]) / 2,
                    'Open': (before['Open'].values[0] + after['Open'].values[0]) / 2,
                    'High': (before['High'].values[0] + after['High'].values[0]) / 2,
                    'Low': (before['Low'].values[0] + after['Low'].values[0]) / 2,
                    'Vol.': (before['Vol.'].values[0] + after['Vol.'].values[0]) / 2
                }
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                print(f"Interpolated: {miss_date} in {file_name}")

        df.sort_values("Date", inplace=True)
        df.to_csv(path, index=False)

def calcReturns(data_dir):

    data_dir = Path(data_dir)

    for file in data_dir.glob("*.csv"):

        df = pd.read_csv(file)

        if 'Price' in df.columns:

            df['Returns'] = df['Price'].pct_change(periods=-1) * 100
            df['Log Returns'] = np.log(df['Price'] / df['Price'].shift(-1))
            df.to_csv(file, index=False)

def addDays(data_dir):

    data_dir = Path(data_dir)

    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']

    for file in data_dir.glob("*.csv"):

        df = pd.read_csv(file)

        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['Day'] = df['Date'].dt.day_name()

        for day in days:
            df[day] = (df['Day'] == day).astype(int)

        df.to_csv(file, index=False)

def volNorm(data_dir):

    data_dir = Path(data_dir)

    for file in data_dir.glob("*.csv"):

        df = pd.read_csv(file)

        df['Vol_Norm'] = (df['Vol.'] - df['Vol.'].mean()) / df['Vol.'].std()

        df.to_csv(file, index=False)



def main():

    data_folder = Path("Data")
    verif_folder = data_folder / "Verified"
    start_date = '2020-01-01'
    start_date_sol = '2020-07-21'
    end_date = '2025-01-01'
    missing_dates_dict = {}

    standardize(data_folder)

    for file in verif_folder.glob("*.csv"):
        df = pd.read_csv(file)
        start = start_date_sol if file.name == "Verif_SOL.csv" else start_date
        missing = checkMissing(df, start, end_date)
        if missing:
            missing_dates_dict[file.name] = missing

    print("Missing Dates:", missing_dates_dict)

    interpolate(missing_dates_dict, verif_folder)
    calcReturns(verif_folder)
    addDays(verif_folder)
    volNorm(verif_folder)

if __name__ == "__main__":
    main()






