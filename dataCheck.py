import pandas as pd
import os

def convert_volume_to_number(volume_str):
    """Convert volume strings with suffixes to numeric values."""
    if pd.isna(volume_str):  # Check for NaN values
        return None  # or return 0 if you prefer to replace NaN with 0

    if isinstance(volume_str, str):
        if volume_str.endswith('k') or volume_str.endswith('K'):
            return float(volume_str[:-1]) * 1000  # Convert 'k' to thousands
        elif volume_str.endswith('M') or volume_str.endswith('m'):
            return float(volume_str[:-1]) * 1000000  # Convert 'M' to millions
        elif volume_str.endswith('B') or volume_str.endswith('b'):
            return float(volume_str[:-1]) * 1000000000  # Convert 'B' to billions
        else:
            # Handle cases where the string might have commas or other characters
            volume_str = volume_str.replace(',', '')  # Remove commas
            try:
                return float(volume_str)  # Convert to float
            except ValueError:
                print(f"Warning: Could not convert '{volume_str}' to float.")
                return None  # Return None or some default value if conversion fails
    return float(volume_str)  # Return as float if no suffix


def standardize_csv_files(data_folder):
    # Create a new subfolder for standardized files
    output_folder = os.path.join(data_folder, 'Verified')
    os.makedirs(output_folder, exist_ok=True)

    # Get a list of all CSV files in the specified folder
    for filename in os.listdir(data_folder):
        if filename.endswith('.csv'):
            file_path = os.path.join(data_folder, filename)
            print(f"Processing file: {filename}")

            # Read the CSV file
            df = pd.read_csv(file_path)

            # Standardize the specified columns
            for column in ['Price', 'Open', 'High', 'Low']:
                if column in df.columns:
                    # Convert the column to string type to check for commas
                    df[column] = df[column].astype(str)

                    # Check if commas are present in the column
                    if df[column].str.contains(',').any():
                        print(f"Commas found in column '{column}' of {filename}. Standardizing...")
                        # Remove commas and convert to float
                        df[column] = df[column].str.replace(',', '').astype(float)

            # Standardize the 'Vol.' column
            if 'Vol.' in df.columns:
                print(f"Standardizing 'Vol.' column in {filename}...")
                df['Vol.'] = df['Vol.'].apply(convert_volume_to_number)
                df['Vol.'] = df['Vol.'].apply(lambda x: round(x) if pd.notna(x) else None)  # Handle NaN

            # Save the modified DataFrame to the new subfolder
            new_file_path = os.path.join(output_folder, f'Verif_{filename}') 
            df.to_csv(new_file_path, index=False)
            print(f"Standardized columns in {filename} and saved to {new_file_path}.")


def check_date_coverage(df, start_date, end_date):
    # Ensure the 'Date' column is in datetime format
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # Generate the complete date range
    complete_date_range = pd.date_range(start=start_date, end=end_date)

    # Convert DataFrame dates to a set of dates for comparison
    df_dates_set = set(df['Date'].dt.date.dropna())  # Convert to date and drop NaT

    # Identify missing dates
    missing_dates = [date for date in complete_date_range.date if date not in df_dates_set]

    return missing_dates  # Return the list of missing dates


def interpolate_missing_values(missing_dates_dict, data_directory):

    for filename, missing_dates in missing_dates_dict.items():
        # Read the original DataFrame
        df = pd.read_csv(os.path.join(data_directory, filename))
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

        for missing_date in missing_dates:
            missing_date_dt = pd.to_datetime(missing_date)

            # Find the dates before and after the missing date
            before_date = missing_date_dt - pd.Timedelta(days=1)
            after_date = missing_date_dt + pd.Timedelta(days=1)

            # Check if both surrounding dates are in the DataFrame
            before_row = df[df['Date'] == before_date]
            after_row = df[df['Date'] == after_date]

            if not before_row.empty and not after_row.empty:
                # Check if required columns exist
                required_columns = ['Price', 'Open', 'High', 'Low', 'Vol.']
                if all(col in df.columns for col in required_columns):
                    # Both surrounding dates are available
                    avg_price = (before_row['Price'].values[0] + after_row['Price'].values[0]) / 2
                    avg_open = (before_row['Open'].values[0] + after_row['Open'].values[0]) / 2
                    avg_high = (before_row['High'].values[0] + after_row['High'].values[0]) / 2
                    avg_low = (before_row['Low'].values[0] + after_row['Low'].values[0]) / 2
                    avg_volume = (before_row['Vol.'].values[0] + after_row['Vol.'].values[0]) / 2

                    # Create a new DataFrame for the new row
                    new_row = pd.DataFrame({
                        'Date': [missing_date_dt],
                        'Price': [avg_price],
                        'Open': [avg_open],
                        'High': [avg_high],
                        'Low': [avg_low],
                        'Vol.': [avg_volume]
                    })

                    # Print the interpolated date and data
                    print(f"Interpolated Date: {missing_date_dt.date()}")
                    print(new_row)

                    # Concatenate the new row to the original DataFrame
                    df = pd.concat([df, new_row], ignore_index=True)

        # Sort the DataFrame by date
        df.sort_values('Date', inplace=True)

        # Save the updated DataFrame back to the original CSV file
        df.to_csv(os.path.join(data_directory, filename), index=False)
        print(f"Updated data saved to {filename}")

# Example usage
data_folder = 'Data'  # Path to the folder containing CSV files
standardize_csv_files(data_folder)

start_date = '2020-01-01'
start_date_sol = '2020-07-21'
end_date = '2025-01-01'

missing_dates_dict = {}  # Initialize a dictionary to store missing dates

for filename in os.listdir(data_folder + '/Verified'):
    if filename.endswith('.csv'):
        df = pd.read_csv(data_folder + '/Verified' + '/' + filename)

        if filename == 'SOL.csv':
            missing_dates = check_date_coverage(df, start_date_sol, end_date)
        else:
            missing_dates = check_date_coverage(df, start_date, end_date)

        if missing_dates:  # If there are missing dates, add them to the dictionary
            missing_dates_dict[filename] = missing_dates

print(missing_dates_dict)

interpolate_missing_values(missing_dates_dict, 'Data/Verified')


