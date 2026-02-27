"""
Excel Files Combiner Script

Description:
    This script processes Excel files containing meteorological data (precipitation and temperature)
    organized in separate folders by variable type (Prec, TMax, TMed, TMin).
    It reads all .xls files from each folder, extracts monthly data for each year, and combines
    them into a single Excel file with a standardized structure.

Data Structure:
    Input: Individual Excel files with sheets named 'Datos' containing columns:
        - Año (Year)
        - Mínimo (Minimum value)
        - Media (Mean value)
        - Máximo (Maximum value)
    
    Output: combined_data.xlsx with columns:
        - Year
        - Month
        - Variable (precipitation, temperature_max, temperature_mean, temperature_min)
        - Minimum
        - Mean
        - Maximum

Usage:
    1. Organize your Excel files in subdirectories named: Prec/, TMax/, TMed/, TMin/
    2. Place them in the same directory as this script
    3. Run: python combinarExcels.py
    4. The combined_data.xlsx file will be generated in the same directory

Requirements:
    - pandas
    - openpyxl (for Excel support)

Note:
    - Files must be named with month abbreviations (Jan, Feb, Mar, Apr, May, Jun, Jul, Aug, Sep, Oct, Nov, Dec)
    - Excel files should be in .xls format
    - The 'Datos' sheet should have a header row and data starting from row 2
"""

import os
import pandas as pd
from pathlib import Path

# Configuration of folders and variables
BASE_FOLDER = Path(__file__).parent
VARIABLE_MAPPING = {
    'Prec': {
        'name': 'precipitation',
        'folder': BASE_FOLDER / 'Prec'
    },
    'TMax': {
        'name': 'temperature_max',
        'folder': BASE_FOLDER / 'TMax'
    },
    'TMed': {
        'name': 'temperature_mean',
        'folder': BASE_FOLDER / 'TMed'
    },
    'TMin': {
        'name': 'temperature_min',
        'folder': BASE_FOLDER / 'TMin'
    }
}

# Month to number mapping
MONTHS = {
    'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
    'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
}

def extract_month_from_filename(filename):
    """Extract the month from the filename"""
    for month_str, month_num in MONTHS.items():
        if month_str in filename:
            return month_num
    return None

def read_excel_data(file_path, variable_name):
    """Read data from an Excel file and return a DataFrame"""
    try:
        # Attempt to read without specifying engine (automatic detection)
        df = pd.read_excel(file_path, sheet_name='Datos', skiprows=1)
        
        # Clean whitespace from column names
        df.columns = df.columns.str.strip()
        
        # Ensure the expected columns are present
        expected_columns = ['Año', 'Mínimo', 'Media', 'Máximo']
        if not all(col in df.columns for col in expected_columns):
            print(f"Warning: Unexpected columns in {file_path}")
            print(f"Columns found: {df.columns.tolist()}")
            return None
        
        # Remove completely empty rows
        df = df.dropna(how='all')
        
        return df
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def process_all_excel_files():
    """Process all Excel files and generate the combined file"""
    
    combined_data = []
    
    # Iterate through each variable
    for var_key, var_info in VARIABLE_MAPPING.items():
        variable_name = var_info['name']
        folder = var_info['folder']
        
        if not folder.exists():
            print(f"Warning: folder {folder} does not exist")
            continue
        
        print(f"\nProcessing variable: {variable_name}")
        
        # Iterate through all .xls files in the folder
        for file in sorted(folder.glob('*.xls')):
            month_num = extract_month_from_filename(file.name)
            
            if month_num is None:
                print(f"  Could not extract month from: {file.name}")
                continue
            
            print(f"  Reading: {file.name} (month {month_num})")
            
            # Read the data from the Excel file
            df = read_excel_data(file, variable_name)
            
            if df is None:
                continue
            
            # Process each row (each year)
            for _, row in df.iterrows():
                try:
                    year = int(row['Año'])
                    minimum = float(row['Mínimo'])
                    mean = float(row['Media'])
                    maximum = float(row['Máximo'])
                    
                    combined_data.append({
                        'Year': year,
                        'Month': month_num,
                        'Variable': variable_name,
                        'Minimum': minimum,
                        'Mean': mean,
                        'Maximum': maximum
                    })
                except (ValueError, TypeError) as e:
                    print(f"    Error in row: {row.to_dict()} - {e}")
                    continue
    
    # Create final DataFrame
    df_final = pd.DataFrame(combined_data)
    
    if len(df_final) == 0:
        print("\nError: No data was successfully processed")
        return False
    
    # Sort by Year, Month, and Variable
    df_final = df_final.sort_values(['Year', 'Month', 'Variable']).reset_index(drop=True)
    
    # Save to Excel file
    output_file = BASE_FOLDER / 'combined_data.xlsx'
    
    try:
        df_final.to_excel(output_file, index=False, engine='openpyxl')
        print(f"\n✓ File successfully generated: {output_file}")
        print(f"  Total rows: {len(df_final)}")
        print(f"\nFirst rows:")
        print(df_final.head(10))
        return True
    except Exception as e:
        print(f"Error saving file: {e}")
        return False

if __name__ == '__main__':
    print("=" * 60)
    print("EXCEL FILES COMBINER")
    print("=" * 60)
    process_all_excel_files()
