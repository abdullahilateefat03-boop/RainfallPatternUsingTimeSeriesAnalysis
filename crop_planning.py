import pandas as pd
import numpy as np

def calculate_onset_cessation_for_year(year_df):
    """
    Calculates the onset and cessation dekad of the year (1 to 36) for a single year's dataframe.
    year_df must contain columns: 'dekad_of_year', 'rfh'
    Sorted by 'dekad_of_year' from 1 to 36.
    """
    # Pad the dataframe to have exactly 36 dekads (1 to 36) if any are missing to prevent IndexErrors
    if len(year_df) < 36 or not set(range(1, 37)).issubset(year_df['dekad_of_year']):
        template = pd.DataFrame({'dekad_of_year': range(1, 37)})
        # Keep track of column names to avoid duplication or loss
        cols_to_keep = [c for c in year_df.columns if c != 'dekad_of_year']
        year_df = pd.merge(template, year_df, on='dekad_of_year', how='left')
        if 'rfh_avg' in year_df.columns:
            year_df['rfh'] = year_df['rfh'].fillna(year_df['rfh_avg'])
        year_df['rfh'] = year_df['rfh'].fillna(0.0)
        
    year_df = year_df.sort_values('dekad_of_year').reset_index(drop=True)
    rfh = year_df['rfh'].values
    dekads = year_df['dekad_of_year'].values
    
    # 1. ONSET LOGIC (Search starting from April 1st, which is Dekad 10)
    # Walter/Benoit standard definition:
    # First dekad starting from dekad 10 where:
    # - Cumulative rainfall of current and next dekad >= 20mm
    # - No subsequent severe dry spell in the next 3 dekads (we define dry spell as any of the next 3 dekads having < 5mm)
    onset_dekad = None
    for i in range(9, 34):  # Index 9 corresponds to dekad 10
        current_rain = rfh[i]
        next_rain = rfh[i+1]
        
        # Check cumulative rain threshold
        if (current_rain + next_rain) >= 20.0:
            # Check for dry spell in the next 3 dekads (i+2, i+3, i+4)
            dry_spell = False
            for j in range(i+2, min(i+5, 36)):
                if rfh[j] < 5.0:
                    dry_spell = True
                    break
            
            if not dry_spell:
                onset_dekad = dekads[i]
                break
                
    # Fallback if no dekad satisfies the strict rule
    if onset_dekad is None:
        # Fallback to the first dekad after dekad 10 where rain > 10mm
        for i in range(9, 36):
            if rfh[i] > 10.0:
                onset_dekad = dekads[i]
                break
        if onset_dekad is None:
            onset_dekad = 12  # April 3rd dekad default
            
    # 2. CESSATION LOGIC (Search starting from September 1st, which is Dekad 25)
    # Standard definition:
    # The first dekad after September 1st (index 24) where for all subsequent dekads, rainfall remains below a threshold (e.g., < 10mm)
    cessation_dekad = None
    for i in range(24, 36):
        # Check if all dekads from i to the end of the year have rain < 10mm
        if all(rfh[j] < 10.0 for j in range(i, 36)):
            cessation_dekad = dekads[i]
            break
            
    # Fallback: scan backward from end of year, find first dekad with rain >= 10mm, cessation is the next one
    if cessation_dekad is None:
        for i in range(35, 23, -1):
            if rfh[i] >= 10.0:
                cessation_dekad = dekads[min(i + 1, 35)]
                break
        if cessation_dekad is None:
            cessation_dekad = 30  # October 3rd dekad default
            
    return int(onset_dekad), int(cessation_dekad)

def dekad_to_date_str(dekad, year=None):
    """
    Converts a dekad number (1 to 36) to a human-readable date string.
    Example: 13 -> "May 01" (or "May 01, 2026" if year is provided)
    """
    month_names = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", 
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
    ]
    month_idx = (dekad - 1) // 3
    dekad_idx = (dekad - 1) % 3
    
    day_str = "01" if dekad_idx == 0 else ("11" if dekad_idx == 1 else "21")
    month_str = month_names[month_idx]
    
    if year:
        return f"{month_str} {day_str}, {year}"
    return f"{month_str} {day_str}"

def extract_historical_events(df):
    """
    Extracts onset and cessation dates for all years in the dataframe.
    """
    df_clean = df.copy()
    df_clean['year'] = df_clean['date'].dt.year
    
    # Calculate dekad of year
    df_clean['month'] = df_clean['date'].dt.month
    df_clean['dekad_in_month'] = df_clean['date'].dt.day.map(lambda d: 1 if d <= 10 else (2 if d <= 20 else 3))
    df_clean['dekad_of_year'] = (df_clean['month'] - 1) * 3 + df_clean['dekad_in_month']
    
    records = []
    for year, group in df_clean.groupby('year'):
        # We need a full year of data to compute onset/cessation reliably
        if len(group) == 36:
            onset, cessation = calculate_onset_cessation_for_year(group)
            records.append({
                'year': year,
                'onset_dekad': onset,
                'cessation_dekad': cessation,
                'onset_date': dekad_to_date_str(onset, year),
                'cessation_date': dekad_to_date_str(cessation, year),
                'season_length_dekads': cessation - onset
            })
            
    return pd.DataFrame(records)

if __name__ == "__main__":
    # Test on historical data
    from data_pipeline import load_data
    print("Testing crop planning calculations...")
    df = load_data()
    events_df = extract_historical_events(df)
    print("\nHistorical Onset & Cessation Dates (Last 10 Years):")
    print(events_df.tail(10).to_string(index=False))
