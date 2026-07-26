import pandas as pd
import sys
import logging

# Configure robust logging for monitoring terminal execution
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Use the reliable, flat subnational baseline repository link
DATA_URL = "https://data.humdata.org/dataset/ff5dba07-397b-44fb-b04f-bbca782ce793/resource/114874de-df99-4102-b4c8-b44e2db44a5e/download/nga-rainfall-subnat-full.csv"
OUTPUT_FILE = "kogi_chirps_monthly.csv"

# OCHA/WFP Standard P-Code for Kogi State, Nigeria
KOGI_PCODE = "NG023" 

def extract_kogi_timeline(url, output_path, target_pcode):
    try:
        logging.info("Streaming subnational rainfall records from repository...")
        df_national = pd.read_csv(url)
        logging.info(f"Download complete. Global matrix shape: {df_national.shape}")
        
    except Exception as e:
        logging.error(f"Failed to fetch data from remote repository: {str(e)}")
        sys.exit(1)

    try:
        logging.info("Validating structural attributes for spatial slice...")
        if 'PCODE' not in df_national.columns:
            raise KeyError(f"Expected tracking key 'PCODE' missing from file schema. Found: {list(df_national.columns)}")
        
        # Coerce values to string and strip spaces to prevent matching mismatches
        df_national['PCODE'] = df_national['PCODE'].astype(str).str.strip()
        
        logging.info(f"Filtering dataset rows for Kogi State using geospatial footprint key: '{target_pcode}'...")
        df_kogi = df_national[df_national['PCODE'] == target_pcode].copy()
        
        if df_kogi.empty:
            # Fallback check: if the file uses numeric IDs instead of country strings
            logging.warning(f"No exact match for string P-Code '{target_pcode}'. Attempting numeric fallback...")
            df_kogi = df_national[df_national['adm_id'] == 2223].copy()
            
        if df_kogi.empty:
            raise ValueError(f"Geospatial slicing returned 0 rows for Kogi parameters. Available unique codes: {df_national['PCODE'].unique()[:5]}")
            
        # Ensure the temporal series is ordered logically before downstream modeling
        if 'date' in df_kogi.columns:
            df_kogi['date'] = pd.to_datetime(df_kogi['date'])
            df_kogi = df_kogi.sort_values(by='date')
            
        logging.info(f"Success! Isolated {len(df_kogi)} historical time-series intervals for Kogi State.")
        
    except Exception as e:
        logging.error(f"Data frame filtering pipeline failed: {str(e)}")
        sys.exit(1)

    try:
        logging.info(f"Writing structured historical baseline data matrix to disk: '{output_path}'")
        df_kogi.to_csv(output_path, index=False)
        logging.info("Pipeline successful. Your Chapter Three baseline dataset is completely ready.")
        
    except IOError as e:
        logging.error(f"File system I/O error while writing file: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    extract_kogi_timeline(DATA_URL, OUTPUT_FILE, KOGI_PCODE)