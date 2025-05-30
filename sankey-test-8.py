
import streamlit as st
import pandas as pd
import yfinance as yf
from sqlalchemy import create_engine
import plotly.graph_objects as go
from yfinance import Ticker
import requests
import time
from functools import lru_cache
import mammoth
import io
import pandas as pd
import requests
import datetime
from dateutil.relativedelta import relativedelta
import streamlit as st
import fitz  # PyMuPDF
from PIL import Image
import io



st.set_page_config(layout="wide")


company_tickers = {
    "AMERICAN EXPRESS CO": "AXP",
    "ANALOG DEVICES INC": "ADI",
    "ASTRAZENECA PLC": "AZN",
    "BIOGEN INC": "BIIB",
    "BRISTOL-MYERS SQUIBB CO": "BMY",
    "THE CIGNA GROUP": "CI",
    "CUMMINS INC": "CMI",
    "DISNEY WALT CO": "DIS",
    "ECOLAB INC": "ECL",
    "ENTERGY CORP": "ETR",
    "FEDERAL HOME LOAN BANKS": "Agency Bonds", 
    "FEDERAL FARM CREDIT BANKS": "Agency Bonds", 
    "GILEAD SCIENCES INC": "GILD",
    "HUMANA INC": "HUM",
    "JOHNSON & JOHNSON": "JNJ",
    "LABORATORY CORP AMERICA HOLDIN": "LH",
    "MASTERCARD INCORPORATED": "MA",
    "MCKESSON CORP": "MCK",
    "UNION PACIFIC CORP": "UNP",
    "UNITED STATES TREAS BILLS": "Agency Bonds",  
    "UNITED STATES TREAS NOTES": "Agency Bonds",  
    "UNITEDHEALTH GROUP INC": "UNH",
    "VISA INC": "V",
    "VODAFONE GROUP PLC": "VOD"
}

# Initialize session state variables
if 'df_stock_info' not in st.session_state:
    st.session_state.df_stock_info = pd.DataFrame(columns=[
        'Stock', 'Units', 'Purchase Date', 'Current Price ($)', 'Initial Investment ($)', 
        'Gain/Loss ($)', 'Gain/Loss (%)', 'Portfolio Allocation', 'Sector'
    ])

# Initialize session state for bond holdings if not exists
if 'df_bonds' not in st.session_state:
    st.session_state.df_bonds = pd.DataFrame(columns=[
        'CUSIP', 'Name', 'Industry Group', 'Issuer', 'Units', 'Purchase Price',
        'Purchase Date', 'Current Price', 'Coupon', 'Maturity Date', 'YTM',
        'Market Value', 'Total Cost', 'Price Return', 'Income Return', 'Total Return'
    ])

if 'df_bonds_scores' not in st.session_state:
    st.session_state.df_bonds_scores = None
# Cache for API responses - persists across reruns
if 'sector_cache' not in st.session_state:
    st.session_state.sector_cache = {}
# Cache for Kataly holdings to avoid repeated DB queries
if 'kataly_holdings' not in st.session_state:
    st.session_state.kataly_holdings = None

if 'kataly_holdings1' not in st.session_state:
    st.session_state.kataly_holdings1 = None
if 'stock_holdings' not in st.session_state:
    st.session_state.stock_holdings = None

# Cache for sector harm scores
if 'sector_harm_scores' not in st.session_state:
    st.session_state.sector_harm_scores = {}


if 'calculation_display' not in st.session_state:
    st.session_state.calculation_display = 'Bonds'  # or 'Stocks' as your default


# Database configuration
db_config = {
    'user': 'doadmin',
    'password': 'AVNS_xKVgSkiz4gkauzSux86',
    'host': 'db-mysql-nyc3-25707-do-user-19616823-0.l.db.ondigitalocean.com',
    'port': 25060,
    'database': 'defaultdb'
}

# Create SQLAlchemy connection string
db_connection_string = f"mysql+pymysql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"

# SEC API configuration
sec_api_key = "d4b2ad17695a7f448a38d2100a85dac3cfcee69d5590a45f39c8e9d8c9200053"

# Database connection pool - cached across session
@st.cache_resource
def get_db_engine():
    return create_engine(db_connection_string)

# Fetch Kataly holdings - Now with better caching strategy
def get_bond_info(cusip):
    """Fetch bond information from EODHD API"""
    API_TOKEN = "681bef9cbfd8f3.10724014"  # In production, use st.secrets or environment variables
    url = f'https://eodhd.com/api/bond-fundamentals/{cusip}?api_token={API_TOKEN}&fmt=json'
    
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise exception for 4XX/5XX responses
        data = response.json()
        
        # Extract relevant information
        bond_info = {
            'Name': data.get('Name', 'Unknown'),
            'Industry Group': data.get('ClassificationData', {}).get('IndustryGroup', 'Unknown'),
            'Issuer': data.get('IssueData', {}).get('Issuer', 'Unknown'),
            'Price': float(data.get('Price', 0)),
            'Coupon': float(data.get('Coupon', 0)),
            'Maturity Date': data.get('Maturity_Date', 'Unknown'),
            'YTM': float(data.get('YieldToMaturity', 0)),
        }
        return bond_info
    
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching bond data: {str(e)}")
        return None
    except (ValueError, KeyError) as e:
        st.error(f"Error processing bond data: {str(e)}")
        return None

def calculate_returns(row):
    """Calculate various return metrics for a bond"""
    # Market value calculation - use Current Price key (from the API response)
    market_value = row['Units'] * float(row['Current Price'] if row['Current Price'] not in [None, 'None', ''] else 0) / 100

    total_cost = row['Units'] * float(row['Purchase Price'] if row['Purchase Price'] not in [None, 'None', ''] else 0) / 100

    
    # Price return
    price_return = market_value - total_cost
    today = datetime.datetime.now().date()
    days_held = (today - row['Purchase Date']).days

    
    # Calculate accrued interest (income return)
    # Simple calculation: (coupon rate * par value * days held / 365)
    annual_interest = row['Units'] * (float(row['Coupon']) / 100)
    income_return = annual_interest * (days_held / 365)
    
    # Total return
    total_return = price_return + income_return
    
    return {
        'Market Value': market_value,
        'Total Cost': total_cost,
        'Price Return': price_return,
        'Income Return': income_return,
        'Total Return': total_return,
        
    }

def fetch_kataly_holdings():
    if st.session_state.kataly_holdings is not None:
        return st.session_state.kataly_holdings
    
    try:
        engine = get_db_engine()
        # Method 1: Use SQLAlchemy's text() function for safer SQL
        from sqlalchemy import text
        query = text("SELECT * FROM `Kataly-Holdings`")
        with engine.connect() as connection:
            df = pd.read_sql(query, connection)
        st.session_state.kataly_holdings = df
        return df
    except Exception as e:
        st.error(f"Error fetching Kataly holdings: {e}")
        return pd.DataFrame()

  # Cache for 1 hour
def fetch_sector_scoring_data():
    """Fetch sector scoring data from RHG-Sector-Scoring table"""
    try:
        engine = get_db_engine()
        from sqlalchemy import text
        query = text("SELECT * FROM `RHG-Sector-Scoring`")
        
        with engine.connect() as connection:
            df = pd.read_sql(query, connection)
        
        return df
    except Exception as e:
        st.error(f"Error fetching sector scoring data: {e}")
        return pd.DataFrame()

def add_scoring_columns_to_bonds(df, sector_scoring_df):
    """Add scoring columns to bond holdings dataframe"""
    if df.empty or sector_scoring_df.empty:
        return df
    
    # Create a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Initialize new columns
    df_copy['Sector Total Score'] = 0.0
    df_copy['Sector Mean Score'] = 0.0
    df_copy['Security Total Score'] = 0.0
    df_copy['Security Mean Score'] = 0.0
    
    # Create a mapping dictionary from sector scoring data
    sector_mapping = {}
    for _, row in sector_scoring_df.iterrows():
        sector = row.get('Sector', '')
        sector_mapping[sector] = {
            'total_score': float(row.get('Sector-Total-Score', 0)),
            'mean_score': float(row.get('Min-Max-Norm', 0))
        }
    
    # Apply scoring for each row
    for idx, row in df_copy.iterrows():
        sector = row.get('Sector', 'N/A')
        quantity = float(row.get('Quantity', 0)) if pd.notna(row.get('Quantity', 0)) else 0
        
        if sector in sector_mapping:
            sector_total = sector_mapping[sector]['total_score']
            sector_mean = sector_mapping[sector]['mean_score']
            
            df_copy.at[idx, 'Sector Total Score'] = sector_total
            df_copy.at[idx, 'Sector Mean Score'] = sector_mean
            df_copy.at[idx, 'Security Total Score'] = sector_total * quantity
            df_copy.at[idx, 'Security Mean Score'] = sector_mean * quantity
    
    return df_copy

from rapidfuzz import process, fuzz
import pandas as pd

def add_scoring_columns_to_bonds1(df, sector_scoring_df):
    """Add harm scoring columns to bond holdings dataframe using fuzzy-matched sector names."""
    
    if df.empty or sector_scoring_df.empty:
        return df
    
    # Create a copy of the input dataframe
    df_copy = df.copy()
    
    # Initialize scoring columns
    df_copy['Sector Total Score'] = 0.0
    df_copy['Sector Mean Score'] = 0.0
    df_copy['Security Total Score'] = 0.0
    df_copy['Security Mean Score'] = 0.0

    # Build a mapping of known sectors to their scores
    sector_mapping = {}
    for _, row in sector_scoring_df.iterrows():
        sector = row.get('Sector', '')
        if pd.notna(sector):
            sector_mapping[sector] = {
                'total_score': float(row.get('Sector-Total-Score', 0)),
                'mean_score': float(row.get('Min-Max-Norm', 0))
            }

    known_sectors = list(sector_mapping.keys())

    # Apply scoring based on fuzzy-matched sectors
    for idx, row in df_copy.iterrows():
        input_sector = row.get('Industry Group', 'N/A')
        quantity = float(row.get('Units', 0)) if pd.notna(row.get('Units', 0)) else 0

        if input_sector == 'Financial':
            input_sector = 'Financial Services'  # Handle missing sectors gracefully
        # Find the best fuzzy match (token sort helps ignore word order)
        best_match, match_score, _ = process.extractOne(input_sector, known_sectors, scorer=fuzz.token_sort_ratio)

        if match_score >= 80:  # Acceptable similarity threshold
            sector_total = sector_mapping[best_match]['total_score']
            sector_mean = sector_mapping[best_match]['mean_score']

            df_copy.at[idx, 'Sector Total Score'] = sector_total
            df_copy.at[idx, 'Sector Mean Score'] = sector_mean
            df_copy.at[idx, 'Security Total Score'] = sector_total * quantity
            df_copy.at[idx, 'Security Mean Score'] = sector_mean * quantity
        else:
            # If no good match found, leave default scores
            continue

    return df_copy


def add_scoring_columns_to_stocks(df, sector_scoring_df):
    """Add scoring columns to stock holdings dataframe"""
    if df.empty or sector_scoring_df.empty:
        return df
    
    # Create a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Initialize new columns
    df_copy['Sector Total Score'] = 0.0
    df_copy['Sector Mean Score'] = 0.0
    df_copy['Security Total Score'] = 0.0
    df_copy['Security Mean Score'] = 0.0
    
    # Create a mapping dictionary from sector scoring data
    sector_mapping = {}
    for _, row in sector_scoring_df.iterrows():
        sector = row.get('Sector', '')
        sector_mapping[sector] = {
            'total_score': float(row.get('Sector-Total-Score', 0)),
            'mean_score': float(row.get('Min-Max-Norm', 0))
        }
    
    # Apply scoring for each row
    for idx, row in df_copy.iterrows():
        sector = row.get('Sector', 'N/A')
        units = float(row.get('Units', 0)) if pd.notna(row.get('Units', 0)) else 0
        
        if sector in sector_mapping:
            sector_total = sector_mapping[sector]['total_score']
            sector_mean = sector_mapping[sector]['mean_score']
            
            df_copy.at[idx, 'Sector Total Score'] = sector_total
            df_copy.at[idx, 'Sector Mean Score'] = sector_mean
            df_copy.at[idx, 'Security Total Score'] = sector_total * units
            df_copy.at[idx, 'Security Mean Score'] = sector_mean * units
    
    return df_copy


# Bond info retrieval with caching
def get_bond_info(cusip):
    """Fetch bond information from EODHD API"""
    API_TOKEN = "681bef9cbfd8f3.10724014"  # In production, use st.secrets or environment variables
    url = f'https://eodhd.com/api/bond-fundamentals/{cusip}?api_token={API_TOKEN}&fmt=json'
    
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise exception for 4XX/5XX responses
        data = response.json()
        
        # Extract relevant information
        bond_info = {
            'Name': data.get('Name', 'Unknown'),
            'Industry Group': data.get('ClassificationData', {}).get('IndustryGroup', 'Unknown'),
            'Issuer': data.get('IssueData', {}).get('Issuer', 'Unknown'),
            'Current Price': float(data.get('Price') or '0'),
            'Coupon':  float(data.get('Coupon') or '0'),
            'Maturity Date': data.get('Maturity_Date', 'Unknown'),
            'YTM': float(data.get('YieldToMaturity') or '0'),
        }
        
        # Print for debugging
        print(f"API Response: {data}")
        print(f"Extracted Bond Info: {bond_info}")
        
        return bond_info
    
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching bond data: {str(e)}")
        return None
    except (ValueError, KeyError) as e:
        st.error(f"Error processing bond data: {str(e)}")
        return None

# Stock info with better caching
@lru_cache(maxsize=100)
def get_stock_info(ticker):
    try:
        # Check session cache first
        cache_key = f"stock_{ticker}"
        if cache_key in st.session_state.sector_cache:
            return st.session_state.sector_cache[cache_key]
            
        stock = Ticker(ticker)
        info = stock.info
        
        stock_info = {
            'Sector': info.get('sector', 'N/A'),
            'Industry': info.get('industry', 'N/A'),
            'Market Cap': info.get('marketCap', 'N/A'),
            'Open': info.get('open', 'N/A'),
            'High': info.get('high', 'N/A'),
            'Low': info.get('low', 'N/A'),
            'Beta': info.get('beta', 'N/A'),
            'Trailing PE': info.get('trailingPE', 'N/A'),
            '52 Week High': info.get('52WeekHigh', 'N/A')
        }
        
        # Cache the result
        st.session_state.sector_cache[cache_key] = stock_info
        return stock_info
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return {
            'Sector': 'N/A',
            'Industry': 'N/A',
            'Market Cap': 'N/A',
            'Open': 'N/A',
            'High': 'N/A',
            'Low': 'N/A',
            'Beta': 'N/A',
            'Trailing PE': 'N/A',
            '52 Week High': 'N/A'
        }

# More efficient sector retrieval
def get_sector(ticker, api_type='yahoo'):
    cache_key = f"{api_type}_{ticker}"
    
    # Check if we already have it cached
    if cache_key in st.session_state.sector_cache:
        return st.session_state.sector_cache[cache_key]
    
    try:
        if api_type == 'yahoo':
            stock = Ticker(ticker)
            sector = stock.info.get('sector', 'N/A')
        else:  # FINRA API
            
            time.sleep(0.1)  # Reduced delay but still prevent rate limiting
            url = f"https://api.finra.org/data/group/otcMarket/name/otcSymbolDirectory/securities/{ticker}"
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                sector = data.get('sector', 'N/A')
            else:
                sector = 'N/A'
                
        # Cache the result
        st.session_state.sector_cache[cache_key] = sector
        print(f"Fetched {api_type} sector for {ticker}: {sector}")
        return sector
    except Exception as e:
        print(f"Error fetching {api_type} sector for {ticker}: {e}")
        return 'N/A'


def map_kataly_holdings_to_sectors(df):
    if df.empty:
        return df
    
    # Ensure the security column exists
    security_col = "Security"
    if security_col not in df.columns:
        st.error(f"Column '{security_col}' not found in Kataly Holdings data")
        st.write("Available columns:", df.columns.tolist())
        return df
    
    # Create the Sector column if it doesn't exist
    if "Sector" not in df.columns:
        df["Sector"] = "N/A"
    
    # Show progress bar
    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()
    status_text.text("Preparing to process tickers...")
    
    # Define which rows should use which API based on requirements
    yahoo_rows = list(range(0, 10)) + list(range(15, 22)) + list(range(27, 30))
    finra_rows = list(range(10, 15)) + list(range(22, 27))
    
    # Process only if dataframe has enough rows
    max_row = max(max(yahoo_rows, default=0), max(finra_rows, default=0))
    if len(df) <= max_row:
        st.warning(f"Dataframe only has {len(df)} rows, but processing requires at least {max_row+1} rows")
    
    # Sequential processing (no threading)
    total_rows = len(yahoo_rows) + len(finra_rows)
    processed = 0
    
    # Process Yahoo Finance API rows
    for idx in yahoo_rows:
        if idx < len(df):
            security = str(df.iloc[idx][security_col])
            # Extract ticker symbol (first part before any space)
            ticker = security
        
            print(f"Processing {ticker} (row {idx+1}) with Yahoo Finance...")
            status_text.text(f"Processing {ticker} (row {idx+1}) with Yahoo Finance...")
            print(f"Ticker: {ticker}")
            ticker= company_tickers[ticker]
            try:
                sector = get_sector(ticker, 'yahoo')
                if sector != 'N/A':
                    df.at[idx, "Sector"] = sector
                    print(f"Processed {ticker} (row {idx+1}) with Yahoo: {sector}")
            except Exception as e:
                print(f"Error processing ticker {ticker} (row {idx+1}): {e}")
            
            processed += 1
            progress_bar.progress(processed / total_rows)
    
    # Process FINRA API rows
    for idx in finra_rows:
        if idx < len(df):
            security = str(df.iloc[idx][security_col])
            ticker = security
            
            status_text.text(f"Processing {ticker} (row {idx+1}) with FINRA...")
            
            try:
                sector = company_tickers[ticker]
                if sector != 'N/A':
                    df.at[idx, "Sector"] = sector
                    print(f"Processed {ticker} (row {idx+1}) with FINRA: {sector}")
            except Exception as e:
                print(f"Error processing ticker {ticker} (row {idx+1}): {e}")
            
            processed += 1
            progress_bar.progress(processed / total_rows)
    
    # Clean up progress indicators
    progress_bar.progress(100)
    time.sleep(0.5)  # Brief pause to show completion
    progress_bar.empty()
    status_text.empty()
    
    return df

import os
def read_disclaimer_file(filename):
    """Read and convert the Kataly-Disclaimer.docx file to HTML"""
    try:
        # Get the current directory and construct file path
        current_dir = os.getcwd()
        file_path = os.path.join(current_dir, filename)
        
        # Check if file exists
        if not os.path.exists(file_path):
            return "<p><em>Disclaimer file not found. Please ensure 'Kataly-Disclaimer.docx' is in the application directory.</em></p>"
        
        # Read the .docx file
        with open(file_path, 'rb') as docx_file:
            result = mammoth.convert_to_html(docx_file)
            html_content = result.value
            
            # Extract any conversion messages/warnings
            messages = result.messages
            if messages:
                print(f"Conversion messages: {messages}")
            
            return html_content
        
    except FileNotFoundError:
        return "<p><em>Disclaimer file not found. Please ensure 'Kataly-Disclaimer.docx' is in the application directory.</em></p>"
    except Exception as e:
        return f"<p><em>Error reading disclaimer file: {str(e)}</em></p>"

@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_sector_data(sector):
    try:
        engine = get_db_engine()
        
        # Method 1: Using SQLAlchemy text() with parameterized query (RECOMMENDED)
        from sqlalchemy import text
        query = text("""
            SELECT Sector, SDH_Category, SDH_Indicator, Harm_Description, 
                  Claim_Quantification, Harm_Typology, Total_Magnitude, Reach, 
                  Harm_Direction, Harm_Duration, Total_Score, `Key Citation 1`, `Key Citation 2`
            FROM rh_sankey 
            WHERE Sector = :sector
        """)
        
        # Pass parameters separately for SQL injection protection
        params = {"sector": sector}
        
        # Execute with parameters
        with engine.connect() as connection:
            df = pd.read_sql(query, connection, params=params)
        
        print(df)
        return df
    except Exception as e:
        st.error(f"Error fetching data for sector {sector}: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600) 
def fetch_sector_score_sankey(sector):
    try:
        engine = get_db_engine()
        from sqlalchemy import text
        
        query = text("""
            SELECT `Sector-Total-Score` FROM `RHG-Sector-Scoring`
            WHERE Sector = :sector
        """)
        
        params = {"sector": sector}
        
        with engine.connect() as connection:
            df = pd.read_sql(query, connection, params=params)
        
        if not df.empty:
            # Return the single value
            return df.iloc[0, 0]  # First row, first column value
        else:
            st.warning(f"No data found for sector: {sector}")
            return None

    except Exception as e:
        st.error(f"Error fetching data for sector {sector}: {e}")
        return None

@st.cache_data(ttl=3600) 
def fetch_sector_score_sankey_minmax(sector):
    try:
        engine = get_db_engine()
        from sqlalchemy import text
        
        query = text("""
            SELECT `Min-Max-Norm` FROM `RHG-Sector-Scoring`
            WHERE Sector = :sector
        """)
        
        params = {"sector": sector}
        
        with engine.connect() as connection:
            df = pd.read_sql(query, connection, params=params)
        
        if not df.empty:
            # Return the single value
            return df.iloc[0, 0]  # First row, first column value
        else:
            st.warning(f"No data found for sector: {sector}")
            return None

    except Exception as e:
        st.error(f"Error fetching data for sector {sector}: {e}")
        return None

# Function to calculate sector Min-Max-Norm value
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_sector_min_max_norm(sector):
    # Check if we already have it in cache
    if sector in st.session_state.sector_harm_scores:
        return st.session_state.sector_harm_scores[sector]
    
    # If not, fetch it from the database
    try:
        df = fetch_sector_data(sector)
        if not df.empty:
            # Calculate the Min-Max-Norm value (using average of Total_Score for the sector)
            min_max_norm = df['Total_Score'].mean()
            
            # Cache the result
            st.session_state.sector_harm_scores[sector] = min_max_norm
            return min_max_norm
        else:
            # Default value if no data found
            return 0.0
    except Exception as e:
        print(f"Error calculating Min-Max-Norm for sector {sector}: {e}")
        return 0.0


# Function to prepare Sankey data with proper flow conservation
def prepare_sankey_data(df, sector):
    harm_typologies = df['Harm_Typology'].unique().tolist()
    sdh_categories = df['SDH_Category'].unique().tolist()
    sdh_indicators = df['SDH_Indicator'].unique().tolist()

    node_dict = {}
    node_list = []

    # Build node dictionary and list
    node_dict[sector] = len(node_list)
    node_list.append(sector)

    for harm_typology in harm_typologies:
        node_dict[harm_typology] = len(node_list)
        node_list.append(harm_typology)

    for sdh_category in sdh_categories:
        if sdh_category not in node_dict:
            node_dict[sdh_category] = len(node_list)
            node_list.append(sdh_category)

    for sdh_indicator in sdh_indicators:
        if sdh_indicator not in node_dict:
            node_dict[sdh_indicator] = len(node_list)
            node_list.append(sdh_indicator)

    source = []
    target = []
    value = []
    
    # Track links to avoid duplicates
    link_aggregation = {}
    
    # Process each row in the dataframe
    for _, row in df.iterrows():
        harm_typology = row['Harm_Typology']
        sdh_category = row['SDH_Category']
        sdh_indicator = row['SDH_Indicator']
        raw_score = float(row['Total_Score'])
        
        # Get node indices
        sector_index = node_dict[sector]
        harm_typology_index = node_dict[harm_typology]
        sdh_category_index = node_dict[sdh_category]
        sdh_indicator_index = node_dict[sdh_indicator]
        
        # Link 1: Sector -> Harm Typology
        link1_key = (sector_index, harm_typology_index)
        if link1_key not in link_aggregation:
            link_aggregation[link1_key] = 0
        link_aggregation[link1_key] += raw_score
        
        # Link 2: Harm Typology -> SDH Category
        link2_key = (harm_typology_index, sdh_category_index)
        if link2_key not in link_aggregation:
            link_aggregation[link2_key] = 0
        link_aggregation[link2_key] += raw_score
        
        # Link 3: SDH Category -> SDH Indicator
        link3_key = (sdh_category_index, sdh_indicator_index)
        if link3_key not in link_aggregation:
            link_aggregation[link3_key] = 0
        link_aggregation[link3_key] += raw_score

    # Convert aggregated links to lists
    for (src, tgt), val in link_aggregation.items():
        source.append(src)
        target.append(tgt)
        value.append(val)

    return node_list, source, target, value

# Updated function to style Sankey nodes with consistent level colors
def style_sankey_nodes(node_list, sector, df):
    # Define colors for each level
    level_colors = {
        'sector': '#1f77b4',      # Blue
        'harm_typology': '#ff7f0e', # Orange  
        'sdh_category': '#2ca02c',  # Green
        'sdh_indicator': '#d62728'  # Red
    }
    
    node_colors = []
    
    # Get unique items for each level from the dataframe
    harm_typologies = df['Harm_Typology'].unique().tolist()
    sdh_categories = df['SDH_Category'].unique().tolist()
    sdh_indicators = df['SDH_Indicator'].unique().tolist()
    
    # Assign colors based on node type
    for node in node_list:
        if node == sector:
            node_colors.append(level_colors['sector'])
        elif node in harm_typologies:
            node_colors.append(level_colors['harm_typology'])
        elif node in sdh_categories:
            node_colors.append(level_colors['sdh_category'])
        elif node in sdh_indicators:
            node_colors.append(level_colors['sdh_indicator'])
        else:
            # Fallback color (shouldn't happen with proper data)
            node_colors.append('#999999')
    
    return node_colors, level_colors

# Create legend for Sankey diagram
def create_sankey_legend(level_colors):
    import plotly.graph_objects as go
    
    # Create a legend using invisible scatter traces
    fig_legend = go.Figure()
    
    legend_labels = {
        'sector': 'Economic Sector (GICS)',
        'harm_typology': 'Desperate Impact Category',
        'sdh_category': 'SDH Category', 
        'sdh_indicator': 'SDH Indicator'
    }
    
    for key, color in level_colors.items():
        fig_legend.add_trace(go.Scatter(
            x=[None],  # No visible point
            y=[None],
            mode='markers',
            marker=dict(size=15, color=color),
            name=legend_labels.get(key, key),  # fallback to key name
            showlegend=True
        ))
    
    fig_legend.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=100,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return fig_legend

# Function to create Sankey diagram with proper hover templates
def create_sankey_diagram(node_list, source, target, value, node_colors):
    import plotly.graph_objects as go
    
    # Create hover text for nodes showing total inflow/outflow
    node_hover_text = []
    for i, node in enumerate(node_list):
        # Calculate total inflow
        inflow = sum(value[j] for j in range(len(source)) if target[j] == i)
        # Calculate total outflow  
        outflow = sum(value[j] for j in range(len(source)) if source[j] == i)
        
        if inflow > 0 and outflow > 0:
            hover_text = f"{node}<br>Inflow: {inflow:.2f}<br>Outflow: {outflow:.2f}"
        elif inflow > 0:
            hover_text = f"{node}<br>Total: {inflow:.2f}"
        elif outflow > 0:
            hover_text = f"{node}<br>Total: {outflow:.2f}"
        else:
            hover_text = f"{node}<br>Total: 0.00"
            
        node_hover_text.append(hover_text)
    
    # Create hover text for links
    link_hover_text = []
    for i in range(len(source)):
        source_node = node_list[source[i]]
        target_node = node_list[target[i]]
        link_value = value[i]
        hover_text = f"{source_node} → {target_node}<br>Flow: {link_value:.2f}"
        link_hover_text.append(hover_text)
    
    # Create the Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=12,
            line=dict(color="black", width=0.5),
            label=node_list,
            color=node_colors,
            hovertemplate='%{customdata}<extra></extra>',
            customdata=node_hover_text
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
            hovertemplate='%{customdata}<extra></extra>',
            customdata=link_hover_text
        )
    )])
    
    # Update text formatting
    fig.update_traces(
        selector=dict(type='sankey'),
        textfont=dict(color='black', size=14)
    )
    
    return fig


# Function to calculate portfolio harm scores using existing sector and security mean scores
def calculate_portfolio_harm_scores(kataly_holdings=None):
    
    # if stock_holdings is not None and not stock_holdings.empty:
    #     print("Stock Holdings Columns:", stock_holdings.columns.tolist())
    
    sector_mean_scores = []
    security_mean_scores = []
    total_units = 0
    print(kataly_holdings)
    # Process bond holdings (Kataly holdings)
    if kataly_holdings is not None and not kataly_holdings.empty:
        for _, row in kataly_holdings.iterrows():
                sector_score = row.get('Sector Mean Score', 0)
                if pd.isna(sector_score):
                    sector_score = 0
                try:
                    sector_score = float(sector_score)
                except (ValueError, TypeError):
                    sector_score = 0
                
                # Get security mean score  
                raw_score = row.get('Security Mean Score', 0)
                
                
                try:
                    
                    security_score = float(str(raw_score).replace(',', ''))
                except (ValueError, TypeError):
                    
                    security_score = 0
                
                # Add to lists for averaging
                sector_mean_scores.append(sector_score)
                security_mean_scores.append(security_score)
                
              
    
    # If no valid data found, return default values
    if len(sector_mean_scores) == 0 or len(security_mean_scores) == 0:
        return {
            'average_score': 0.0,
            'total_score': 0.0,
            'quartile': "N/A"
        }
    
    # Calculate average scores as specified
    average_score = sum(sector_mean_scores) / len(sector_mean_scores)  # Average of sector mean scores
    total_score = sum(security_mean_scores) / len(security_mean_scores)  # Average of security mean scores
    
    # Determine quartile (keeping the same quartile boundaries)
    quartile = "N/A"
    if average_score <= 38.80:
        quartile = f"{average_score:.2f} - Quartile 1"
    elif 38.80 < average_score <= 50.00:
        quartile = f"{average_score:.2f} - Quartile 2"
    elif 50.00 < average_score <= 82.40:
        quartile = f"{average_score:.2f} - Quartile 3"
    elif average_score > 82.40:
        quartile = f"{average_score:.2f} - Quartile 4"

    return {
        'average_score': average_score,
        'total_score': total_score,
        'quartile': quartile
    }


def calculate_portfolio_harm_scores_stocks(stock_holdings):
    
    if stock_holdings is not None and not stock_holdings.empty:
        print("Stock Holdings Columns:", stock_holdings.columns.tolist())
    
    sector_mean_scores = []
    security_mean_scores = []
    print(stock_holdings)
   
                  
    #Process stock holdings (if provided and has similar columns)
    if stock_holdings is not None and not stock_holdings.empty:
        for _, row in stock_holdings.iterrows():
    
                # Get sector mean score (if exists in stock holdings)
                sector_score = row.get('Sector Mean Score', 0)
                if pd.isna(sector_score):
                    sector_score = 0
                try:
                    sector_score = float(sector_score)
                except (ValueError, TypeError):
                    sector_score = 0
                
                # Get security mean score (if exists in stock holdings)
                security_score = row.get('Security Mean Score', 0)
                print(f"Security Score: {security_score}")
                if pd.isna(security_score):
                    security_score = 0
                try:
                    security_score = float(str(security_score).replace(',', ''))
                except (ValueError, TypeError):
                    security_score = 0
                
                # Add to lists for averaging
                sector_mean_scores.append(sector_score)
                security_mean_scores.append(security_score)
                print(f"Sector Score: {sector_score}, Security Score: {security_score}")
    
    # If no valid data found, return default values
    if len(sector_mean_scores) == 0 or len(security_mean_scores) == 0:
        return {
            'average_score': 0.0,
            'total_score': 0.0,
            'quartile': "N/A"
        }
    
    # Calculate average scores as specified
    average_score = sum(sector_mean_scores) / len(sector_mean_scores)  # Average of sector mean scores
    total_score = sum(security_mean_scores) / len(security_mean_scores)  # Average of security mean scores
    
    # Determine quartile (keeping the same quartile boundaries)
    quartile = "N/A"
    if average_score <= 38.80:
        quartile = f"{average_score:.2f} - Quartile 1"
    elif 38.80 < average_score <= 50.00:
        quartile = f"{average_score:.2f} - Quartile 2"
    elif 50.00 < average_score <= 82.40:
        quartile = f"{average_score:.2f} - Quartile 3"
    elif average_score > 82.40:
        quartile = f"{average_score:.2f} - Quartile 4"
    print(f"Average Score: {average_score}, Total Score: {total_score}, Quartile: {quartile}")
    return {
        'average_score': average_score,
        'total_score': total_score,
        'quartile': quartile
    }

def get_combined_portfolio_harm_scores():
    """
    Convenience function to get portfolio harm scores using bonds, manually added bonds, and stocks from session state
    """
    # Get Kataly bond holdings from session state
    kataly_holdings = st.session_state.get('kataly_holdings1', pd.DataFrame())
    
    # Get manually added bond holdings from session state
    manual_bonds = st.session_state.get('df_bonds_scores', pd.DataFrame())
    
    # Combine all bond holdings (Kataly + manually added)
    all_bond_holdings = pd.DataFrame()
    
    # Add Kataly holdings if available
    if kataly_holdings is not None and not kataly_holdings.empty:
        all_bond_holdings = kataly_holdings.copy()
        print("Kataly Holdings Columns:", kataly_holdings.columns.tolist())
    
    if manual_bonds is not None and not manual_bonds.empty:
        print("Manual Holdings Columns:", manual_bonds.columns.tolist())

    # Add manually added bonds if available
    if manual_bonds is not None and not manual_bonds.empty:
        # Ensure manual bonds have the required columns and format
        manual_bonds_formatted = manual_bonds.copy()
        
        # Map column names if needed (manual bonds might have different column structure)
        manual_bonds_formatted['Quantity'] = manual_bonds_formatted['Units']
        manual_bonds_formatted['Sector'] = manual_bonds_formatted['Industry Group']
        
        # Combine with Kataly holdings
        if all_bond_holdings.empty:
            all_bond_holdings = manual_bonds_formatted
        else:
            # Ensure both dataframes have the same columns before concatenating
            common_columns = list(set(all_bond_holdings.columns) & set(manual_bonds_formatted.columns))
            if common_columns:
                all_bond_holdings = pd.concat([
                    all_bond_holdings[common_columns], 
                    manual_bonds_formatted[common_columns]
                ], ignore_index=True)
    
    return calculate_portfolio_harm_scores(all_bond_holdings)

def update_portfolio_allocation(df):
    if not df.empty:
        total_value = df['Current Value ($)'].astype(float).sum()
        df.loc[:, 'Portfolio Allocation'] = df['Current Value ($)'].astype(float) / total_value * 100
        df.loc[:, 'Portfolio Allocation'] = df['Portfolio Allocation'].apply(lambda x: f"{x:.2f}%")
    return df

def get_gics_sector(ticker):
    try:
        stock = yf.Ticker(ticker)
        return stock.info.get('sector', 'N/A')
    except:
        return 'N/A'

# Add caching status indicator to sidebar
# Add caching status indicator to sidebar
def show_sidebar():
    with st.sidebar:
        st.image("Kataly-Featured-Logo.png")
        st.header("Add Stock to Portfolio")
        ticker = st.text_input("Enter a stock ticker", placeholder="AAPL", key="stock_ticker_input")
        units_stock = st.number_input("Enter number of units", min_value=1, step=1)
        transaction_date = st.date_input("Select transaction date")
        add_button = st.button("Add Stock", key="add_stock_button")

        # Bond input section
        st.header("Add Bond by CUSIP")
        cusip = st.text_input("Enter 9-character CUSIP", placeholder="910047AG4", key="cusip_input")
        units = st.number_input("Units (Face Value)", min_value=1000, step=1000, value=10000)
        purchase_price = st.number_input("Purchase Price (% of par)", min_value=1.0, max_value=200.0, value=100.0, step=0.01)
        purchase_date = st.date_input("Purchase Date", value=datetime.datetime.now().date() - relativedelta(months=1))
        
        add_bond_button = st.button("Add Bond", key="add_bond_button")

        # Placeholder for Download section
        download_section_placeholder = st.empty()

        # Handle add stock button
        if add_button and ticker:
            with st.spinner(f"Fetching data for {ticker}..."):
                if units_stock == 0:
                    st.error("Units cannot be zero. Please enter a valid number of units.")
                else:
                    
                        try:
                            stock = yf.Ticker(ticker)
                            hist = stock.history(start=transaction_date)

                            if hist.empty:
                                st.error("No data available for the selected date. Please choose a valid trading day.")
                            else:
                                purchase_price_stock = hist.iloc[0]['Close']
                                current_price_stock = stock.info['currentPrice']

                                initial_investment_stock = purchase_price_stock * units_stock
                                current_value = current_price_stock * units_stock
                                gain_loss = current_value - initial_investment_stock
                                gain_loss_percentage = (gain_loss / initial_investment_stock) * 100

                                gics_sector = get_gics_sector(ticker)
            
                                
                            
                            
                                new_row = pd.DataFrame({
                                    'Stock': [ticker],
                                    'Units': [units_stock],
                                    'Purchase Date': [transaction_date],
                                    'Purchase Price ($)': [f"{purchase_price_stock:.2f}"],
                                    'Current Price ($)': [f"{current_price_stock:.2f}"],
                                    'Initial Investment ($)': [f"{initial_investment_stock:.2f}"],
                                    'Current Value ($)': [f"{current_value:.2f}"],
                                    'Gain/Loss ($)': [f"{gain_loss:.2f}"],
                                    'Gain/Loss (%)': [gain_loss_percentage],
                                    'Portfolio Allocation': ["0.00%"],
                                    'Sector': [gics_sector],
                                })
                                
                                # After adding the new row, recalculate harm score contributions for all stocks
                                st.session_state.df_stock_info = pd.concat([st.session_state.df_stock_info, new_row], ignore_index=True)
                                st.session_state.df_stock_info = update_portfolio_allocation(st.session_state.df_stock_info)
                                st.success(f"Added {ticker} to your portfolio.")

                        except Exception as e:
                            st.error(f"An error occurred: {str(e)}")
        # Handle add bond button
        if add_bond_button and cusip:
            if len(cusip) != 9:
                st.error("CUSIP must be 9 characters")
            else:
                with st.spinner(f"Fetching data for CUSIP {cusip}..."):
                    if cusip in st.session_state.df_bonds['CUSIP'].values:
                        st.warning(f"CUSIP {cusip} is already in your bond holdings.")
                    else:
                        bond_info = get_bond_info(cusip)
                        print(f"Bond info for {cusip}: {bond_info}")
                        if bond_info:
                            bond_info['CUSIP'] = cusip
                            bond_info['Units'] = units
                            bond_info['Purchase Price'] = purchase_price
                            bond_info['Purchase Date'] = purchase_date
                            # Calculate returns
                            returns = calculate_returns(bond_info)
                            bond_info.update(returns)
                            
                            new_bond = pd.DataFrame([bond_info])
                            st.session_state.df_bonds = pd.concat(
                                [st.session_state.df_bonds, new_bond],
                                ignore_index=True
                            )
                            st.success(f"Added CUSIP {cusip} to bond holdings")

        # Fill the bottom Download section in the placeholder
        with download_section_placeholder.container():
           

            st.header("Download Report")
            selected_sector = 'None'
            profile_df = None

            # Get unique sectors
            stock_sectors = st.session_state.df_stock_info['Sector'].unique().tolist()
            kataly_holdings = st.session_state.kataly_holdings
            kataly_sectors = []
            if kataly_holdings is not None and not kataly_holdings.empty and 'Sector' in kataly_holdings.columns:
                kataly_sectors = kataly_holdings['Sector'].unique().tolist()

            all_sectors = list(set(stock_sectors + kataly_sectors))
            all_sectors = [sector for sector in all_sectors if sector != 'N/A' and pd.notna(sector)]

            if all_sectors:
                if 'selected_sector' in st.session_state:
                    selected_sector = st.session_state.selected_sector
                else:
                    selected_sector = all_sectors[0]

                df = fetch_sector_data(selected_sector)
                if not df.empty:
                    profile_df = df[['SDH_Indicator', 'SDH_Category', 'Harm_Description', 'Harm_Typology',
                                     'Total_Magnitude', 'Reach', 'Harm_Direction', 'Harm_Duration', 'Total_Score',"Key Citation 1","Key Citation 2"]]
                    profile_df = profile_df.rename(columns={
                        'SDH_Indicator': 'SDH Indicator',
                        'SDH_Category': 'SDH Category',
                        'Harm_Description': 'Equity Description',
                        'Harm_Typology': 'Equity Typology',
                        'Total_Magnitude': 'Total Magnitude',
                        'Harm_Direction': 'Equity Direction',
                        'Harm_Duration': 'Equity Duration',
                        'Total_Score': 'Total Score',
            
                    })

                    portfolio_harm_scores = get_combined_portfolio_harm_scores()

                    if selected_sector and profile_df is not None:
                        pdf_buffer = generate_report(selected_sector, profile_df, portfolio_harm_scores)
                        st.download_button(
                            label="Download PDF Report",
                            data=pdf_buffer,
                            file_name=f"racial_Equity_report_{selected_sector}.pdf",
                            mime="application/pdf",
                            key="pdf_download"
                        )


def main():
    # Display sidebar
    show_sidebar()
    
    st.title("Corporate Racial Justice Intelligence Canvas")
    
    # Portfolio Holdings Summary
    st.text("")
    st.subheader("Portfolio Holdings Summary", divider="blue")
    st.markdown(" ")
    
    # Fetch sector scoring data
    sector_scoring_df = fetch_sector_scoring_data()
    
    # Only fetch Kataly holdings once per session
    with st.spinner("Loading Kataly holdings data..."):
        kataly_holdings = fetch_kataly_holdings()
    
    # Process Kataly holdings to map sectors - only if needed
    if not kataly_holdings.empty:
        if "Sector" not in kataly_holdings.columns or kataly_holdings["Sector"].eq("N/A").all():
            with st.spinner("Mapping sectors for holdings..."):
                kataly_holdings = map_kataly_holdings_to_sectors(kataly_holdings)
                # Update the cached version
                st.session_state.kataly_holdings = kataly_holdings
    
    # Create tabs for stocks and Kataly bonds
    tab1, tab2 = st.tabs(["Kataly Bond Portfolio", "Stocks"])
    
    with tab1:
        
        if not kataly_holdings.empty:
            # Add scoring columns to kataly holdings
            kataly_with_scores = add_scoring_columns_to_bonds(kataly_holdings, sector_scoring_df)
            
            # Create a copy of the dataframe for formatting
            formatted_kataly = kataly_with_scores.copy()

            # Apply specific formatting based on column name
            for col in formatted_kataly.columns:
                if col == 'Quantity':
                    # Add commas
                    formatted_kataly[col] = formatted_kataly[col].apply(
                        lambda x: f"{int(x):,}" if pd.notna(x) else ""
                    )
                elif col == 'Unit_Cost':
                    # Add "$"
                    formatted_kataly[col] = formatted_kataly[col].apply(
                        lambda x: f"${float(x):.2f}" if pd.notna(x) else ""
                    )
                elif col == 'Total_Cost':
                    # Add commas and "$"
                    formatted_kataly[col] = formatted_kataly[col].apply(
                        lambda x: f"${int(round(float(x))):,}" if pd.notna(x) else ""
                    )
                elif col == 'Units':
                    # Add commas
                    formatted_kataly[col] = formatted_kataly[col].apply(
                        lambda x: f"{int(round(float(x))):,}" if pd.notna(x) else ""
                    )
                elif col == 'Purchase':
                    # Add "$"
                    formatted_kataly[col] = formatted_kataly[col].apply(
                        lambda x: f"${float(x):.4f}" if pd.notna(x) else ""
                    )
                elif col == 'Market_Value':
                    # Add "$", commas, remove decimal, round up
                    formatted_kataly[col] = formatted_kataly[col].apply(
                        lambda x: f"${int(round(float(x))):,}" if pd.notna(x) else ""
                    )
                elif col == 'Accrued':
                    # Add "$", commas, remove decimal, round up
                    formatted_kataly[col] = formatted_kataly[col].apply(
                        lambda x: f"${int(round(float(x))):,}" if pd.notna(x) else ""
                    )
                elif col == 'Original':
                    # Add commas and "$"
                    formatted_kataly[col] = formatted_kataly[col].apply(
                        lambda x: f"${int(round(float(x))):,}" if pd.notna(x) else ""
                    )
                elif col == 'Percent_of_Assets':
                    # Add "%", 2 decimal points
                    formatted_kataly[col] = formatted_kataly[col].apply(
                        lambda x: f"{float(x):.2f}%" if pd.notna(x) else ""
                    )
                elif col == 'Coupon':
                    # Add "$" and 2 decimal points
                    formatted_kataly[col] = formatted_kataly[col].apply(
                        lambda x: f"${float(x):.2f}" if pd.notna(x) else ""
                    )
                elif col in ['Sector Total Score', 'Sector Mean Score']:
                    # Format scoring columns with 2 decimal places
                    formatted_kataly[col] = formatted_kataly[col].apply(
                        lambda x: f"{float(x):.2f}" if pd.notna(x) else "0.00"
                    )
                elif col in ['Security Total Score', 'Security Mean Score']:
                    # Format security scoring columns with commas and 2 decimal places
                    formatted_kataly[col] = formatted_kataly[col].apply(
                        lambda x: f"{float(x):,.2f}" if pd.notna(x) else "0.00"
                    )
            st.session_state.kataly_holdings1 = formatted_kataly
            # Display the formatted Kataly bond holdings
            st.dataframe(formatted_kataly, use_container_width=True)
        
            if not st.session_state.df_bonds.empty:
                # Display bond holdings

                st.session_state.df_bonds_scores = add_scoring_columns_to_bonds1( st.session_state.df_bonds[['CUSIP', 'Industry Group', 'Issuer', 'Units','Current Price' ,'Purchase Price','Coupon','Price Return','Income Return', 'Total Return']], sector_scoring_df)
                st.text("Bond Holdings")
                st.dataframe(
                    st.session_state.df_bonds_scores[['CUSIP', 'Industry Group', 'Issuer', 'Units','Current Price' ,'Purchase Price','Coupon','Price Return','Income Return', 'Total Return']],
                    use_container_width=True
                )
    
    with tab2:
       
        if not st.session_state.df_stock_info.empty:
            # Add scoring columns to stock holdings
            stocks_with_scores = add_scoring_columns_to_stocks(st.session_state.df_stock_info, sector_scoring_df)
            
            # Format the scoring columns for display
            formatted_stocks = stocks_with_scores.copy()
            
            # Format the new scoring columns
            for col in ['Sector Total Score', 'Sector Mean Score']:
                formatted_stocks[col] = formatted_stocks[col].apply(
                    lambda x: f"{float(x):.2f}" if pd.notna(x) else "0.00"
                )
            
            for col in ['Security Total Score', 'Security Mean Score']:
                formatted_stocks[col] = formatted_stocks[col].apply(
                    lambda x: f"{float(x):,.2f}" if pd.notna(x) else "0.00"
                )
            st.session_state.stock_holdings = formatted_stocks
            # Display the stock info with sector mapping and scoring
            st.dataframe(formatted_stocks, use_container_width=True)
        
    
    
    
    
        

    # Add space 
    st.markdown(" ") 
    
    # NEW SECTION: Portfolio Racial Harm Summary 
    st.markdown("<h3 style='color: #333; border-bottom: 2px solid #6082B6;'>Portfolio Racial Equity Summary</h3>", unsafe_allow_html=True) 
    st.markdown(" ") 
    # Calculate portfolio harm scores
    
    portfolio_harm_scores_bonds = get_combined_portfolio_harm_scores()
    portfolio_harm_scores_stocks = calculate_portfolio_harm_scores_stocks(st.session_state.stock_holdings)
   
    st.markdown("""
    <style>
        .metric-box {
            border: 2px dashed #999;
            border-radius: 12px;
            padding: 10px;
            text-align: center;
            margin: 5px;
        }
        .metric-title {
            font-size: 14px;
            color: #666;
            margin-bottom: 10px;
            text-align: center; 
        }
        .metric-value {
            font-size: 34px;
            font-weight: bold;
            margin: 10px 0;
        }
        .metric-container {
            display: flex;
            justify-content: space-between;
        }
        
        .tooltip {
            position: relative;
            display: inline-block;
        }

        .tooltip .tooltiptext {
            visibility: hidden;
            width: 250px;
            background-color: #333;
            color: #fff;
            text-align: left;
            border-radius: 6px;
            padding: 10px;
            position: absolute;
            z-index: 1000;
            bottom: 125%;
            left: 50%;
            margin-left: -125px;
            opacity: 0;
            transition: opacity 0.3s;
            font-size: 12px;
            line-height: 1.4;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        }

        .tooltip .tooltiptext::after {
            content: "";
            position: absolute;
            top: 100%;
            left: 50%;
            margin-left: -5px;
            border-width: 5px;
            border-style: solid;
            border-color: #333 transparent transparent transparent;
        }

        .tooltip:hover .tooltiptext {
            visibility: visible;
            opacity: 1;
        }

        .info-icon {
            display: inline-block;
            width: 16px;
            height: 16px;
            background-color: #666;
            color: white;
            border-radius: 50%;
            text-align: center;
            line-height: 16px;
            font-size: 10px;
            margin-left: 5px;
            cursor: help;
        }
    </style>
    """, unsafe_allow_html=True)

    # Create the container for the metrics
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)

    # Create each box with HTML/CSS
    harm_tab1, harm_tab2 = st.tabs(["Bond Portfolio Harm Scores", "Stock Portfolio Harm Scores"])

    with harm_tab1:
        
            # Create the container for the metrics
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)

            # Create each box with HTML/CSS
        col1, col2, col3 = st.columns(3)

        with col1:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-title">
                        <div class="tooltip">
                            Average Portfolio Harm Score
                            <span class="info-icon">?</span>
                            <span class="tooltiptext">This score represents a portfolio level weighted average based on the number of units for each security. It provides a normalized view of harm across your entire bond portfolio holdings.</span>
                        </div>
                    </div>
                    <div class="metric-value">{portfolio_harm_scores_bonds['average_score']:.1f}</div>
                </div>
                """, unsafe_allow_html=True)

        with col2:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-title">
                        <div class="tooltip">
                            Total Portfolio Harm Score
                            <span class="info-icon">?</span>
                            <span class="tooltiptext">The total score represented by all bond holdings. This value is most useful when comparing against other portfolios or portfolio compositions of different sizes.</span>
                        </div>
                    </div>
                    <div class="metric-value">{int(portfolio_harm_scores_bonds['total_score']):,}</div>
                </div>
                """, unsafe_allow_html=True)

        with col3:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-title">
                        <div class="tooltip">
                            Total Portfolio Harm Quartile
                            <span class="info-icon">?</span>
                            <span class="tooltiptext">Where the average bond portfolio sits relative to other potential portfolio compositions using Min/Max data standardization. A portfolio exclusively comprised of the lowest harm security would score "0".</span>
                        </div>
                    </div>
                    <div class="metric-value">{portfolio_harm_scores_bonds['quartile']}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)
            
        

    with harm_tab2:
        
        
        
            # Create the container for the metrics
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)

            # Create each box with HTML/CSS
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-title">
                        <div class="tooltip">
                            Average Portfolio Harm Score
                            <span class="info-icon">?</span>
                            <span class="tooltiptext">This score represents a portfolio level weighted average based on the number of shares for each security. It provides a normalized view of harm across your entire stock portfolio holdings.</span>
                        </div>
                    </div>
                    <div class="metric-value">{portfolio_harm_scores_stocks['average_score']:.1f}</div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-title">
                        <div class="tooltip">
                            Total Portfolio Harm Score
                            <span class="info-icon">?</span>
                            <span class="tooltiptext">The total score represented by all stock holdings. This value is most useful when comparing against other portfolios or portfolio compositions of different sizes.</span>
                        </div>
                    </div>
                    <div class="metric-value">{int(portfolio_harm_scores_stocks['total_score']):,}</div>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-title">
                        <div class="tooltip">
                            Total Portfolio Harm Quartile
                            <span class="info-icon">?</span>
                            <span class="tooltiptext">Where the average stock portfolio sits relative to other potential portfolio compositions using Min/Max data standardization. A portfolio exclusively comprised of the lowest harm security would score "0".</span>
                        </div>
                    </div>
                    <div class="metric-value">{portfolio_harm_scores_stocks['quartile']}</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)
    
    # Add space
    st.markdown(" ")
    
    # Corporate Racial Harm Canvas section
    st.subheader(f"Corporate Racial Equity Canvas", divider="blue")
    st.markdown(" ")
    # Collect available sectors from both stock info and Kataly holdings
    stock_sectors = st.session_state.df_stock_info['Sector'].unique().tolist()
    kataly_sectors = []
    if not kataly_holdings.empty and 'Sector' in kataly_holdings.columns:
        kataly_sectors = kataly_holdings['Sector'].unique().tolist()

    # Combine sectors and filter out invalid values
    all_sectors = list(set(stock_sectors + kataly_sectors))
    all_sectors = [sector for sector in all_sectors if sector != 'N/A' and pd.notna(sector)]

    if all_sectors:
        selected_sector = st.selectbox("Select a sector for the Sankey diagram",['']+ all_sectors)
        
        # Replace this section in your Streamlit code:

        if selected_sector:
            # Fetch data for the selected sector
            with st.spinner(f"Loading data for {selected_sector} sector..."):
                df = fetch_sector_data(selected_sector)
            
            # Prepare data for the Sankey diagram
            if not df.empty:
                st.info(f"Corporate Racial Equity Canvas for {selected_sector} Sector")
                node_list, source, target, value = prepare_sankey_data(df, selected_sector)
                node_colors, level_colors = style_sankey_nodes(node_list, selected_sector, df)
                
                # Create and display the legend
                legend_fig = create_sankey_legend(level_colors)
                st.plotly_chart(legend_fig, use_container_width=True)
                
                # Create the Sankey diagram with proper hover tooltips
                fig = create_sankey_diagram(node_list, source, target, value, node_colors)
                st.plotly_chart(fig, use_container_width=True)
                
                # Rest of your code for total_score, mean_score, etc.
                total_score = fetch_sector_score_sankey(selected_sector)
                mean_score = fetch_sector_score_sankey_minmax(selected_sector)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"""
                    <div class="metric-box">
                        <div class="metric-title">
                            <div class="tooltip">
                                Total Sector Score
                            </div>
                        </div>
                        <div class="metric-value">{total_score:.2f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-box">
                        <div class="metric-title">
                            <div class="tooltip">
                                Sector Mean Score
                            </div>
                        </div>
                        <div class="metric-value">{mean_score:.2f}</div>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Continue with the rest of your existing code...
                
                st.markdown(" ")

                st.subheader(f"Detailed {selected_sector} Sector Equity Profile", divider="blue")
                st.markdown(" ")
                
                # Create a DataFrame with required columns
                profile_df = df[['SDH_Indicator', 'SDH_Category', 'Harm_Description', 'Harm_Typology', 
                            'Total_Magnitude', 'Reach', 'Harm_Direction', 'Harm_Duration',"Total_Score","Key Citation 1","Key Citation 2"]]
                
                # Rename columns for display
                profile_df = profile_df.rename(columns={
                    'SDH_Indicator': 'SDH Indicator',
                    'SDH_Category': 'SDH Category', 
                    'Harm_Typology': 'Equity Typology',
                    "Harm_Description": 'Equity Description',
                    'Total_Magnitude': 'Total Magnitude',
                    'Harm_Direction': 'Equity Direction',
                    'Harm_Duration': 'Equity Duration',
                    'Total_Score': 'Total Score'
                })
                numeric_cols = ['Total Magnitude', 'Reach', 'Equity Direction', 'Equity Duration', 'Total Score']
                profile_df[numeric_cols] = profile_df[numeric_cols].applymap(lambda x: f"{x:.2f}")
                
                # Display the styled table
                st.dataframe(profile_df, use_container_width=True)
            else:
                st.info("No data found for the selected sector.")
    else:
        st.info("Add stocks to your portfolio or map Kataly holdings to view the Sankey diagram.")
    


# Load your JSON file
    st.markdown(" ")
    st.subheader(f"New Racial Justice Research Alert", divider="blue")
    st.markdown(" ")
    data = load_json_data('perplexity_analysis_results_20250528_180826.json')
    df = pd.DataFrame(data)
    df['New_Evidence'] = df['New_Evidence'].replace(
        'Error processing response', 
        'Unable to retrieve new evidence at this time. Please check back later or contact support.'
    )
    st.dataframe(df[["Sector","SDH_Category","SDH_Indicator","Harm_Description","Original_Claim_Quantification","New_Evidence"]],use_container_width=True)

    st.markdown(" ")

      
    st.subheader(f"Legal Disclamer and Score Methodology", divider="blue")
    st.markdown(" ")
    with st.expander("Legal Disclaimer", expanded=False):
        disclaimer_content = read_disclaimer_file("Kataly-Disclaimer.docx")
        st.markdown(disclaimer_content, unsafe_allow_html=True)
  
    with st.expander("Score Methodology", expanded=False):
        render_pdf_pages("Corporate Racial Equity Score - Methodology Statement (1).pdf")




def render_pdf_pages(file_path):
    try:
        # Open PDF
        doc = fitz.open(file_path)

        # Render scrollable container with fixed height
        st.markdown("""
            <style>
            .pdf-scroll {
                max-height: 500px;
                overflow-y: scroll;
                padding-right: 10px;
                border: 1px solid #ccc;
                background-color: #fff;
            }
            </style>
            <div class="pdf-scroll">
        """, unsafe_allow_html=True)

        # Render pages inside scrollable container using st.image base64 technique
        for page_num in range(len(doc)):
            page = doc[page_num]
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            st.image(img, caption=f"Page {page_num + 1}", use_container_width=True)

        # Close container
        st.markdown("</div>", unsafe_allow_html=True)

        doc.close()

    except Exception as e:
        st.error(f"Error rendering PDF: {e}")



import json
def load_json_data(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def generate_report(selected_sector, profile_df, portfolio_harm_scores):
    # Create a buffer to store the report
    buffer = io.BytesIO()
    
    # Create a PDF with reportlab
    from reportlab.lib.pagesizes import letter, landscape
    from reportlab.pdfgen import canvas
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER
    
    # Use landscape orientation for more width
    doc = SimpleDocTemplate(buffer, pagesize=landscape(letter))
    elements = []
    
    # Add styles
    styles = getSampleStyleSheet()
    title_style = styles['Heading1']
    subtitle_style = styles['Heading2']
    normal_style = styles['Normal']
    
    # Create a custom style for table cells with word wrapping
    cell_style = ParagraphStyle(
        'CellStyle',
        parent=styles['Normal'],
        fontSize=8,
        leading=10,
        wordWrap='CJK'
    )
    
    # Add title
    elements.append(Paragraph(f"Corporate Racial Equity Intelligence Report", title_style))
    elements.append(Spacer(1, 12))
    
    # Add sector info
    elements.append(Paragraph(f"Sector: {selected_sector}", subtitle_style))
    elements.append(Spacer(1, 12))
    
    # Add portfolio metrics
    elements.append(Paragraph("Portfolio Racial Equity Summary", subtitle_style))
    
    portfolio_data = [
        ["Metric", "Value"],
        ["Average Portfolio Equity Score", f"{portfolio_harm_scores['average_score']:.1f}"],
        ["Total Portfolio Equity Score", f"{int(portfolio_harm_scores['total_score']):,}"],
        ["Total Portfolio Equity Quartile", f"{portfolio_harm_scores['quartile']}"]
    ]
    
    portfolio_table = Table(portfolio_data, colWidths=[300, 200])
    portfolio_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    elements.append(portfolio_table)
    elements.append(Spacer(1, 16))
    
    # Add sector harm profile table
    elements.append(Paragraph(f"Detailed {selected_sector} Sector Equity Profile", subtitle_style))
    elements.append(Spacer(1, 12))
    
    # Convert profile_df to a list for the table, but wrap text in Paragraph objects
    # for proper word wrapping
    header_row = [Paragraph(f"<b>{col}</b>", cell_style) for col in profile_df.columns]
    
    table_data = [header_row]  # Header row
    
    for _, row in profile_df.iterrows():
        table_row = []
        for value in row.values:
            # Convert the value to string and wrap in a Paragraph for text wrapping
            text = str(value)
            p = Paragraph(text, cell_style)
            table_row.append(p)
        table_data.append(table_row)
    
    # Create the table with more appropriate column widths
    # Adjust these widths based on your specific content
    harm_table = Table(
        table_data, 
        colWidths=[80, 80, 150, 80, 60, 60, 70, 70, 60],
        repeatRows=1  # Repeat header row on each page
    )
    
    # Add style to the table
    harm_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    elements.append(harm_table)
    
    # Build the PDF
    doc.build(elements)
    
    buffer.seek(0)
    return buffer.getvalue()
                
if __name__ == "__main__":
    main()
