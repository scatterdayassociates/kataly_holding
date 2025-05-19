
import streamlit as st
import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine
import plotly.graph_objects as go
from yfinance import Ticker
from sec_api import MappingApi
import requests
import time
from functools import lru_cache
import concurrent.futures
import io
import pandas as pd
import requests
import datetime
from dateutil.relativedelta import relativedelta
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
        'Ticker', 'Sector', 'Industry', 'Market Cap', 'Open', 
        'High', 'Low', 'Beta', 'Trailing PE', '52 Week High'
    ])

# Initialize session state for bond holdings if not exists
if 'df_bonds' not in st.session_state:
    st.session_state.df_bonds = pd.DataFrame(columns=[
        'CUSIP', 'Name', 'Industry Group', 'Issuer', 'Units', 'Purchase Price',
        'Purchase Date', 'Current Price', 'Coupon', 'Maturity Date', 'YTM',
        'Market Value', 'Total Cost', 'Price Return', 'Income Return', 'Total Return'
    ])
# Cache for API responses - persists across reruns
if 'sector_cache' not in st.session_state:
    st.session_state.sector_cache = {}

# Cache for Kataly holdings to avoid repeated DB queries
if 'kataly_holdings' not in st.session_state:
    st.session_state.kataly_holdings = None

# Cache for sector harm scores
if 'sector_harm_scores' not in st.session_state:
    st.session_state.sector_harm_scores = {}

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

@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_sector_data(sector):
    try:
        engine = get_db_engine()
        
        # Method 1: Using SQLAlchemy text() with parameterized query (RECOMMENDED)
        from sqlalchemy import text
        query = text("""
            SELECT Sector, SDH_Category, SDH_Indicator, Harm_Description, 
                  Claim_Quantification, Harm_Typology, Total_Magnitude, Reach, 
                  Harm_Direction, Harm_Duration, Total_Score 
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

# Function to prepare Sankey data - no changes needed
def prepare_sankey_data(df, sector):
    harm_typologies = df['Harm_Typology'].unique().tolist()
    sdh_categories = df['SDH_Category'].unique().tolist()
    sdh_indicators = df['SDH_Indicator'].unique().tolist()

    node_dict = {}
    node_list = []

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
    
    # Track added links to avoid duplicates
    added_links = set()

    # Create dictionaries to aggregate scores by different levels
    harm_typology_scores = {}
    sdh_category_scores = {}
    
    # First pass to calculate sums for each group
    for _, row in df.iterrows():
        harm_typology = row['Harm_Typology']
        sdh_category = row['SDH_Category']
        raw_score = float(row['Total_Score'])
        
        if harm_typology not in harm_typology_scores:
            harm_typology_scores[harm_typology] = 0
        harm_typology_scores[harm_typology] += raw_score
        
        if sdh_category not in sdh_category_scores:
            sdh_category_scores[sdh_category] = 0
        sdh_category_scores[sdh_category] += raw_score

    # Calculate total score for sector
    sector_total_score = sum(harm_typology_scores.values())
    
    # Second pass to create links
    for _, row in df.iterrows():
        sector_index = node_dict[sector]
        harm_typology = row['Harm_Typology']
        harm_typology_index = node_dict[harm_typology]
        sdh_category = row['SDH_Category']
        sdh_category_index = node_dict[sdh_category]
        sdh_indicator = row['SDH_Indicator']
        sdh_indicator_index = node_dict[sdh_indicator]
        raw_score = float(row['Total_Score'])

        # Link from sector to harm typology (only add once per harm typology)
        link_key = f"{sector_index}-{harm_typology_index}"
        if link_key not in added_links:
            source.append(sector_index)
            target.append(harm_typology_index)
            value.append(harm_typology_scores[harm_typology])
            added_links.add(link_key)

        # Link from harm typology to SDH category (only add once per category per harm typology)
        link_key = f"{harm_typology_index}-{sdh_category_index}"
        if link_key not in added_links:
            source.append(harm_typology_index)
            target.append(sdh_category_index)
            # Calculate the portion of the category score that belongs to this harm typology
            category_harm_score = 0
            for _, r in df[(df['Harm_Typology'] == harm_typology) & (df['SDH_Category'] == sdh_category)].iterrows():
                category_harm_score += float(r['Total_Score'])
            value.append(category_harm_score)
            added_links.add(link_key)

        # Link from SDH category to SDH indicator
        link_key = f"{sdh_category_index}-{sdh_indicator_index}"
        if link_key not in added_links:
            source.append(sdh_category_index)
            target.append(sdh_indicator_index)
            value.append(raw_score)  # Use the Total_Score for this specific indicator
            added_links.add(link_key)

    return node_list, source, target, value

# Function to style Sankey nodes - no changes needed
def style_sankey_nodes(node_list, sector):
    colors = ["blue", "green", "orange", "red"]  # Colors for each level
    node_colors = []

    # Sector level
    node_colors.append(colors[0])

    # Harm Typology level
    harm_typologies = node_list[1:len(node_list)//3 + 1]
    for _ in harm_typologies:
        node_colors.append(colors[1])

    # SDH Category level
    sdh_categories = node_list[len(node_list)//3 + 1:2*len(node_list)//3 + 1]
    for _ in sdh_categories:
        node_colors.append(colors[2])

    # SDH Indicator level
    sdh_indicators = node_list[2*len(node_list)//3 + 1:]
    for _ in sdh_indicators:
        node_colors.append(colors[3])

    return node_colors

# Function to calculate portfolio harm scores
def calculate_portfolio_harm_scores(kataly_holdings):
    if kataly_holdings.empty or 'Sector' not in kataly_holdings.columns or 'Units' not in kataly_holdings.columns:
        return {
            'average_score': 0.0,
            'total_score': 0.0,
            'quartile': "N/A"
        }
    
    # Calculate weighted harm scores
    sector_units = {}
    sector_harm_scores = {}
    weighted_scores = []
    total_units = 0
    
    # Group holdings by sector and calculate units
    for _, row in kataly_holdings.iterrows():
        sector = row['Sector']
        if pd.isna(sector) or sector == 'N/A':
            continue
        
        units = row.get('Units', 0)
        if pd.isna(units):
            units = 0
            
        try:
            units = float(units)
        except (ValueError, TypeError):
            units = 0
            
        if units > 0:
            if sector not in sector_units:
                sector_units[sector] = 0
            sector_units[sector] += units
            total_units += units
    
    # Get harm scores for each sector and calculate weighted scores
    for sector, units in sector_units.items():
        harm_score = get_sector_min_max_norm(sector)
        sector_harm_scores[sector] = harm_score
        weighted_scores.append(harm_score * units)
    
    # Calculate average and total harm scores
    if total_units > 0:
        average_score = sum(weighted_scores) / total_units
        total_score = sum(weighted_scores)
    else:
        average_score = 0.0
        total_score = 0.0
    
    # Determine quartile
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

# Add caching status indicator to sidebar
# Add caching status indicator to sidebar
def show_sidebar():
    with st.sidebar:
        st.image("Kataly-Featured-Logo.png")
        st.header("Add Stock to Portfolio")
        ticker = st.text_input("Enter a stock ticker", placeholder="AAPL", key="stock_ticker_input")
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
                stock_info = get_stock_info(ticker)
                stock_info['Ticker'] = ticker
                new_row = pd.DataFrame([stock_info])
                if ticker in st.session_state.df_stock_info['Ticker'].values:
                    st.warning(f"{ticker} is already in your portfolio.")
                else:
                    st.session_state.df_stock_info = pd.concat(
                        [st.session_state.df_stock_info, new_row],
                        ignore_index=True
                    )
                    st.success(f"Added {ticker} to the portfolio.")

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
            selected_sector = None
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
                                     'Total_Magnitude', 'Reach', 'Harm_Direction', 'Harm_Duration', 'Total_Score']]
                    profile_df = profile_df.rename(columns={
                        'SDH_Indicator': 'SDH Indicator',
                        'SDH_Category': 'SDH Category',
                        'Harm_Typology': 'Harm Typology',
                        'Total_Magnitude': 'Total Magnitude',
                        'Harm_Direction': 'Harm Direction',
                        'Harm_Duration': 'Harm Duration',
                        'Total_Score': 'Total Score'
                    })

                    portfolio_harm_scores = calculate_portfolio_harm_scores(kataly_holdings)

                    if selected_sector and profile_df is not None:
                        pdf_buffer = generate_report(selected_sector, profile_df, portfolio_harm_scores)
                        st.download_button(
                            label="Download PDF Report",
                            data=pdf_buffer,
                            file_name=f"racial_harm_report_{selected_sector}.pdf",
                            mime="application/pdf",
                            key="pdf_download"
                        )


def main():
    # Display sidebar
    show_sidebar()
    
    st.title("Corporate Racial Harm Intelligence Canvas")
    
    # Portfolio Holdings Summary
    st.text("")
    st.subheader("Portfolio Holdings Summary", divider="blue")
    st.markdown(" ")
    
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
    tab1, tab2 = st.tabs([ "Kataly Bond Portfolio","Stocks"])
    
    with tab1:
        if not kataly_holdings.empty:
            # Display the Kataly bond holdings with sector mapping
            st.dataframe(kataly_holdings, use_container_width=True)
        
    
    with tab2:
        if not st.session_state.df_stock_info.empty:
            # Display the stock info with sector mapping
            st.dataframe(st.session_state.df_stock_info, use_container_width=True)
        
    
    
    
    if not st.session_state.df_bonds.empty:
        # Display bond holdings
        st.text("Bond Holdings")
        st.dataframe(
            st.session_state.df_bonds[['CUSIP', 'Industry Group', 'Issuer', 'Units','Current Price' ,'Purchase Price','Coupon','Price Return','Income Return', 'Total Return']],
            use_container_width=True
        )
        

    # Add space 
    st.markdown(" ") 
    
    # NEW SECTION: Portfolio Racial Harm Summary 
    st.markdown("<h3 style='color: #333; padding-bottom: 10px; border-bottom: 2px solid #6082B6;'>Portfolio Racial Harm Summary</h3>", unsafe_allow_html=True) 

    # Calculate portfolio harm scores
    portfolio_harm_scores = calculate_portfolio_harm_scores(kataly_holdings) 

    # Define the custom CSS for the boxes
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
    </style>
    """, unsafe_allow_html=True)

    # Create the container for the metrics
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)

    # Create each box with HTML/CSS
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-title">Average Portfolio Harm Score</div>
            <div class="metric-value">{portfolio_harm_scores['average_score']:.1f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-title">Total Portfolio Harm Score</div>
            <div class="metric-value">{int(portfolio_harm_scores['total_score']):,}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-title">Total Portfolio Harm Quartile</div>
            <div class="metric-value">{portfolio_harm_scores['quartile']}</div>
            
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
    
    # Add space
    st.markdown(" ")
    
    # Corporate Racial Harm Canvas section
    st.subheader(f"Corporate Racial Harm Canvas", divider="blue")
    
    # Collect available sectors from both stock info and Kataly holdings
    stock_sectors = st.session_state.df_stock_info['Sector'].unique().tolist()
    kataly_sectors = []
    if not kataly_holdings.empty and 'Sector' in kataly_holdings.columns:
        kataly_sectors = kataly_holdings['Sector'].unique().tolist()
    
    # Combine sectors and filter out invalid values
    all_sectors = list(set(stock_sectors + kataly_sectors))
    all_sectors = [sector for sector in all_sectors if sector != 'N/A' and pd.notna(sector)]
    
    if all_sectors:
        selected_sector = st.selectbox("Select a sector for the Sankey diagram", all_sectors)
        
        # Fetch data for the selected sector
        with st.spinner(f"Loading data for {selected_sector} sector..."):
            df = fetch_sector_data(selected_sector)
        
        # Prepare data for the Sankey diagram
        # Add this after displaying the Sankey diagram and tables
        if not df.empty:
            node_list, source, target, value = prepare_sankey_data(df, selected_sector)
            node_colors = style_sankey_nodes(node_list, selected_sector)
            
            # Create the Sankey diagram
            fig = go.Figure(data=[go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=node_list,
                    color=node_colors
                   
                ),
                link=dict(
                    source=source,
                    target=target,
                    value=value
                )
            )])
            
            fig.update_layout(title_text="")
            st.info(f"Corporate Racial Harm Canvas for {selected_sector} Sector")
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed Sector Harm Profile section
            st.subheader(f"Detailed {selected_sector} Sector Harm Profile", divider="blue")
            st.markdown(" ")
            
            # Create a DataFrame with required columns
            profile_df = df[['SDH_Indicator', 'SDH_Category', 'Harm_Description', 'Harm_Typology', 
                        'Total_Magnitude', 'Reach', 'Harm_Direction', 'Harm_Duration',"Total_Score"]]
            
            # Rename columns for display
            profile_df = profile_df.rename(columns={
                'SDH_Indicator': 'SDH Indicator',
                'SDH_Category': 'SDH Category', 
                'Harm_Typology': 'Harm Typology',
                'Total_Magnitude': 'Total Magnitude',
                'Harm_Direction': 'Harm Direction',
                'Harm_Duration': 'Harm Duration',
                'Total_Score': 'Total Score'
            })
            
            # Display the styled table
            st.dataframe(profile_df, use_container_width=True)
            
            
    
        else:
            st.info("No data found for the selected sector.")
    else:
        st.info("Add stocks to your portfolio or map Kataly holdings to view the Sankey diagram.")

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
    elements.append(Paragraph(f"Corporate Racial Harm Intelligence Report", title_style))
    elements.append(Spacer(1, 12))
    
    # Add sector info
    elements.append(Paragraph(f"Sector: {selected_sector}", subtitle_style))
    elements.append(Spacer(1, 12))
    
    # Add portfolio metrics
    elements.append(Paragraph("Portfolio Racial Harm Summary", subtitle_style))
    
    portfolio_data = [
        ["Metric", "Value"],
        ["Average Portfolio Harm Score", f"{portfolio_harm_scores['average_score']:.1f}"],
        ["Total Portfolio Harm Score", f"{int(portfolio_harm_scores['total_score']):,}"],
        ["Total Portfolio Harm Quartile", f"{portfolio_harm_scores['quartile']}"]
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
    elements.append(Paragraph(f"Detailed {selected_sector} Sector Harm Profile", subtitle_style))
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
