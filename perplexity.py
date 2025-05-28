import os
import pandas as pd
import schedule
import time
from datetime import datetime
import logging
from typing import Dict, List, Any
import json
from pathlib import Path

# LangChain imports
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.callbacks import get_openai_callback
from langchain.document_loaders import DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

# HTTP client for Perplexity API
import requests
import httpx

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('perplexity_scheduler.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PerplexityScheduler:
    """
    A comprehensive scheduler that processes Excel files and runs weekly queries 
    via Perplexity API using LangChain framework.
    """
    
    def __init__(self, perplexity_api_key: str, excel_file_path: str, output_dir: str = "output"):
        """
        Initialize the scheduler with API key and file paths.
        
        Args:
            perplexity_api_key: Your Perplexity API key
            excel_file_path: Path to the Excel file to process
            output_dir: Directory to save output files
        """
        self.perplexity_api_key = perplexity_api_key
        self.excel_file_path = excel_file_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Perplexity API configuration
        self.perplexity_url = "https://api.perplexity.ai/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.perplexity_api_key}",
            "Content-Type": "application/json"
        }
        
        # Initialize LangChain components
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        logger.info("PerplexityScheduler initialized successfully")
    
    def load_excel_data(self) -> pd.DataFrame:
        """
        Load and validate Excel data.
        
        Returns:
            DataFrame containing the Excel data
        """
        try:
            # Support both .xlsx and .xls files
            if self.excel_file_path.endswith('.xlsx'):
                df = pd.read_excel(self.excel_file_path, engine='openpyxl')
            else:
                df = pd.read_excel(self.excel_file_path, engine='xlrd')
            
            # Validate required columns
            required_columns = [
                'Sector', 'SDH_Category', 'SDH_Indicator', 
                'Harm_Description', 'Claim_Quantification',
                'Citation-1', 'Citation-2'
            ]
            
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.warning(f"Missing columns: {missing_columns}")
                # Create missing columns with empty values
                for col in missing_columns:
                    df[col] = ""
            
            logger.info(f"Loaded Excel data with {len(df)} rows and {len(df.columns)} columns")
            return df
            
        except Exception as e:
            logger.error(f"Error loading Excel file: {str(e)}")
            raise
    
    def create_vectorstore_from_data(self, df: pd.DataFrame):
        """
        Create a vector store from the DataFrame for similarity searches.
        
        Args:
            df: DataFrame containing the data
        """
        try:
            # Convert DataFrame to documents for LangChain
            loader = DataFrameLoader(df, page_content_column="Claim_Quantification")
            documents = loader.load()
            
            # Split documents
            texts = self.text_splitter.split_documents(documents)
            
            # Note: You would need OpenAI API key for embeddings
            # For this example, we'll store the data structure for reference
            self.data_store = df
            logger.info("Data store created successfully")
            
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            self.data_store = df
    
    def build_query_for_row(self, row: pd.Series) -> str:
        """
        Build a specific query for each row in the dataset.
        
        Args:
            row: A pandas Series representing a row from the DataFrame
            
        Returns:
            Formatted query string for Perplexity API
        """
        sector = row.get('Sector', '')
        sdh_category = row.get('SDH_Category', '')
        sdh_indicator = row.get('SDH_Indicator', '')
        harm_description = row.get('Harm_Description', '')
        existing_quantification = row.get('Claim_Quantification', '')
        citation1 = row.get('Citation-1', '')
        citation2 = row.get('Citation-2', '')
        
        query = f"""
        Please search for new research related to disparate impacts in the {sector} sector, 
        specifically focusing on {sdh_category} (Social Determinants of Health category) 
        and {sdh_indicator} (SDH indicator).
        
        The harm being investigated is: {harm_description}
        
        Current evidence includes: {existing_quantification}
        
        Existing citations to avoid duplication:
        - Citation 1: {citation1}
        - Citation 2: {citation2}
        
        Please find NEW research that is not already covered in the existing evidence 
        and citations. Focus on recent studies, reports, or academic papers published 
        in the last 2 years that provide quantifiable impacts or new insights.
        
        Provide findings in a structured format with:
        1. New evidence description
        2. Quantifiable impacts (if available)
        3. Full citation information
        4. Publication date
        5. Brief summary of methodology
        """
        
        return query
    
    def query_perplexity_api(self, query: str, model: str = "sonar-deep-research") -> Dict[str, Any]:
        """
        Send query to Perplexity API using the Sonar model.
        
        Args:
            query: The query string to send
            model: The Perplexity model to use
            
        Returns:
            API response as dictionary
        """
        try:
            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a research assistant specialized in finding and analyzing academic research related to social determinants of health and industry impacts. Provide detailed, accurate, and well-cited responses."
                    },
                    {
                        "role": "user",
                        "content": query
                    }
                ],
                "max_tokens": 4000,
                "temperature": 0.1,  # Low temperature for more factual responses
                "top_p": 0.9,
                "return_citations": True,
                "search_domain_filter": ["pubmed.ncbi.nlm.nih.gov", "scholar.google.com", "jstor.org", "springer.com", "sciencedirect.com"],
                "return_images": False,
                "return_related_questions": False,
                "search_recency_filter": "month",  # Focus on recent research
                "top_k": 0,
                "stream": False,
                "presence_penalty": 0,
                "frequency_penalty": 1
            }
            
            response = requests.post(
                self.perplexity_url,
                headers=self.headers,
                json=payload,
                timeout=60000
            )
            
            response.raise_for_status()
            result = response.json()
            
            logger.info(f"Perplexity API call successful")
            return result
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling Perplexity API: {str(e)}")
            return {"error": str(e)}
        except Exception as e:
            logger.error(f"Unexpected error in API call: {str(e)}")
            return {"error": str(e)}
    
    def process_api_response(self, response: Dict[str, Any]) -> Dict[str, str]:
        """
        Process and extract relevant information from API response.
        
        Args:
            response: Raw API response
            
        Returns:
            Processed response with extracted information
        """
        try:
            if "error" in response:
                return {
                    "new_evidence": "Error occurred during API call",
                    "new_citation": response["error"],
                    "status": "error"
                }
            
            # Extract the main content
            content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            # Extract citations if available
            citations = response.get("citations", [])
            citation_text = " | ".join([f"{cite.get('title', '')}: {cite.get('url', '')}" for cite in citations[:3]])
            
            return {
                "new_evidence": content,
                "new_citation": citation_text,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error processing API response: {str(e)}")
            return {
                "new_evidence": "Error processing response",
                "new_citation": str(e),
                "status": "error"
            }
    
    def process_single_row(self, row: pd.Series, index: int) -> Dict[str, str]:
        """
        Process a single row from the dataset.
        
        Args:
            row: Pandas Series representing the row
            index: Row index for logging
            
        Returns:
            Dictionary with processed results
        """
        logger.info(f"Processing row {index + 1}: {row.get('Sector', 'Unknown Sector')}")
        
        # Build query for this specific row
        query = self.build_query_for_row(row)
        
        # Query Perplexity API
        api_response = self.query_perplexity_api(query)
        
        # Process response
        processed_result = self.process_api_response(api_response)
        
        # Add original row data
        result = {
            "Sector": row.get('Sector', ''),
            "SDH_Category": row.get('SDH_Category', ''),
            "SDH_Indicator": row.get('SDH_Indicator', ''),
            "Harm_Description": row.get('Harm_Description', ''),
            "Original_Claim_Quantification": row.get('Claim_Quantification', ''),
            "New_Evidence": processed_result.get("new_evidence", ""),
            "New_Citation": processed_result.get("new_citation", ""),
            "Processing_Status": processed_result.get("status", "unknown")
        }
        
        # Add small delay to respect API rate limits
        time.sleep(2)
        
        return result
    
    def run_weekly_analysis(self):
        """
        Main function to run the weekly analysis.
        """
        try:
            logger.info("Starting weekly Perplexity analysis")
            
            # Load Excel data
            df = self.load_excel_data()
            
            # Create vector store for reference
            self.create_vectorstore_from_data(df)
            
            # Process each row
            results = []
            total_rows = len(df)
            
            for index, row in df.iterrows():
                try:
                    result = self.process_single_row(row, index)
                    results.append(result)
                    
                    logger.info(f"Completed {index + 1}/{total_rows} rows")
                    
                except Exception as e:
                    logger.error(f"Error processing row {index + 1}: {str(e)}")
                    # Add error result
                    results.append({
                        "Sector": row.get('Sector', ''),
                        "SDH_Category": row.get('SDH_Category', ''),
                        "SDH_Indicator": row.get('SDH_Indicator', ''),
                        "Harm_Description": row.get('Harm_Description', ''),
                        "Original_Claim_Quantification": row.get('Claim_Quantification', ''),
                        "New_Evidence": f"Error processing: {str(e)}",
                        "New_Citation": "N/A",
                        "Processing_Status": "error"
                    })
            
            # Save results
            self.save_results(results)
            logger.info("Weekly analysis completed successfully")
            
        except Exception as e:
            logger.error(f"Error in weekly analysis: {str(e)}")
            raise
    
    def save_results(self, results: List[Dict[str, str]]):
        """
        Save results to Excel file with timestamp.
        
        Args:
            results: List of result dictionaries
        """
        try:
            # Create DataFrame from results
            results_df = pd.DataFrame(results)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"perplexity_analysis_results_{timestamp}.xlsx"
            filepath = self.output_dir / filename
            
            # Save to Excel
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                results_df.to_excel(writer, sheet_name='Analysis_Results', index=False)
                
                # Also save summary statistics
                summary_data = {
                    "Total_Rows_Processed": len(results),
                    "Successful_Queries": len([r for r in results if r.get('Processing_Status') == 'success']),
                    "Failed_Queries": len([r for r in results if r.get('Processing_Status') == 'error']),
                    "Analysis_Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                summary_df = pd.DataFrame([summary_data])
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            logger.info(f"Results saved to: {filepath}")
            
            # Also save as JSON for programmatic access
            json_filename = f"perplexity_analysis_results_{timestamp}.json"
            json_filepath = self.output_dir / json_filename
            
            with open(json_filepath, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"JSON results saved to: {json_filepath}")
            
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            raise
    
    def schedule_weekly_run(self):
        """
        Schedule the analysis to run weekly.
        """
        # Schedule for every Monday at 9:00 AM
        schedule.every().monday.at("09:00").do(self.run_weekly_analysis)
        
        logger.info("Weekly schedule configured for Mondays at 9:00 AM")
        
        # Keep the scheduler running
        while True:
            schedule.run_pending()
            time.sleep(3600)  # Check every hour

def main():
    """
    Main function to initialize and start the scheduler.
    """
    # Configuration
    PERPLEXITY_API_KEY = "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"
    EXCEL_FILE_PATH = "Racial-Harm-Captal-Markets.xlsx"  # Update this path
    OUTPUT_DIR = "weekly_analysis_output"
    
    try:
        # Initialize scheduler
        scheduler = PerplexityScheduler(
            perplexity_api_key=PERPLEXITY_API_KEY,
            excel_file_path=EXCEL_FILE_PATH,
            output_dir=OUTPUT_DIR
        )
        
        # Option 1: Run immediately (for testing)
        print("Running immediate analysis for testing...")
        scheduler.run_weekly_analysis()
        
        # Option 2: Start weekly scheduler (uncomment to use)
        # print("Starting weekly scheduler...")
        # scheduler.schedule_weekly_run()
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()