import os
import csv
import logging
from re import X
import pandas as pd
import requests
from simple_salesforce import Salesforce, SalesforceResourceNotFound, SalesforceMalformedRequest
from dotenv import load_dotenv
import time
from bs4 import BeautifulSoup

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create a custom requests session
session = requests.Session()

# EITHER disable SSL verification (for testing only)
session.verify = False

# OR provide your corporate CA bundle
# session.verify = "/path/to/corporate-ca.pem"

def init_salesforce():
    """
    Initialize Salesforce connection using environment variables:
      - SF_USERNAME
      - SF_PASSWORD
      - SF_SECURITY_TOKEN
      - SF_DOMAIN (optional, defaults to 'login') for production or 'test' for sandbox
    Returns:
        sf (simple_salesforce.Salesforce): authenticated client
    Raises:
        ValueError: if required credentials are missing
    """
    username = os.getenv('SF_USERNAME')
    password = os.getenv('SF_PASSWORD')
    token = os.getenv('SF_SECURITY_TOKEN')
    domain = os.getenv('SF_DOMAIN', 'login')  # 'login' or 'test'

    if not all([username, password, token]):
        logger.error("Missing Salesforce credentials. Ensure SF_USERNAME, SF_PASSWORD, and SF_SECURITY_TOKEN are set.")
        raise ValueError("Salesforce credentials not provided in environment variables.")

    # Initialize Salesforce using the session
    sf = Salesforce(
        username="a128842@kp.org.hr",
        password="Ishanj2004@123",
        security_token="EhqJFYcVOexTVP2izVZuMePYX",
        domain="login",
        session=session
    )
    return sf


def fetch_article_fields(article_id, sf):
    """
    Fetch key fields for a given KnowledgeArticleVersion record.

    Args:
        article_id (str): Salesforce record ID (e.g., ka03...).
        sf (Salesforce): authenticated client

    Returns:
        list or None: [Id, KnowledgeArticleId, Title, FunctionalArea, FunctionalSubject, ArticleBody, LastPublishedDate]
                      or None if fetch failed.
    """
    fields = [
        'Id',
        'KnowledgeArticleId',
        'Title',
        'Functional_Area__c',
        'Functional_Subject__c',
        'ArticleBody',
        'LastPublishedDate'
    ]

    try:
        record = sf.Knowledge__kav.get(article_id)
        return [
            record.get('Id'),
            record.get('KnowledgeArticleId'),
            record.get('Title', ''),
            record.get('Functional_Area__c', ''),
            record.get('Functional_Subject__c', ''),
            record.get('ArticleBody', ''),
            record.get('LastPublishedDate', '')
        ]
    except SalesforceResourceNotFound:
        logger.warning(f"Article {article_id} not found in Salesforce.")
        return None
    except SalesforceMalformedRequest as e:
        logger.error(f"Malformed request for ID {article_id}: {e.content}")
        return None
    except Exception as e:
        logger.exception(f"Unexpected error fetching {article_id}: {e}")
        return None


def append_articles_to_csv(article_ids, output_csv, sf):
    """
    Loop through a list of article IDs, fetch their fields, and append to a CSV file.
    Creates the file with header if it does not exist.

    Args:
        article_ids (list of str): list of Salesforce record IDs
        output_csv (str): path to output CSV file
        sf (Salesforce): authenticated client
    """
    header = [
        'Id',
        'KnowledgeArticleId',
        'Title',
        'Functional_Area',
        'Functional_Subject',
        'ArticleBody',
        'LastPublishedDate'
    ]

    file_exists = os.path.isfile(output_csv)
    with open(output_csv, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        for aid in article_ids:
            row = fetch_article_fields(aid, sf)
            if row:
                writer.writerow(row)
                logger.info(f"Appended article {aid} to CSV.")
            else:
                logger.info(f"Skipping article {aid} due to fetch error.")


def clean_html(raw_html):
    """
    Convert messy HTML into readable plain text.
    - Preserves bullet points
    - Converts <br> and <li> into line breaks
    - Removes all tags and styles
    """
    if raw_html is None:
        return ""

    # Parse with BeautifulSoup
    soup = BeautifulSoup(raw_html, "html.parser")

    # Replace <br> and <p> with newlines
    for br in soup.find_all(["br", "p", "div"]):
        br.append("\n")

    # Convert <li> to "- " prefix
    for li in soup.find_all("li"):
        li.insert(0, "- ")

    # Extract all text, preserving newlines
    text = soup.get_text(separator=" ", strip=True)

    # Collapse excessive spaces/newlines
    text = "\n".join(line.strip() for line in text.splitlines() if line.strip())

    return text

def main():
    """
    Main entry point.
    Reads a CSV or text file of article IDs (one per line),
    initializes Salesforce connection, and appends article data to output CSV.

    Environment Variables:
      - ARTICLE_IDS_FILE: path to input file (default 'Contact_Center_Knowledge_Articles.csv')
      - OUTPUT_CSV: path to output CSV (default 'articles_export.csv')
    """
    input_file = os.getenv('ARTICLE_IDS_FILE', 'Contact_Center_Knowledge_Articles.csv')
    output_csv = os.getenv('OUTPUT_CSV', 'articles_export.csv')

    if not os.path.isfile(input_file):
        logger.error(f"Input file {input_file} not found.")
        return

    # Read article IDs from input file
    df = pd.read_csv(input_file, dtype={"Article Number": str})
    article_numbers = df["Article Number"].tolist()
    print(df["Article Number"].head())

    

    if not article_numbers:
        logger.error("No article IDs found in input file.")
        return
    
    print(article_numbers)
    
    # Build a comma-separated string of quoted article numbers
    article_numbers_str = ",".join(f"'{num}'" for num in article_numbers)
    
    # Initialize Salesforce client
    sf = init_salesforce()

    desc = sf.Knowledge__kav.describe()
    for field in desc["fields"]:
        print(field["name"])

    # 4) Query once to get the real record IDs
    mapping_query = f"""
    SELECT Id, ArticleNumber
    FROM Knowledge__kav
    WHERE ArticleNumber IN ({article_numbers_str})
    """
    mapping_res = sf.query_all(mapping_query)

    # 5) Build your map: normalized number → record ID
    id_map = {
        rec["ArticleNumber"]: rec["Id"]
        for rec in mapping_res["records"]
    }

    # 6) Compute the list of real IDs to fetch
    real_ids = [
        id_map[num]
        for num in article_numbers
        if num in id_map
    ]

    missing = set(article_numbers) - set(id_map.keys())
    if missing:
        logger.warning(f"No record IDs found for: {missing}")

    # 7) Finally, append using the real Salesforce IDs
    # append_articles_to_csv(real_ids, output_csv, sf)
    # Now write all records to CSV

    # Instead of append_articles_to_csv(), do it all here:

    id_list_str = ",".join(f"'{rid}'" for rid in real_ids)

    detail_query = f"""
    SELECT
    Id,
    KnowledgeArticleId,
    Title,
    HRHD_CCS_HRHD_Article_Body__c,
    Summary,
    HRHD_CCS_HRHD_Keywords__c,
    LastPublishedDate
    FROM Knowledge__kav
    WHERE Id IN ({id_list_str})
    """
    detail_res = sf.query_all(detail_query)

    header = [
        "Id",
        "KnowledgeArticleId",
        "Title",
        "ArticleBody",
        "Summary",
        "Keywords",
        "LastPublishedDate"
    ]

    is_new = not os.path.isfile(output_csv)

    with open(output_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(header)
        for rec in detail_res["records"]:
            writer.writerow([
                rec.get("Id", ""),
                rec.get("KnowledgeArticleId", ""),
                rec.get("Title", ""),
                rec.get("HRHD_CCS_HRHD_Article_Body__c", ""),
                rec.get("Summary", ""),
                rec.get("HRHD_CCS_HRHD_Keywords__c", ""),
                rec.get("LastPublishedDate", "")
            ])

    logger.info("Finished exporting all articles.")
    


if __name__ == '__main__':
    main()
