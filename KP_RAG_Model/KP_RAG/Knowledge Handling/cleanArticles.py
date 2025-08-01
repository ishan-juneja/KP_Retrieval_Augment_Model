from bs4 import BeautifulSoup
import pandas as pd

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

def clean_article_body_column(input_csv, output_csv, body_column="ArticleBody"):
    """
    Loads a CSV, cleans HTML in the specified column,
    creates a new 'CleanedArticleBody' column, drops the old column,
    and saves to a new file.

    Args:
        input_csv (str): Path to input CSV.
        output_csv (str): Path to output cleaned CSV.
        body_column (str): Name of the column containing HTML.
    """
    # Load data
    df = pd.read_csv(input_csv)

    if body_column not in df.columns:
        raise ValueError(f"Column '{body_column}' not found in CSV.")

    # Clean HTML
    df["CleanedArticleBody"] = df[body_column].apply(clean_html)

    # Drop the old HTML column
    df = df.drop(columns=[body_column])

    # Save to new CSV
    df.to_csv(output_csv, index=False, encoding="utf-8")

    print(f"✅ Cleaned CSV saved to: {output_csv}")

