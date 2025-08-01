import os
import time
from typing import Any
from dotenv import load_dotenv
import pandas as pd
import openai
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pinecone import Pinecone, PodSpec
from openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
os.environ["TOKENIZERS_PARALLELISM"] = "false"

model = SentenceTransformer("all-mpnet-base-v2")

subtopic_to_parent_map = {
    0: "Leave Management / FMLA",                       # Employee Leave Management (California, FMLA)
    1: "HR General / Operations",                       # Form Assistance & Status Check
    2: "Leave Management / FMLA",                       # Maternity Leave & FMLA Eligibility Clarification
    3: "Enrollment & Benefits",                         # Benefits Enrollment & COBRA Coverage
    4: "Job Changes & Terminations",                    # Job Transfer & Position Change Updates
    5: "Verification & Documentation",                  # Employment Verification Document Submission
    6: "HR General / Operations",                       # HR Service Center Case Resolution
    7: "Disability & State Claims",                     # Disability Claims & Salary Continuance (EDD, MetLife)
    8: "Enrollment & Benefits",                         # Benefits Plan & Severance Enrollment Issues
    9: "Payroll / Compensation",                        # Paycheck Discrepancies & Disability Coordination
    10: "Retirement",                                   # Retirement & Pension Contributions (Vanguard, Fidelity)
    11: "Enrollment & Benefits",                        # Adding Spouse or Dependents (Certificates)
    12: "Payroll / Compensation",                       # Bank Account Updates & Payment Recalls
    13: "Access & Technical Issues",                    # HR Systems & Employee Portals (MyHR, HRConnect)
    14: "Leave Management / FMLA",                      # State-Specific Leave Policies (Colorado, Washington)
    15: "Payroll / Compensation",                       # Department Transfer & Payroll Verification
    16: "Access & Technical Issues",                    # CRM System & Call Logging Issues
    17: "Retirement",                                   # Retirement Procedures & Kaiser Permanente Retirement Center
    18: "Taxes & Withholding",                          # Tax Withholding & Payroll Deductions
    19: "Payroll / Compensation",                       # Overpayment & Repayment Processes
    20: "Payroll / Compensation",                       # Wage Increase & Record Verification
    21: "Verification & Documentation",                 # Document Receipt & Processing Timeframes
    22: "Access & Technical Issues",                    # HR System Delegation & User Access
    23: "Enrollment & Benefits",                        # Flexible Spending Accounts (FSA/DCSA)
    24: "Verification & Documentation",                 # Wage Loss Verification (WLV) Letters
    25: "Payroll / Compensation",                       # Paycheck Deductions & Tax Coordination
    26: "Enrollment & Benefits",                        # Mercer Insurance & Benefit Deductions
    27: "Access & Technical Issues",                    # Communication & Call Connection Issues
    28: "Enrollment & Benefits",                        # Delta Dental Coverage Questions
    29: "Disability & State Claims",                    # MetLife Disability Insurance & Claim Processing
    30: "Enrollment & Benefits",                        # Tuition Reimbursement & Education Benefits
    31: "HR General / Operations",                      # Employee Personal Information Updates (Name, Address)
    32: "Enrollment & Benefits",                        # Spousal Insurance Surcharges & Billing
    33: "Job Changes & Terminations",                   # Training Application (TRA) Status & Withdrawals
    34: "Other / Miscellaneous",                        # Missing or Insufficient Information Provided
    35: "Enrollment & Benefits",                        # Dental Procedure & Coverage Queries
    36: "HR General / Operations",                      # Supervisor Escalation & Callback Requests
    37: "Payroll / Compensation",                       # ADP Paystub Retrieval & Documentation
    38: "Verification & Documentation",                 # Licensing, Certification & Registration (LCR Compliance)
    39: "Disability & State Claims",                    # Workers' Compensation & Injury Reporting
    40: "Leave Management / FMLA",                      # Family Leave (CFRA) Expansion Queries
    41: "Leave Management / FMLA",                      # Bereavement Leave Policies
    42: "Enrollment & Benefits",                        # Spousal Surcharge & Letter Clarification
    43: "Access & Technical Issues",                    # IT Account & Password Issues
    44: "Leave Management / FMLA",                      # Bereavement Policy Clarification
    45: "Disability & State Claims",                    # Workers' Compensation Claim Submission (Sedgwick)
    46: "Taxes & Withholding",                          # IRS Tax Withholding & Garnishment Issues
    47: "Taxes & Withholding",                          # Garnishment Processes & Debt Collection
    48: "Timekeeping & Scheduling",                     # Vacation Conversion & Absence Management
    49: "Leave Management / FMLA",                      # FMLA Leave Approval & Notifications
    50: "HR General / Operations",                      # HR Delegation Permissions & Submission Issues
    51: "Enrollment & Benefits",                        # Grandchild Dependent Addition Issues
    52: "Payroll / Compensation",                       # Per Diem Contract & Worker Classification
    53: "Enrollment & Benefits",                        # Death Reporting & Survivor Benefits
    54: "HR General / Operations",                      # OSHA Training & Case Escalations
    55: "Timekeeping & Scheduling",                     # Extended Sick Leave (ESL) Policy Clarification
    56: "Verification & Documentation",                 # Callback Verification & Contractor Security Protocols
    57: "Other / Miscellaneous",                        # Unclear or Incomplete Employee Queries
    58: "HR General / Operations"                       # Performance Improvement Plans (PIP) & Feedback
}

taxonomy_labels = [
    "Employee Leave Management (California, FMLA, Salesforce Portal)",              # topic 0
    "Form Assistance & Status Check",                                               # topic 1
    "Maternity Leave & FMLA Eligibility Clarification",                             # topic 2
    "Benefits Enrollment & COBRA Coverage",                                         # topic 3
    "Job Transfer & Position Change Updates",                                       # topic 4
    "Employment Verification Document Submission",                                  # topic 5
    "HR Service Center Case Resolution",                                            # topic 6
    "Disability Claims & Salary Continuance (EDD, MetLife, SDI)",                   # topic 7
    "Benefits Plan & Severance Enrollment Issues",                                  # topic 8
    "Paycheck Discrepancies & Disability Coordination",                             # topic 9
    "Retirement & Pension Contributions (Vanguard, Fidelity)",                      # topic 10
    "Adding Spouse or Dependents (Certificates & Documentation)",                   # topic 11
    "Bank Account Updates & Payment Recalls",                                       # topic 12
    "HR Systems & Employee Portals (MyHR, HRConnect)",                              # topic 13
    "State-Specific Leave Policies (Colorado, Washington)",                         # topic 14
    "Department Transfer & Payroll Verification",                                   # topic 15
    "CRM System & Call Logging Issues",                                             # topic 16
    "Retirement Procedures & Kaiser Permanente Retirement Center",                  # topic 17
    "Tax Withholding & Payroll Deductions",                                         # topic 18
    "Overpayment & Repayment Processes",                                            # topic 19
    "Wage Increase & Record Verification",                                          # topic 20
    "Document Receipt & Processing Timeframes",                                     # topic 21
    "HR System Delegation & User Access",                                           # topic 22
    "Flexible Spending Accounts (FSA/DCSA) Issues",                                 # topic 23
    "Wage Loss Verification (WLV) Letters",                                         # topic 24
    "Paycheck Deductions & Tax Coordination",                                       # topic 25
    "Mercer Insurance & Benefit Deductions",                                        # topic 26
    "Communication & Call Connection Issues",                                       # topic 27
    "Delta Dental Coverage Questions",                                              # topic 28
    "MetLife Disability Insurance & Claim Processing",                              # topic 29
    "Tuition Reimbursement & Education Benefits",                                   # topic 30
    "Employee Personal Information Updates (Name, Address)",                        # topic 31
    "Spousal Insurance Surcharges & Billing",                                       # topic 32
    "Training Application (TRA) Status & Withdrawals",                              # topic 33
    "Missing or Insufficient Information Provided",                                 # topic 34
    "Dental Procedure & Coverage Queries",                                          # topic 35
    "Supervisor Escalation & Callback Requests",                                    # topic 36
    "ADP Paystub Retrieval & Documentation",                                        # topic 37
    "Licensing, Certification & Registration (LCR Compliance)",                     # topic 38
    "Workers' Compensation & Injury Reporting",                                     # topic 39
    "Family Leave (CFRA) Expansion Queries",                                        # topic 40
    "Bereavement Leave Policies",                                                   # topic 41
    "Spousal Surcharge & Letter Clarification",                                     # topic 42
    "IT Account & Password Issues",                                                 # topic 43
    "Bereavement Policy Clarification",                                             # topic 44
    "Workers' Compensation Claim Submission (Sedgwick)",                            # topic 45
    "IRS Tax Withholding & Garnishment Issues",                                     # topic 46
    "Garnishment Processes & Debt Collection",                                      # topic 47
    "Vacation Conversion & Absence Management",                                     # topic 48
    "FMLA Leave Approval & Notifications",                                          # topic 49
    "HR Delegation Permissions & Submission Issues",                                # topic 50
    "Grandchild Dependent Addition Issues",                                         # topic 51
    "Per Diem Contract & Worker Classification",                                    # topic 52
    "Death Reporting & Survivor Benefits",                                          # topic 53
    "OSHA Training & Case Escalations",                                             # topic 54
    "Extended Sick Leave (ESL) Policy Clarification",                               # topic 55
    "Callback Verification & Contractor Security Protocols",                        # topic 56
    "Unclear or Incomplete Employee Queries",                                       # topic 57
    "Performance Improvement Plans (PIP) & Feedback Processes"                      # topic 58
]


# Pre-embed the taxonomy labels
taxonomy_label_embeddings = model.encode(taxonomy_labels, normalize_embeddings=True)



load_dotenv()

# Load environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)


# Check if the index name is valid
if not PINECONE_INDEX_NAME:
    raise ValueError("PINECONE_INDEX_NAME is not set")

# Create the index if it doesn't already exist
if PINECONE_INDEX_NAME not in [idx.name for idx in pc.list_indexes()]:
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=3072,
        metric="cosine",
        spec={
            "serverless": {
                "cloud": "aws",
                "region": "us-east-1"
            }
        }
    )

# Safely get the index
index = pc.Index(PINECONE_INDEX_NAME)

# Helper: embed text with OpenAI
def embed_text(text):
    while True:
        try:
            text = text.encode("utf-8", errors="ignore").decode("utf-8").strip()
            response = client.embeddings.create(
                input=[text],
                model="text-embedding-3-large"
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error embedding text: {e}")
            print("Retrying in 3 seconds...")
            time.sleep(3)

# Helper: chunk text into token-safe pieces
def chunk_text(text: str, chunk_size: int = 3000, overlap: int = 600, SAMPLE_SIZE=10, ) -> list:
    tokens = text.split()
    chunks = []
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk = " ".join(tokens[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def embed_and_upsert_articles(namespace="knowledge-articles", batch_size=100, sample_size=10, start=0, end=None):
    
    print("Reading the articles")
    df = pd.read_csv("/Users/ishanjuneja/Desktop/KP_RAG_Model/KP_RAG/Knowledge Handling/articles_export_cleaned.csv")
    df = df.iloc[start:end]

    print(f"Processing rows from {start} to {end or 'end'}")

    print("Checking for Sample Size")
    if sample_size:
        df = df.head(sample_size)
        print(f"🔍 SAMPLE_MODE: using only first {sample_size} rows")

    all_records = []

    print("Going row by row")
    count = start
    for _, row in df.iterrows():
        article_id = str(row["Id"])
        article_number = str(row.get("KnowledgeArticleId", ""))
        title = str(row.get("Title", ""))
        print("On the following article: ", title, "count is ", count)
        count += 1
        keywords = str(row.get("HRHD_CCS_HRHD_Keywords__c", ""))
        body = str(row.get("CleanedArticleBody", ""))
        func_area = str(row.get("HRHD_CCS_HRHD_HR_Functional_Area__c", ""))
        func_subject = str(row.get("HRHD_CCS_HRHD_HR_Sub_Function__c", ""))

        def record(id_suffix, content, segment, extra_meta={}):
            return {
                "id": f"{article_id}_{id_suffix}",
                "values": embed_text(content),
                "metadata": {
                    "article_id": article_id,
                    "article_number": article_number,
                    "segment": segment,
                    "text": content,
                    "functional_area": func_area,
                    "functional_subject": func_subject,
                    **extra_meta
                }
            }

        if title:
            all_records.append(record("title", title, "title"))
        if keywords:
            all_records.append(record("keywords", keywords, "keywords"))
        for i, chunk in enumerate(chunk_text(body)):
            all_records.append(record(f"body_{i}", chunk, "body", {"chunk_index": i}))

    # print("Upserting records")
    # for i in range(0, len(all_records), batch_size):
    #     batch = all_records[i:i + batch_size]
    #     index.upsert(vectors=batch, namespace=namespace)
    #     print(f"✅ Upserted batch {i // batch_size + 1} with {len(batch)} vectors.")
    #     time.sleep(0.5)

    import json

    print("Upserting records")

    def split_batches(records, max_bytes=2 * 1024 * 1024):
        batch, size = [], 0
        for r in records:
            r_str = json.dumps(r)
            r_size = len(r_str.encode("utf-8"))
            if size + r_size > max_bytes:
                yield batch
                batch, size = [], 0
            batch.append(r)
            size += r_size
        if batch:
            yield batch

    for i, batch in enumerate(split_batches(all_records)):
        index.upsert(vectors=batch, namespace=namespace)
        print(f"✅ Upserted batch {i + 1} with {len(batch)} vectors.")
        time.sleep(0.5)

    print(f"✅ Finished embedding and upserting {len(all_records)} vectors.")


def embed_single_article(article_row, namespace="knowledge-articles"):
    """
        Re-embed a single article and upsert its vectors into Pinecone.

        Args:
            article_row (pd.Series): A row with all article columns
            embedder (OpenAIEmbeddings): embedding client
            index (PineconeIndex): Pinecone index client
            namespace (str): Pinecone namespace
    """

    article_id = str(article_row["Id"])
    article_number = str(article_row.get("ArticleNumber", ""))
    title = str(article_row.get("Title", ""))
    keywords = str(article_row.get("HRHD_CCS_HRHD_Keywords__c", ""))
    body = str(article_row.get("CleanedArticleBody", ""))

    records = []

    

    if title:
        records.append({
            "id": f"{article_id}_title",
            "values": embed_text(title),
            "metadata": {
                "article_id": article_id,
                "article_number": article_number,
                "segment": "title",
                "text": title
            }
        })

    if keywords:
        records.append({
            "id": f"{article_id}_keywords",
            "values": embed_text(keywords),
            "metadata": {
                "article_id": article_id,
                "article_number": article_number,
                "segment": "keywords",
                "text": keywords
            }
        })

    for i, chunk in enumerate(chunk_text(body)):
        records.append({
            "id": f"{article_id}_body_{i}",
            "values": embed_text(chunk),
            "metadata": {
                "article_id": article_id,
                "article_number": article_number,
                "segment": "body",
                "chunk_index": i,
                "text": chunk
            }
        })

    index.upsert(vectors=records, namespace=namespace)
    print(f"✅ Re-embedded and upserted article {article_number} with {len(records)} vectors.")

# ---- Query Pinecone ----
def search_index(query: str, top_k: int = 5, namespace="knowledge-articles",):
    print(f"Searching for: '{query}'")
    query_embedding = embed_text(query)
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        namespace=namespace,
        include_metadata=True
    )

    matches = results.get("matches", [])
    if not matches:
        print("No results found.")
        return

    print(f"\nTop {len(matches)} Results:\n")
    for match in matches:
        metadata = match["metadata"]
        score = match["score"]
        title = metadata.get("title", "No Title")
        content = metadata.get("text", "")[:300] + "..."  # Shorten preview
        print(f"Title: {title}\nScore: {score:.4f}\nContent Preview: {content}\n---")

def rag_chat(query: str, chat_history=None, top_k: int = 5, namespace="knowledge-articles", model="gpt-3.5-turbo"):
    """
    RAG chatbot function: retrieves context from Pinecone and answers using OpenAI Chat API.
    """
    if chat_history is None:
        chat_history = []

    # Embed the user query
    query_embedding = embed_text(query)

    # Search Pinecone for relevant context
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        namespace=namespace,
        include_metadata=True
    )

    matches = results.get("matches", [])
    if not matches:
        return "🤖 Sorry, I couldn't find any relevant information to answer that."

    # Combine top results into one context block
    context_blocks = [match["metadata"]["text"] for match in matches if "text" in match["metadata"]]
    context = "\n\n---\n\n".join(context_blocks)

    # System prompt for RAG behavior
    system_prompt = (
        "You are a helpful assistant answering user questions using internal company knowledge base articles.\n"
        "Use only the context provided below. If the answer isn't in the context, respond with 'I don't know.'\n\n"
        "Please provide a detailed and accurate answer to the user's query.\n"
        f"Context:\n{context}"
    )

    # Construct messages for Chat API
    messages = [
        {"role": "system", "content": system_prompt}
    ] + chat_history + [
        {"role": "user", "content": query}
    ]
    # Call OpenAI Chat Completion API using langchain_openai's ChatOpenAI
    llm = ChatOpenAI(model=model, temperature=0.2)
    response = llm.invoke(messages)
    return str(response.content), context

def parallel_query_analysis_runner(
    query_list_path: str,
    namespace: str = "knowledge-articles",
    top_k: int = 5,
    relevance_threshold: float = 0.75,
    start: int = 0,
    end: int = None,
    max_workers: int = 5
):
    """
    Runs query analysis in parallel using threads.

    Parameters:
    - query_list_path: Path to CSV with 'Query', 'Topics', 'Sub-Topics'
    - namespace: Pinecone namespace
    - top_k: Top K results from Pinecone
    - relevance_threshold: Cosine similarity threshold
    - start, end: Slice range for queries
    - max_workers: Number of threads to use

    Saves results to: query_analysis_results.csv
    """
    file_path = "query_analysis_results.csv"
    queries_df = pd.read_csv(query_list_path).iloc[start:end]
    results = []

    # For ID generation
    start_index = 0
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        existing_df = pd.read_csv(file_path)
        start_index = len(existing_df)

    print(f"🚀 Starting parallel analysis for queries {start} to {end or 'end'} with {max_workers} workers...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i, row in queries_df.iterrows():
            query_id = start_index + i + 1
            futures.append(
                executor.submit(
                    preprocess_query_row,
                    query_id,
                    row,
                    top_k,
                    namespace,
                    relevance_threshold
                )
            )

        for future in as_completed(futures):
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as e:
                print(f"❌ Error during processing: {e}")

    if results:
        results_df = pd.DataFrame(results)
        if os.path.exists(file_path):
            results_df.to_csv(file_path, mode='a', index=False, header=False)
        else:
            results_df.to_csv(file_path, mode='w', index=False, header=True)
        print(f"✅ Saved {len(results)} results to {file_path}")
    else:
        print("⚠️ No results to save.")


# This is the function that will be used to preprocess the query row
def preprocess_query_row(query_id, row, count, top_k=5, namespace="knowledge-articles", relevance_threshold=0.75):
    """
    Preprocess a query row to extract the query, topics, and sub-topics.
    """
    # Pulling values from spreadsheet
    user_query = str(row['Query'])
    preassigned_topic_label = str(row['Topics'])
    preassigned_subtopic_label = str(row['Sub-Topics'])
    print(f"Processing Query ID: {query_id} | Query: {user_query} | Count: {count}")

    query_embedding = embed_text(user_query)

    pinecone_results = index.query(
        vector=query_embedding,
        top_k=top_k,
        namespace=namespace,
        include_metadata=True
    )

    matches = pinecone_results.get("matches", [])
    top_score = matches[0]['score'] if matches else 0.0
    avg_score = sum(m['score'] for m in matches) / len(matches) if matches else 0.0
    top_match_metadata = matches[0]['metadata'] if matches else {}
    top_match_article_id = top_match_metadata.get("article_id", "")
    top_match_segment_type = top_match_metadata.get("segment", "")

    retrieved_article_ids = list({m['metadata'].get('article_id', '') for m in matches})
    retrieved_segments = [{
        "segment": m['metadata'].get("segment", ""),
        "score": m['score']
    } for m in matches]
    print("Retrieved segments: ", retrieved_segments)

    # Ensure user_query is a string (fixes type error if it's a pandas Series)
    user_query_str = str(user_query)
    answer, context = rag_chat(query=user_query_str, chat_history=[], top_k=top_k, namespace=namespace)

    answer_provided = answer.strip().lower() not in ["i don't know", "i do not know"]

    # Classify query quality
    print("EVALUATING QUERY QUALITY")
    query_quality = classify_query_quality(user_query)
    # with our query quality, we can diagnose if the query is missing from the knowledge base or unclear
    # we will pass it to the query diagnosis function

    # we also need to score our response to the query
    my_score = 0.0
    my_label = "NO"
    print("EVALUATING ANSWER QUALITY")
    if query_quality == "vague_query" or query_quality == "unstructured_query" or query_quality == "well_structured_query":
        my_tuple = evaluate_answer_quality(user_query, context, answer)
        print("MY TUPLE IS: THIS IS WHERE WE ARE MESSING UP ", my_tuple)
        # Split and parse correctly
        score_str, label = my_tuple.split(",")
        my_score = float(score_str.strip())
        my_label = label.strip()

        # Generate KB recommendation
        kb_recommendation = generate_kb_recommendation(user_query, answer, top_match_metadata)
        print("kb_recommendation is ", kb_recommendation)

    # Diagnose query
    query_diagnosis = diagnose_query(user_query, top_score, avg_score, my_score, retrieved_segments, query_quality)

    matched_score_threshold = top_score >= relevance_threshold

    # Classify article topic
    topic_result = classify_article_topic(context)

    # Classify query topic
    query_topic_result = classify_article_topic(user_query)

    result_row = {
        "query_id": query_id,
        "user_query": user_query,
        "top_score": top_score,
        "avg_score": avg_score,
        "top_match_article_id": top_match_article_id,
        "top_match_segment_type": top_match_segment_type,
        "retrieved_segments": retrieved_segments,
        "retrieved_article_ids": retrieved_article_ids,
        "matched_score_threshold": matched_score_threshold,
        "answer_provided": answer_provided,
        "handled" : answer_provided and query_diagnosis == "resolved",
        "llm_answer": answer,
        "query_quality": query_quality, # the quality of the query (general_conversation, vague_query, unstructured_query, well_structured_query)
        "query_diagnosis": query_diagnosis, # the diagnosis of the query (kb_missing, unclear_query, both_possible, hallucination, uncertain, resolved)
        "answer_score": my_score, # the score of the answer
        "answer_label": my_label, # the label of the answer
        "kb_recommendation": kb_recommendation, # the recommendation for the KB
        "answer_subtopic_label": topic_result["subtopic_label"],
        "answer_subtopic_score": topic_result["subtopic_score"],
        "answer_parent_topic_label": topic_result["parent_topic_label"],
        "query_subtopic_label": query_topic_result["subtopic_label"],
        "query_subtopic_score": query_topic_result["subtopic_score"],
        "query_parent_topic_label": query_topic_result["parent_topic_label"],
        "preassigned_topic_label": preassigned_topic_label,
        "preassigned_subtopic_label": preassigned_subtopic_label,
        "retrieved_context_word_count": len(context.split())
    }
    return result_row

# This is the main function that will be used to run the query analysis and collect diagnostic data
def run_query_analysis(
    query_list_path: str,
    namespace: str,
    top_k: int = 5,
    relevance_threshold: float = 0.75,
    start: int = 0,
    end: int = None
) -> list[dict[str, Any]]:
    """
    Run RAG query analysis for a list of queries and collect diagnostic data.

    Parameters:
    - query_list_path: Path to CSV containing queries (with columns 'Query', 'Topics', 'Sub-Topics')
    - namespace: Pinecone namespace to search
    - top_k: Number of top results to retrieve from Pinecone
    - relevance_threshold: Cosine score threshold for high relevance

    Returns:
    - List of dictionaries with detailed diagnostics per query
    """
    results = []
    queries_df = pd.read_csv(query_list_path)

    count = 0
    queries_df = queries_df.iloc[start:end]

    # Get the start index of the existing file to generate the query_id if it doesn't exist
    file_path = "query_analysis_results.csv"
    start_index = 0
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        existing_df = pd.read_csv(file_path)
        start_index = len(existing_df)
    else:
        start_index = 0

    print("Starting query analysis")
    for i, row in queries_df.iterrows():
        for attempt in range(5):
            try:
                query_id = getattr(row, 'query_id', start_index + count + i)  # Use existing query_id if present, otherwise fallback to row index
                # Pulling values from spreadsheet
                user_query = str(row['Query'])
                preassigned_topic_label = str(row['Topics'])
                preassigned_subtopic_label = str(row['Sub-Topics'])
                print(f"Processing Query ID: {query_id} | Query: {user_query} | Count: {count}")

                query_embedding = embed_text(user_query)

                pinecone_results = index.query(
                    vector=query_embedding,
                    top_k=top_k,
                    namespace=namespace,
                    include_metadata=True
                )

                matches = pinecone_results.get("matches", [])
                top_score = float(matches[0]['score']) if matches else 0.0
                avg_score = float(sum(m['score'] for m in matches) / len(matches)) if matches else 0.0
                top_match_metadata = matches[0]['metadata'] if matches else {}
                top_match_article_id = top_match_metadata.get("article_id", "")
                top_match_segment_type = top_match_metadata.get("segment", "")

                retrieved_article_ids = list({m['metadata'].get('article_id', '') for m in matches})
                retrieved_segments = [{
                    "segment": m['metadata'].get("segment", ""),
                    "score": m['score']
                } for m in matches]
                print("Retrieved segments: ", retrieved_segments)

                # Ensure user_query is a string (fixes type error if it's a pandas Series)
                user_query_str = str(user_query)
                answer, context = rag_chat(query=user_query_str, chat_history=[], top_k=top_k, namespace=namespace)

                answer_provided = answer.strip().lower() not in ["i don't know", "i do not know"]

                # Classify query quality
                print("EVALUATING QUERY QUALITY")
                query_quality = classify_query_quality(user_query)
                # with our query quality, we can diagnose if the query is missing from the knowledge base or unclear
                # we will pass it to the query diagnosis function

                # we also need to score our response to the query
                my_score = 0.0
                my_label = "NO"
                print("EVALUATING ANSWER QUALITY")
                if query_quality == "vague_query" or query_quality == "unstructured_query" or query_quality == "well_structured_query":
                    my_tuple = evaluate_answer_quality(user_query, context, answer)
                    print("MY TUPLE IS: THIS IS WHERE WE ARE MESSING UP ", my_tuple)
                    # Split and parse correctly
                    score_str, label = my_tuple.split(",")
                    my_score = float(score_str.strip())
                    my_label = label.strip()

                    # Generate KB recommendation
                    kb_recommendation = generate_kb_recommendation(user_query, answer, top_match_metadata)
                    print("kb_recommendation is ", kb_recommendation)

                # Diagnose query
                query_diagnosis = diagnose_query(user_query, top_score, avg_score, my_score, retrieved_segments, query_quality)

                matched_score_threshold = top_score >= relevance_threshold

                # Classify article topic
                topic_result = classify_article_topic(context)

                # Classify query topic
                query_topic_result = classify_article_topic(user_query)

                result_row = {
                    "query_id": query_id,
                    "user_query": user_query,
                    "top_score": top_score,
                    "avg_score": avg_score,
                    "top_match_article_id": top_match_article_id,
                    "top_match_segment_type": top_match_segment_type,
                    "retrieved_segments": retrieved_segments,
                    "retrieved_article_ids": retrieved_article_ids,
                    "matched_score_threshold": matched_score_threshold,
                    "answer_provided": answer_provided,
                    "handled" : answer_provided and query_diagnosis == "resolved",
                    "llm_answer": answer,
                    "query_quality": query_quality, # the quality of the query (general_conversation, vague_query, unstructured_query, well_structured_query)
                    "query_diagnosis": query_diagnosis, # the diagnosis of the query (kb_missing, unclear_query, both_possible, hallucination, uncertain, resolved)
                    "answer_score": my_score, # the score of the answer
                    "answer_label": my_label, # the label of the answer
                    "kb_recommendation": kb_recommendation, # the recommendation for the KB
                    "answer_subtopic_label": topic_result["subtopic_label"],
                    "answer_subtopic_score": topic_result["subtopic_score"],
                    "answer_parent_topic_label": topic_result["parent_topic_label"],
                    "query_subtopic_label": query_topic_result["subtopic_label"],
                    "query_subtopic_score": query_topic_result["subtopic_score"],
                    "query_parent_topic_label": query_topic_result["parent_topic_label"],
                    "preassigned_topic_label": preassigned_topic_label,
                    "preassigned_subtopic_label": preassigned_subtopic_label,
                    "retrieved_context_word_count": len(context.split())
                }

                results.append(result_row)
                count += 1
                # time.sleep(0.3)
                break
            except Exception as e:
                print(f"⚠️ Retry {attempt+1}/5 failed for Query ID {query_id}: {e}")
                time.sleep(5)

    # Save results to CSV
    results_df = pd.DataFrame(results)
    column_order = [
        "query_id",
        "user_query",
        "top_score",
        "avg_score",
        "top_match_article_id",
        "top_match_segment_type",
        "retrieved_segments",
        "retrieved_article_ids",
        "matched_score_threshold",
        "answer_provided",
        "handled",
        "llm_answer",
        "query_quality",
        "query_diagnosis",
        "answer_score",
        "answer_label",
        "kb_recommendation",
        "answer_subtopic_label",
        "answer_subtopic_score",
        "answer_parent_topic_label",
        "query_subtopic_label",
        "query_subtopic_score",
        "query_parent_topic_label",
        "preassigned_topic_label",
        "preassigned_subtopic_label",
        "retrieved_context_word_count"
    ]

    # Reorder columns and ensure consistent output format
    results_df = results_df[column_order]
    if os.path.exists(file_path):
        results_df.to_csv(file_path, mode='a', index=False, header=False)
    else:
        results_df.to_csv(file_path, mode='w', index=False, header=True)

    print(f"✅ Saved results to query_analysis_results.csv")
    print(results_df)
    return results

# This is the function that will be used to classify the topic of an article
def classify_article_topic(article_text: str) -> dict:
    article_embedding = model.encode(article_text, normalize_embeddings=True)

    # Compute cosine similarity to all 59 topics
    similarities = [
        (i, taxonomy_labels[i], np.dot(article_embedding, taxonomy_label_embeddings[i]))
        for i in range(len(taxonomy_labels))
    ]

    # Sort and take top match
    best_idx, best_label, best_score = max(similarities, key=lambda x: x[2])
    parent_topic = subtopic_to_parent_map[best_idx]

    return {
        "subtopic_index": best_idx,
        "subtopic_label": best_label,
        "subtopic_score": best_score,
        "parent_topic_label": parent_topic
    }


# Classifying our query quality

def classify_query_quality(query: str) -> str:
    """
    Uses GPT to classify the quality of a user query.
    Possible labels:
    - general_conversation
    - vague_query
    - unstructured_query
    - well_structured_query
    """
    system_prompt = (
        "You are a system that classifies the quality of user queries for a knowledge base chatbot.\n"
        "Choose exactly one label from the following list:\n"
        "- general_conversation: For chit-chat like 'hi', 'how are you?'\n"
        "- vague_query: Topic is implied but not clear, e.g., 'my pay stub?'\n"
        "- unstructured_query: Relevant terms but no clear sentence, e.g., 'vacation wrong accrual again why'\n"
        "- well_structured_query: A clear, proper question, e.g., 'Why was my sick leave reduced in April?'\n\n"
        "Only return the label."
    )

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Query: {query}"}
        ],
        temperature=0,
    )
    my_response = str(response.choices[0].message.content).strip()
    return my_response

def diagnose_query(
    query: str,
    top_score,
    avg_score,
    answer_score,
    retrieved_segments: list,
    query_quality: str
) -> str:
    """
    Diagnose why a query might have failed to return a good result.
    Returns one of: kb_missing, unclear_query, both_possible, hallucination, uncertain, resolved
    """
    top_score = float(top_score)
    avg_score = float(avg_score)
    answer_score = float(answer_score)
    
    # If query is a general conversation, it's unclear
    if query_quality == "general_conversation":
        return "unclear_query"

    # If we retrieved nothing at all
    if not retrieved_segments or top_score < 0.2:
        return "kb_missing"

    segment_types = [seg.get("segment") for seg in retrieved_segments if "segment" in seg]

    # Low match and weak query
    if top_score < 0.35:
        if query_quality in ["vague_query", "unstructured_query"]:
            return "unclear_query"
        return "kb_missing"

    # Mid match (0.35–0.5)
    elif top_score < 0.5:
        if query_quality in ["vague_query", "unstructured_query"]:
            return "unclear_query"
        if not any(seg in ["body", "title", "keywords"] for seg in segment_types):
            return "kb_missing"
        if answer_score < 0.5:
            return "kb_missing"
        return "resolved"

    # High match (>= 0.5)
    elif top_score >= 0.5:
        if answer_score < 0.5:
            return "hallucination"
        return "resolved"

    return "uncertain"

def evaluate_answer_quality(query, context, answer, model="gpt-3.5-turbo") -> str:
    prompt = f"""
You are evaluating the quality of an answer produced by an AI based on knowledge base context.
Rate the following answer's quality from 0 to 1 based on how well it uses the provided context to answer the query.

- 1.0 = perfect answer, fully supported by context
- 0.5 = partially relevant, vague, or only somewhat grounded
- 0.0 = completely incorrect or unsupported by context

Then, based on the score, provide a label:
- YES (score > 0.75)
- PARTIAL (0.4–0.75)
- NO (score < 0.4)


Query: {query}

Context:
{context}

Answer:
{answer}

Return your result as a tuple in this format:
<float>,<YES/PARTIAL/NO>
"""
    response = openai.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return str(response.choices[0].message.content).strip().upper()
def generate_kb_recommendation(
    query: str,
    answer: str,
    top_match_metadata: dict,
    namespace: str = "knowledge-articles",
    model: str = "gpt-3.5-turbo"
) -> str:
    """
    Suggest KB article improvements based on a failed AI answer and article content.
    """

    article_id = top_match_metadata.get("article_id")
    matched_paragraph = top_match_metadata.get("text", "")

    if not article_id:
        return "⚠️ Cannot generate recommendation without article_id."

    # --- Query Pinecone for related article chunks ---
    related_vectors = index.query(
        vector=embed_text(query),
        namespace=namespace,
        top_k=50,
        include_metadata=True,
        filter={"article_id": article_id}
    )

    # Extract body chunks (sorted by original order if chunk_index exists)
    body_chunks = [
        (m["metadata"].get("chunk_index", i), m["metadata"]["text"])
        for i, m in enumerate(related_vectors.get("matches", []))
        if m["metadata"].get("segment") == "body"
    ]
    body_chunks.sort(key=lambda x: x[0])  # Sort by chunk index
    top_body = "\n\n".join(chunk for _, chunk in body_chunks[:5])  # Limit to first 5 chunks

    # Fallback if body is missing or small
    if len(top_body) < 100:
        try:
            import pandas as pd
            df = pd.read_csv("articles_export_cleaned.csv")
            full_body = df.loc[df["Id"].astype(str) == str(article_id), "CleanedArticleBody"].values[0]
        except Exception:
            full_body = ""
    else:
        full_body = top_body

    # --- Compose prompt ---
    system_prompt = (
        "You are an AI editor improving internal knowledge base articles.\n"
        "You are given:\n"
        "1. A user question\n"
        "2. The AI's previous answer\n"
        "3. The current content of the article\n\n"
        "Suggest exactly what should be added, reworded, or clarified to better support this query.\n"
        "Respond with specific sentence edits, new content, and specific phrases to add."
    )

    user_prompt = f"""
        Query:
        {query}

        AI's Answer:
        {answer}

        Matched Paragraph:
        {matched_paragraph}

        Current Article Content:
        {full_body}

        ---

        What exact edits and additions (e.g., revised sentences, new explanations, or specific terms) would help this article answer the user query more clearly and improve retrieval accuracy?

        Please respond using the following structured format:

        Content Edits:
        <Clearly list any reworded or revised sentences that would improve the article.>

        Additional Lines:
        <New lines or sections that should be added to address the query more fully.>

        Clarification:
        <Areas that should be clarified, expanded upon, or made more specific.>

        Suggested Keywords:
        <Provide 5–10 relevant keywords or phrases that should be included to strengthen search relevance.>
        """

    # LLM call
    llm = ChatOpenAI(model=model, temperature=0.3)
    response = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ])

    return str(response.content).strip()
