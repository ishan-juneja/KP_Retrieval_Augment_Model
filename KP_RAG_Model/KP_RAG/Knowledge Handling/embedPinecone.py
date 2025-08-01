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
    text = text.encode("utf-8", errors="ignore").decode("utf-8").strip()
    response = client.embeddings.create(
        input=[text],
        model="text-embedding-3-large"
    )
    return response.data[0].embedding

# Helper: chunk text into token-safe pieces
def chunk_text(text: str, chunk_size: int = 3000, overlap: int = 600, SAMPLE_SIZE=10, ) -> list:
    tokens = text.split()
    chunks = []
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk = " ".join(tokens[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def embed_and_upsert_articles(namespace="knowledge-articles", batch_size=100, sample_size=10):
    print("Reading the articles")
    df = pd.read_csv("articles_export_cleaned.csv")

    print("Checking for Sample Size")
    if sample_size:
        df = df.head(sample_size)
        print(f"🔍 SAMPLE_MODE: using only first {sample_size} rows")

    all_records = []

    print("Going row by row")
    for _, row in df.iterrows():
        article_id = str(row["Id"])
        article_number = str(row.get("KnowledgeArticleId", ""))
        title = str(row.get("Title", ""))
        print("On the following article: ", title)
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

    print("Upserting records")
    for i in range(0, len(all_records), batch_size):
        batch = all_records[i:i + batch_size]
        index.upsert(vectors=batch, namespace=namespace)
        print(f"✅ Upserted batch {i // batch_size + 1} with {len(batch)} vectors.")
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

def run_query_analysis(
    query_list_path: str,
    namespace: str,
    top_k: int = 5,
    relevance_threshold: float = 0.75
) -> list[dict[str, Any]]:
    """
    Run RAG query analysis for a list of queries and collect diagnostic data.

    Parameters:
    - query_list_path: Path to CSV containing queries (with columns 'query_id', 'user_query')
    - namespace: Pinecone namespace to search
    - top_k: Number of top results to retrieve from Pinecone
    - relevance_threshold: Cosine score threshold for high relevance

    Returns:
    - List of dictionaries with detailed diagnostics per query
    """
    results = []
    queries_df = pd.read_csv(query_list_path)

    for _, row in queries_df.iterrows():
        query_id = row['query_id']
        user_query = str(row['user_query'])
        print(f"Processing Query ID: {query_id} | Query: {user_query}")

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

        # Ensure user_query is a string (fixes type error if it's a pandas Series)
        user_query_str = str(user_query)
        answer, context = rag_chat(query=user_query_str, chat_history=[], top_k=top_k, namespace=namespace)

        answer_provided = answer.strip().lower() not in ["i don't know", "i do not know"]

        # Classify query quality
        query_quality = classify_query_quality(user_query)
        # with our query quality, we can diagnose if the query is missing from the knowledge base or unclear
        # we will pass it to the query diagnosis function

        # we also need to score our response to the query
        my_score = 0.0
        my_label = "NO"
        if query_quality == "vague_query" or query_quality == "unstructured_query" or query_quality == "well_structured_query":
            my_tuple = evaluate_answer_quality(user_query, context, answer)
            my_score = float(my_tuple[0])
            my_label = my_tuple[1]
            print("my score is ", my_score)
            print("my label is ", my_tuple)

            # Generate KB recommendation
            kb_recommendation = generate_kb_recommendation(user_query, answer, top_match_metadata)
            print("kb_recommendation is ", kb_recommendation)

        # Diagnose query
        query_diagnosis = diagnose_query(user_query, top_score, avg_score, my_score, retrieved_segments, query_quality)

        matched_score_threshold = top_score >= relevance_threshold

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
            "llm_answer": answer,
            "query_quality": query_quality, # the quality of the query (general_conversation, vague_query, unstructured_query, well_structured_query)
            "query_diagnosis": query_diagnosis, # the diagnosis of the query (kb_missing, unclear_query, both_possible, hallucination, uncertain, resolved)
            "answer_score": my_score, # the score of the answer
            "answer_label": my_label, # the label of the answer
            "kb_recommendation": kb_recommendation # the recommendation for the KB
        }

        results.append(result_row)
    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv("query_analysis_results.csv", index=False)
    print(f"✅ Saved results to query_analysis_results.csv")
    print(results_df)
    return results


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
    top_score: float,
    avg_score: float,
    answer_score: float,
    retrieved_segments: list,
    query_quality: str
) -> str:
    """
    Diagnose why a query might have failed to return a good result.
    Returns one of: kb_missing, unclear_query, both_possible, hallucination, uncertain, resolved
    """
    # Normalize and extract segment types
    segment_types = [seg.get("segment") for seg in retrieved_segments if "segment" in seg]

    # Low match and poor query
    if top_score < 0.35:
        if query_quality in ["vague_query", "unstructured_query", "general_conversation"]:
            return "unclear_query"
        elif query_quality == "well_structured_query":
            return "kb_missing"

    # Mid-score, weak segments
    if top_score < 0.5:
        if not any(seg in ["body", "title", "keywords"] for seg in segment_types):
            return "kb_missing"
        if query_quality != "well_structured_query":
            return "both_possible"

    # High match score, but weak answer quality
    if top_score >= 0.5:
        if answer_score < 0.5:
            return "hallucination"
        else:
            return "resolved"  # query was answered well and context matched

    # Mid everything, not clear
    if 0.4 <= top_score <= 0.75 and 0.4 <= answer_score <= 0.75:
        return "both_possible"

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

def generate_test_prompts_from_article(article_text: str, model="gpt-3.5-turbo") -> list[str]:
    prompt = f"""
You are a support agent creating realistic user questions based on the following knowledge base article.

Article Content:
{article_text}

Generate 10 diverse and realistic user queries someone might ask that would be answered by this article.

Return only the list of questions.
"""
    response = openai.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
    )
    return response.choices[0].message.content.strip().split("\n")