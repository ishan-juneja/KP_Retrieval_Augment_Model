# KP RAG Model - Knowledge Base Analytics & Retrieval System

A comprehensive Retrieval-Augmented Generation (RAG) system for analyzing and querying knowledge base articles with interactive dashboards for insights and performance monitoring.

## 📋 Project Overview

This system provides:
- **Knowledge Base Integration**: Pulls articles from Salesforce Knowledge
- **RAG Implementation**: Uses OpenAI embeddings and Pinecone vector database
- **Query Analysis**: Analyzes query performance and provides insights
- **Interactive Dashboards**: Multiple Streamlit dashboards for different use cases
- **Data Processing**: Automated cleaning and embedding of knowledge articles

## 🏗️ Project Structure

```
KP_RAG_Model/
├── KP_RAG/
│   ├── dashboard/                    # Main analytics dashboard
│   │   └── app.py                   # Knowledge base analytics dashboard
│   ├── dashboard_2/                 # HR-specific dashboard
│   │   ├── app.py                   # HR query analytics dashboard
│   │   └── sample_data.csv          # Sample HR conversation data
│   ├── Knowledge Handling/          # Core RAG functionality
│   │   ├── main.py                  # Main execution script
│   │   ├── embedPinecone.py         # Pinecone embedding and RAG logic
│   │   ├── pullArticles.py          # Salesforce article extraction
│   │   ├── cleanArticles.py         # HTML cleaning utilities
│   │   ├── analysis_dashboard.py    # Query analysis dashboard
│   │   ├── articles_export_cleaned.csv  # Cleaned knowledge articles
│   │   ├── KP_AA_Unhandled_Queries_Sum.csv  # Query dataset
│   │   └── query_test.csv           # Test queries
│   ├── InstrumentSans.ttf           # Custom font for dashboards
│   ├── python_backup.py             # Backup utility script
│   └── query_analysis_results.csv   # Analysis results output
└── README.md                        # This file
```

## 🚀 Setup Instructions

### Prerequisites

- Python 3.8+
- OpenAI API key
- Pinecone API key
- Salesforce credentials (for article extraction)

### 1. Environment Setup

Create a `.env` file in the `KP_RAG/` directory with the following variables:

```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Pinecone Configuration
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX_NAME=your_index_name_here

# Salesforce Configuration (for article extraction)
SF_USERNAME=your_salesforce_username
SF_PASSWORD=your_salesforce_password
SF_SECURITY_TOKEN=your_salesforce_security_token
SF_DOMAIN=login  # or 'test' for sandbox
```

### 2. Install Dependencies

```bash
pip install streamlit pandas numpy plotly wordcloud pillow
pip install openai langchain-openai pinecone-client
pip install simple-salesforce beautifulsoup4 python-dotenv
pip install matplotlib altair statsmodels
```

### 3. File Placement Requirements

**Critical**: Ensure the following files are in the correct locations:

- `InstrumentSans.ttf` must be in the `KP_RAG/` directory (for dashboard fonts)
- All CSV data files should be in `KP_RAG/Knowledge Handling/`
- Environment variables file (`.env`) must be in `KP_RAG/`

## 🔧 Core Components

### 1. Knowledge Article Processing (`Knowledge Handling/`)

#### Article Extraction (`pullArticles.py`)
- Connects to Salesforce Knowledge
- Extracts article metadata and content
- Handles authentication and error management
- Outputs raw article data to CSV

#### Article Cleaning (`cleanArticles.py`)
- Removes HTML markup from article bodies
- Preserves formatting (bullet points, line breaks)
- Creates clean, readable text for embedding

#### Embedding & Vector Storage (`embedPinecone.py`)
- Converts articles to OpenAI embeddings
- Stores vectors in Pinecone database
- Implements chunking for long articles
- Provides search and RAG functionality

### 2. Query Analysis System

#### Main Analysis (`embedPinecone.py`)
- `run_query_analysis()`: Analyzes query performance
- `parallel_query_analysis_runner()`: Batch processing for large datasets
- `classify_query_quality()`: Categorizes query types
- `diagnose_query()`: Provides failure diagnosis
- `evaluate_answer_quality()`: Assesses response quality

#### Analysis Features:
- **Query Quality Classification**: Categorizes queries by complexity
- **Performance Metrics**: Top scores, relevance thresholds, answer rates
- **Failure Diagnosis**: Identifies why queries fail
- **Knowledge Base Recommendations**: Suggests article improvements

### 3. Interactive Dashboards

#### Main Analytics Dashboard (`dashboard/app.py`)
**Features:**
- Query performance overview with key metrics
- Score distribution visualizations
- Article-level performance analysis
- Filtering by date ranges and categories
- Export capabilities for analysis results

**Key Metrics Displayed:**
- Average top scores and relevance rates
- Answer provision rates
- Query handling success rates
- Performance trends over time

#### HR Analytics Dashboard (`dashboard_2/app.py`)
**Features:**
- HR-specific query analytics
- Category volume vs error rate analysis
- Agent workload visualization
- Temporal error heatmaps
- Conversation length analysis
- Positive feedback tracking

**Visualizations:**
- Treemaps for sub-topic confusion
- Scatter plots for agent performance
- Heatmaps for temporal patterns
- Sunburst charts for topic relationships

#### Analysis Dashboard (`Knowledge Handling/analysis_dashboard.py`)
**Features:**
- Baseline performance metrics
- Comparative analysis across query types
- Article-level failure analysis
- Dynamic filtering and drill-down capabilities

## 📊 Data Flow

1. **Article Extraction**: Salesforce → Raw CSV
2. **Article Cleaning**: HTML removal → Cleaned CSV
3. **Embedding**: Text → OpenAI embeddings → Pinecone
4. **Query Analysis**: User queries → RAG responses → Performance metrics
5. **Dashboard Visualization**: Metrics → Interactive charts and insights

## 🎯 Usage Instructions

### Running the Main Analysis

```bash
cd KP_RAG/Knowledge Handling/
python main.py
```

**Available Operations in `main.py`:**
- Article cleaning (uncomment as needed)
- Article embedding and vector storage
- Query analysis on test datasets
- Parallel processing for large query sets

### Launching Dashboards

#### Main Analytics Dashboard:
```bash
cd KP_RAG/dashboard/
streamlit run app.py
```

#### HR Analytics Dashboard:
```bash
cd KP_RAG/dashboard_2/
streamlit run app.py
```

#### Analysis Dashboard:
```bash
cd KP_RAG/Knowledge Handling/
streamlit run analysis_dashboard.py
```

### Key Functions

#### Embedding Articles:
```python
from embedPinecone import embed_and_upsert_articles

# Embed all articles
embed_and_upsert_articles(namespace="knowledge-articles", batch_size=100)

# Embed with sample size for testing
embed_and_upsert_articles(namespace="knowledge-articles", sample_size=10)
```

#### Running Query Analysis:
```python
from embedPinecone import parallel_query_analysis_runner

# Analyze queries in parallel
results = parallel_query_analysis_runner(
    query_list_path="KP_AA_Unhandled_Queries_Sum.csv",
    namespace="knowledge-articles",
    start=0  # Starting index for batch processing
)
```

#### RAG Chat Interface:
```python
from embedPinecone import rag_chat

# Get RAG response
response = rag_chat("What is the leave accrual policy?", namespace="knowledge-articles")
print(response)
```

## 📈 Dashboard Features

### Main Analytics Dashboard
- **Performance Overview**: Key metrics at a glance
- **Score Analysis**: Distribution and trends
- **Article Performance**: Success rates by article
- **Query Insights**: Detailed query analysis
- **Export Functionality**: Download results

### HR Analytics Dashboard
- **Category Analysis**: Volume vs error rates
- **Agent Performance**: Workload and success metrics
- **Temporal Patterns**: Time-based error analysis
- **Topic Relationships**: Hierarchical topic mapping
- **Feedback Tracking**: Positive response rates

### Analysis Dashboard
- **Baseline Metrics**: Core performance indicators
- **Comparative Analysis**: Cross-category insights
- **Failure Diagnosis**: Root cause analysis
- **Article Recommendations**: Improvement suggestions

## 🔍 Troubleshooting

### Common Issues:

1. **Font Loading Errors**: Ensure `InstrumentSans.ttf` is in the correct directory
2. **API Key Errors**: Verify all environment variables are set correctly
3. **CSV Path Errors**: Check file paths in the `Knowledge Handling/` directory
4. **Pinecone Index Issues**: Ensure index name matches environment variable

### Performance Optimization:

- Use `sample_size` parameter for testing with smaller datasets
- Implement batch processing for large query sets
- Monitor API rate limits for OpenAI and Pinecone
- Use parallel processing for query analysis

## 📝 Configuration Options

### Embedding Parameters:
- `chunk_size`: Text chunk size for embedding (default: 3000)
- `overlap`: Overlap between chunks (default: 600)
- `batch_size`: Pinecone upsert batch size (default: 100)

### Analysis Parameters:
- `relevance_threshold`: Minimum score for relevance (default: 0.75)
- `top_k`: Number of results to retrieve (default: 5)
- `model`: OpenAI model for analysis (default: "gpt-3.5-turbo")

## 🤝 Contributing

When adding new features:
1. Follow existing code structure and naming conventions
2. Update this README with new functionality
3. Test with sample data before production use
4. Document any new environment variables or dependencies

## 📄 License

This project is proprietary to Kaiser Permanente. Please ensure compliance with organizational policies when using or modifying this code. 