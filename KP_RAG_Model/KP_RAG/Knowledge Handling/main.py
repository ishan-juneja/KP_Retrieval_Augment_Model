from cleanArticles import clean_article_body_column
from embedPinecone import embed_and_upsert_articles, search_index, rag_chat, run_query_analysis, parallel_query_analysis_runner
def main():
    # This cleans our articles' body columns to not have HTML
    # clean_article_body_column(input_csv="articles_export.csv", output_csv="articles_export_cleaned", body_column="ArticleBody")

    # This embedds a list of our articles
    # print("Embedding articles")
    # embed_and_upsert_articles(namespace="knowledge-articles", batch_size=100, sample_size=None, start=0, end=101)

    # print("Searching for articles")
    # search_index(query="Leave Accrual Audit", top_k=5, namespace="knowledge-articles")

    # print("Asking via RAG")
    # response = rag_chat("What is the leave accrual audit?", namespace="knowledge-articles")
    # print(response)

    # print("Running query analysis")
    # next batch should be 100-200

    # parallel query analysis
    analysis_results = parallel_query_analysis_runner(query_list_path="Knowledge Handling/KP_AA_Unhandled_Queries_Sum.csv", namespace="knowledge-articles", start=1556)
    print(analysis_results)
    print("Finished!")

    # analysis_results = run_query_analysis(query_list_path="Knowledge Handling/KP_AA_Unhandled_Queries_Sum.csv", namespace="knowledge-articles", start=0, end=5)
    # analysis_results = run_query_analysis(query_list_path="Knowledge Handling/KP_AA_Unhandled_Queries_Sum.csv", namespace="knowledge-articles", start=0, end=5)
    # print(analysis_results)
    # print("Finished!")
    
if __name__ == '__main__':
    main()
