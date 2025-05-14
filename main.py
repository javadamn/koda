import time
import config 
from pipeline import MicrobialAnalysisPipeline 
from neo4j_handler import Neo4jKnowledgeGraph 

logger = config.get_logger(__name__)

def main():
    """Main execution function."""
    pipeline_instance = None
    try:
        #just to make sure Neo4j connection works before initializing the pipeline
        logger.info("Attempting initial Neo4j connection check...")
        Neo4jKnowledgeGraph.get_driver()
        logger.info("Neo4j connection check successful.")

        pipeline_instance = MicrobialAnalysisPipeline()

        #examples:
        # query1 = "Which microbes produce Butyrate with the highest flux?"
        # query2 = "Show me the interactions between Bacteroides_thetaiotaomicron and Eubacterium_rectale."
        # query3 = "What are the main producers and consumers of Acetate? Compare their abundance if available."
        # query4 = "Find pathways associated with Faecalibacterium_prausnitzii and list their importance scores."
        # query5 = "Tell me about Propionate - which microbes handle it?" # More open-ended
        query6 = "What microbes both produce and consume Thiamine? What is the net flux if possible?" 
        # query6 ="What KEGG Orthologies (KOs) are associated with the microbe Bifidobacterium_longum_longum_JDM301?"

        user_query = query6 

        print(f"\n--- Running Analysis for Query: ---\n{user_query}\n" + "-"*35)
        start_time = time.time()

        final_report_str = pipeline_instance.run_analysis(user_query)

        end_time = time.time()
        print(f"\n--- Analysis Complete (Duration: {end_time - start_time:.2f} seconds) ---")

        print("\n--- Final Report ---")
        print(final_report_str)
        print("--------------------")

        try:
            with open("analysis_report.md", "w", encoding="utf-8") as f:
                f.write(f"# Analysis Report for Query:\n\n> {user_query}\n\n")
                f.write(final_report_str)
            logger.info("Report saved to analysis_report.md")
        except IOError as e:
            logger.error(f"Error saving report to file: {e}")
        except Exception as e:
             logger.error(f"Unexpected error saving report: {e}")


    except Exception as e:
        logger.critical(f"An error occurred during setup or execution: {e}", exc_info=True) 
    finally:
        Neo4jKnowledgeGraph.close_driver()
        logger.info("Pipeline finished.")

if __name__ == "__main__":
    main()