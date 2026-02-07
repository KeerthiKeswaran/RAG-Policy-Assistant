from typing import List, Dict
import pandas as pd

from src.rag_pipeline import RagPipeline
from src.vector_store import VectorStore
# No embeddings import

class Evaluator:
    def __init__(self, pipeline: RagPipeline):
        self.pipeline = pipeline
        self.evaluation_set = [
            # Fully Answerable
            {"question": "What is the return window for a refund?", "type": "Fully Answerable", "expected": "30 days (example)"},
            {"question": "How do I cancel an order before shipping?", "type": "Fully Answerable", "expected": "Contact support via email"},
            
            # Partially Answerable (might be ambiguous or split across docs)
            {"question": "Can I return a sale item if I paid with a gift card?", "type": "Partially Answerable", "expected": "Policy on sale items vs payment methods"},
            {"question": "What happens if my package is lost during shipping to an international address?", "type": "Partially Answerable", "expected": "Lost package policy + International shipping details"},

            # Unanswerable (Not in policy)
            {"question": "Do you offer corporate discounts?", "type": "Unanswerable", "expected": "Refusal"},
            {"question": "What is the CEO's email address?", "type": "Unanswerable", "expected": "Refusal"},
            {"question": "Can I pay with Bitcoin?", "type": "Unanswerable", "expected": "Refusal"},
        ]

    def run_evaluation(self):
        """Runs the evaluation set through the pipeline."""
        results = []
        print(f"Starting evaluation on {len(self.evaluation_set)} questions...")
        
        for item in self.evaluation_set:
            q = item["question"]
            print(f"Processing: {q}")
            try:
                # Direct call to retrieve to check context (returns List[str])
                docs = self.pipeline.retrieve(q, k=3)
                context_found = bool(docs)
                
                # Generation
                if context_found:
                    answer = self.pipeline.run(q)  # This re-retrieves, which is fine
                else:
                    answer = "No relevant context found (Refusal Triggered)"
                
                results.append({
                    "Question": q,
                    "Type": item["type"],
                    "Expected Content": item["expected"],
                    "Actual Answer": answer,
                    "Context Retrieved": context_found
                })
            except Exception as e:
                print(f"Error processing '{q}': {e}")
                results.append({
                    "Question": q,
                    "Type": item["type"],
                    "Expected Content": item["expected"],
                    "Actual Answer": f"ERROR: {str(e)}",
                    "Context Retrieved": False
                })

        # Save to CSV for analysis
        df = pd.DataFrame(results)
        df.to_csv("evaluation_results.csv", index=False)
        print("\nEvaluation complete. Results saved to 'evaluation_results.csv'.")
        
        # Print table for README usage
        try:
            print(df[["Question", "Actual Answer"]].to_markdown())
        except:
             print(df[["Question", "Actual Answer"]])

if __name__ == "__main__":
    # Setup dependencies
    # Vector store setup (auto-loads if needed via app logic, but here we assume it exists or init empty)
    vector_store = VectorStore(persist_directory="chroma_db", collection_name="policy_docs_v2")
    
    pipeline = RagPipeline(vector_store)
    
    evaluator = Evaluator(pipeline)
    evaluator.run_evaluation()
