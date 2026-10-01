"""=======================================================================================
                    Evaluation for advance RAG
==========================================================================================

"""
import json
import rag_corrected as rag

# load the questions (skip the ones that are not in the handbook)
questions = json.load(open("eval_questions.json", encoding="utf-8"))
questions = [q for q in questions if q["evidence"]]

correct = 0
for q in questions:
    chunks = rag.compressor_retriever.invoke(q["question"])
    text = " ".join(c.page_content for c in chunks).lower()
    text = " ".join(text.split())                      # fix line breaks
    if q["evidence"].lower() in text:
        correct += 1
    else:
        print("Missed:", q["question"])

print(f"Score: {correct}/{len(questions)} = {correct / len(questions):.0%}")