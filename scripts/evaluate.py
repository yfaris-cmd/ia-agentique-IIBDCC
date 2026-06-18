"""Script d'évaluation du système RAG agentique."""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import OUTPUT_DIR
from src.graph import build_graph
from langchain_core.messages import HumanMessage


def run_evaluation(questions_path: Path | None = None) -> dict:
    """Exécute les 20 questions et collecte métriques."""
    path = questions_path or PROJECT_ROOT / "evaluation" / "questions.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    app = build_graph()
    results = {"simple": [], "complex": [], "summary": {}}

    for category in ("simple", "complex"):
        for item in data[category]:
            qid = item["id"]
            question = item["question"]
            thread_id = f"eval-{qid}"

            start = time.perf_counter()
            try:
                state = app.invoke(
                    {
                        "messages": [HumanMessage(content=question)],
                        "question": question,
                        "retry_count": 0,
                        "sources": [],
                    },
                    {"configurable": {"thread_id": thread_id}},
                )
                elapsed = time.perf_counter() - start
                answer = state.get("generation", "")
                sources = list(dict.fromkeys(state.get("sources", [])))
                retries = state.get("retry_count", 0)
                n_docs = len(state.get("graded_documents") or state.get("documents", []))
                error = None
            except Exception as e:
                elapsed = time.perf_counter() - start
                answer = ""
                sources = []
                retries = 0
                n_docs = 0
                error = str(e)

            entry = {
                "id": qid,
                "question": question,
                "answer": answer,
                "response_time_s": round(elapsed, 2),
                "sources": sources,
                "documents_retrieved": n_docs,
                "retry_count": retries,
                "error": error,
            }
            results[category].append(entry)
            status = "OK" if not error else "ERR"
            print(f"[{status}] {qid} ({elapsed:.1f}s) — {question[:60]}...")

    simple_times = [r["response_time_s"] for r in results["simple"]]
    complex_times = [r["response_time_s"] for r in results["complex"]]
    all_results = results["simple"] + results["complex"]

    results["summary"] = {
        "timestamp": datetime.now().isoformat(),
        "total_questions": len(all_results),
        "errors": sum(1 for r in all_results if r["error"]),
        "avg_time_simple_s": round(sum(simple_times) / len(simple_times), 2),
        "avg_time_complex_s": round(sum(complex_times) / len(complex_times), 2),
        "avg_docs_retrieved": round(
            sum(r["documents_retrieved"] for r in all_results) / len(all_results), 1
        ),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUTPUT_DIR / f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nRésultats sauvegardés : {out_file}")
    print(f"Temps moyen (simple)  : {results['summary']['avg_time_simple_s']}s")
    print(f"Temps moyen (complexe): {results['summary']['avg_time_complex_s']}s")
    return results


if __name__ == "__main__":
    run_evaluation()
