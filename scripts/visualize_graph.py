"""Visualisation du graphe LangGraph (Mermaid + PNG)."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import OUTPUT_DIR
from src.graph import build_graph


def visualize(output_dir: Path | None = None):
    """Exporte le graphe en Mermaid et PNG."""
    out = output_dir or OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    app = build_graph()
    graph = app.get_graph()

    mermaid = graph.draw_mermaid()
    mermaid_path = out / "graph.mmd"
    mermaid_path.write_text(mermaid, encoding="utf-8")
    print(f"Mermaid exporté : {mermaid_path}")

    try:
        png_path = out / "graph.png"
        graph.draw_mermaid_png(output_file_path=str(png_path))
        print(f"PNG exporté : {png_path}")
    except Exception as e:
        print(f"PNG non généré ({e}). Utilisez graph.mmd sur https://mermaid.live")

    print("\n--- Structure du graphe ---")
    print(mermaid)


if __name__ == "__main__":
    visualize()
