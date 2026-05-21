# --------------------------------------------------
# Export graph picture
# --------------------------------------------------

from langgraph.graph import StateGraph


def export_graph_image(graph: StateGraph, filename: str):
    png_data = graph.get_graph().draw_mermaid_png()

    with open(filename, "wb") as f:
        f.write(png_data)

    print(f"Graph image saved as {filename}")