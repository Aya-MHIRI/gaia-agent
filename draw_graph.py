from agent_langGraph import app

graph = app.get_graph()

mermaid = graph.draw_mermaid()
print(mermaid)
open("graph.mmd", "w", encoding="utf-8").write(mermaid)

try:
    open("graph.png", "wb").write(graph.draw_mermaid_png())
    print("graph.png créé")
except Exception as e:
    print("PNG impossible :", e)