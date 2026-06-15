import json
nb = json.load(open("Local_Random_seeds_train_mmhcl_clothing_colab_original.ipynb", "r", encoding="utf-8"))
for i, c in enumerate(nb["cells"]):
    s = "".join(c.get("source", []))
    head = (s.splitlines() or ["<empty>"])[0][:100]
    print(f"{i:2d} {c['cell_type'][:4]} | {head}")
