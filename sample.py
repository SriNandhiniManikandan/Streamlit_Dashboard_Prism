import json
with open("model_output.json", "r", encoding="utf-8") as f:
    content = f.read()


print(content[52760:52810])