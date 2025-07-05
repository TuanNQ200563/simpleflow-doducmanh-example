import pandas as pd


TOPIC = ["Biomedical", "Dev"]


def main() -> pd.DataFrame:
    all_data = []
    for topic in TOPIC:
        with open(f"./data/raw/{topic}.txt", "r", encoding="utf-8") as f_text, \
            open(f"./data/raw/{topic}_label.txt", "r", encoding="utf-8") as f_label:
                texts = f_text.readlines()
                labels = f_label.readlines()
                
        for text, label in zip(texts, labels):
            all_data.append({
                "text": text.strip(),
                "label": label.strip()
            })
            
    data = pd.DataFrame(all_data)
    return data