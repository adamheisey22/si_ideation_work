import pandas as pd
import os
import re

path = r'C:\Users\qia22677\OneDrive - Intermountain Healthcare\gh-repos\si_ideation_IH\notes'

os.chdir(path)

def extract_notes(path):
    pattern = r"\d+"
    filenames = []
    texts = []
    for filename in os.listdir(path):
        if filename.endswith(".txt"):
            with open(os.path.join(path, filename), "r") as f:
                text = ",".join(line.strip() for line in f)
                filenames.append(int(re.findall(pattern, filename)[0]))
                texts.append(text)
    df = pd.DataFrame({"ID": filenames, "NOTE_TEXT": texts})
    return df

df = extract_notes(path)
df.to_csv('notes_test_1.csv', index=False)
