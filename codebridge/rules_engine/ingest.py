import spacy
import re
import nltk
import pandas as pd

nlp = spacy.load("en_core_web_sm")

def preprocess(text: str):
    """
    Preprocess input text:
    1. Lowercase
    2. Remove special characters
    3. Tokenize
    4. (Optional) POS tagging
    """

    text = text.lower()
    
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    
    tokens = nltk.word_tokenize(text)

    pos_tags = nltk.pos_tag(tokens)

    # correct some coding POS (should be more)
    # this avoids having print as NN instead of VB
    overrides = {
        "print": "VB",
        "input": "VB",
        "import": "VB",
        "return": "VB"
    }
    pos_tags = [(word, overrides.get(word, tag)) for word, tag in pos_tags]

    return pos_tags



def load_txt_as_df(path, limit=None):
    """
    把 Kaggle 提供的 instruction-code txt 文件转成 DataFrame
    每个样本包含: {"instruction": ..., "code": ...}
    """
    with open(path, "r") as f:
        lines = f.readlines()

    data = []
    instruction, code = None, []

    for line in lines:
        line = line.strip()
        if not line: 
            if instruction and code:
                data.append({"instruction": instruction, "code": "\n".join(code)})
                instruction, code = None, []
            continue

        if line.startswith("#"):
            if instruction and code:
                data.append({"instruction": instruction, "code": "\n".join(code)})
                code = []
            instruction = line.lstrip("#").strip()
        else:
            code.append(line)


    if instruction and code:
        data.append({"instruction": instruction, "code": "\n".join(code)})

    df = pd.DataFrame(data)

    if limit:
        df = df.head(limit)

    return df