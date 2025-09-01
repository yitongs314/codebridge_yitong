from ingest import preprocess
from rules_engine import detect_intent, extract_params
from templates import TEMPLATES
import nltk, re


def main():
    print("=== CodeBridge Rules Engine Demo ===")
    while True:
        text = input("\nPlease enter your code description (exit to terminate):\n> ")
        if text.lower() == "exit":
            print("Bye!")
            break

        # 1. preprocess
        pos_tags = preprocess(text)
        print("POS tagging result:", pos_tags)

        # 2. intent detection
        words = [w for (w, _) in pos_tags] # tokens only
        intent = detect_intent(words, text)
        print(f"Intent result: {intent}")

        # 3. param extraction
        params = extract_params(intent, pos_tags)
        print("Params:", params)

        tpl = TEMPLATES.get(intent)
        if not tpl:
            print("no template")
            continue

        try:
            code = tpl.format(**params)
        except KeyError as e:
            print(f"template does not have {e}")
            # catching all logic: set keys to ""
            for k in re.findall(r"\{(\w+)\}", tpl):
                params.setdefault(k, "")
            code = tpl.format(**params)
        
        print("=== result ===")
        print(code)


if __name__ == "__main__":
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger_eng')
    main()