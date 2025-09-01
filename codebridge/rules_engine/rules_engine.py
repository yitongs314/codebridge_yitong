import re
from typing import List, Optional

DEFINE_SYNS   = {"define", "create", "write", "implement", "make", "build", "declare", "function"}
CALL_SYNS     = {"call", "invoke", "run", "execute", "use", "apply"}
LOOP_SYNS     = {"loop", "iterate", "for", "while", "traverse"}
COND_SYNS     = {"if", "else", "elif", "condition", "conditional", "when", "unless"}
IMPORT_SYNS   = {"import", "include", "from"}
ASSIGN_SYNS   = {"set", "assign", "initialize", "init", "put", "store"}
CLASS_SYNS    = {"class"}
PRINT_SYNS    = {"print", "show", "display", "log", "echo"}


def has_any(token_set, syns):
    return len(token_set.intersection(syns)) > 0


def detect_intent(tokens: List[str], clean_text: Optional[str] = None) -> str:
    ts = set(tokens)

    # import
    if has_any(ts, IMPORT_SYNS):
        return "IMPORT"

    # define class/function
    if has_any(ts, CLASS_SYNS):
        return "CLASS_DEF"
    if has_any(ts, DEFINE_SYNS):
        # 带 "function", "method" 强化判断
        if "function" in ts or "method" in ts:
            return "DEFINE_FUNCTION"
        # 句式提示：def xxx(...)
        if clean_text and re.search(r"\bdef\s+\w+\s*\(", clean_text):
            return "DEFINE_FUNCTION"
        # 没有 function 词但命令式“写/实现/创建”
        return "DEFINE_FUNCTION"

    # 3) loops
    if has_any(ts, LOOP_SYNS):
        return "LOOP"
    if has_any(ts, COND_SYNS):
        return "CONDITIONAL"

    # assign variable
    if has_any(ts, ASSIGN_SYNS):
        return "VARIABLE_ASSIGN"
    if clean_text and re.search(r"\b\w+\s*=\s*[^=]", clean_text):
        return "VARIABLE_ASSIGN"

    # call function
    if has_any(ts, CALL_SYNS):
        return "CALL_FUNCTION"
    if clean_text and re.search(r"\b\w+\s*\([^)]*\)", clean_text):
        return "CALL_FUNCTION"

    # print
    if has_any(ts, PRINT_SYNS):
        return "PRINT"
    if clean_text and re.search(r"\bprint\s*\(", clean_text):
        return "PRINT"

    return "UNKNOWN"


def extract_params(intent, pos_tags):

    nouns = [w for (w, tag) in pos_tags if tag.startswith("NN")]  # noun
    verbs = [w for (w, tag) in pos_tags if tag.startswith("VB")]  # verb
    numbers = [w for (w, tag) in pos_tags if tag.startswith("CD")] # numbers
    adj = [w for (w, tag) in pos_tags if tag.startswith("JJ")]    # adjectives
    others = [w for (w, tag) in pos_tags if tag not in ("NN", "VB", "CD", "JJ")]

    if intent == "DEFINE_FUNCTION":
        return {
            "function_name": nouns[0] if nouns else "my_function",
            "params": ", ".join(nouns[1:]) if len(nouns) > 1 else "",
            "body": "# TODO"
        }
    
    elif intent == "CALL_FUNCTION":
        return {
            "function_name": nouns[0] if nouns else "my_function",
            "args": ", ".join(nouns[1:]) if len(nouns) > 1 else ""
        }

    elif intent == "LOOP":
        return {
            "var": nouns[0] if nouns else "item",
            "iterable": nouns[1] if len(nouns) > 1 else "items"
        }
    
    elif intent == "CONDITIONAL":
        return {
            "condition": " ".join(nouns) if nouns else "x > 0",
            "body": "# do something",
            "else_body": "# else do something"
        }

    elif intent == "IMPORT":
        return {
            "module": nouns[0] if nouns else "math"
        }

    elif intent == "VARIABLE_ASSIGN":
        return {
            "var": nouns[0] if nouns else "x",
            "value": numbers[0] if numbers else "0"
        }

    elif intent == "CLASS_DEF":
        return {
            "class_name": nouns[0].capitalize() if nouns else "MyClass",
            "params": ", ".join(nouns[1:]) if len(nouns) > 1 else "",
            "body": "# class body"
        }

    elif intent == "PRINT":
        return {
            "content": nouns[0] if nouns else "'something'"
        }

    return {}