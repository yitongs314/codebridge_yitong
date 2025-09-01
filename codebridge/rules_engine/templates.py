TEMPLATES = {
    "DEFINE_FUNCTION": """def {function_name}({params}):
    {body}
""",

    "CALL_FUNCTION": """{function_name}({args})""",

    "LOOP": """for {var} in {iterable}:
    print({var})""",

    "CONDITIONAL": """if {condition}:
    {body}
else:
    {else_body}""",

    "IMPORT": """import {module}""",

    "VARIABLE_ASSIGN": """{var} = {value}""",

    "CLASS_DEF": """class {class_name}:
    def __init__(self, {params}):
        {body}""",

    "PRINT": """print({content})"""
}