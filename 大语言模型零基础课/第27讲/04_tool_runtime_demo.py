# 第27讲实验：Toy Tool Runtime
# 目标：演示“模型提议 Tool Call，Runtime 校验并执行”。
# 不调用任何真实外部服务，全部是本地玩具工具。

from dataclasses import dataclass
import json

# -------------------------
# 1. Tools
# -------------------------
TOY_RECORDS = {
    "project": "Agent Systems",
    "budget_usd": 200,
}
TOY_FILES = {
    "/workspace/report.txt",
    "/workspace/data.csv",
}

def calculator(a, b, op):
    if op == "add":
        return a + b
    if op == "sub":
        return a - b
    if op == "mul":
        return a * b
    if op == "div":
        if b == 0:
            raise ValueError("division by zero")
        return a / b
    raise ValueError(f"unsupported op: {op}")

def lookup_record(key):
    if key not in TOY_RECORDS:
        raise KeyError(f"record not found: {key}")
    return TOY_RECORDS[key]

def file_exists(path):
    # 只允许查询 /workspace 下的路径
    if not path.startswith("/workspace/"):
        raise PermissionError("path outside allowed workspace")
    return path in TOY_FILES

TOOLS = {
    "calculator": {
        "required": {"a": (int, float), "b": (int, float), "op": str},
        "fn": calculator,
    },
    "lookup_record": {
        "required": {"key": str},
        "fn": lookup_record,
    },
    "file_exists": {
        "required": {"path": str},
        "fn": file_exists,
    },
}

# -------------------------
# 2. Runtime validation
# -------------------------
def validate_call(call):
    if not isinstance(call, dict):
        return False, "call must be an object"
    tool = call.get("tool")
    args = call.get("arguments")
    if tool not in TOOLS:
        return False, f"unknown tool: {tool}"
    if not isinstance(args, dict):
        return False, "arguments must be an object"

    required = TOOLS[tool]["required"]
    for name, expected_type in required.items():
        if name not in args:
            return False, f"missing argument: {name}"
        if not isinstance(args[name], expected_type):
            return False, f"wrong type for {name}: {type(args[name]).__name__}"
    return True, "ok"

def execute_tool_call(call):
    ok, msg = validate_call(call)
    if not ok:
        return {"status": "rejected", "error": msg}

    tool = call["tool"]
    args = call["arguments"]
    try:
        value = TOOLS[tool]["fn"](**args)
        return {"status": "success", "value": value}
    except Exception as e:
        return {"status": "error", "error": f"{type(e).__name__}: {e}"}

# -------------------------
# 3. Toy "LLM" proposals
# -------------------------
# 这里只是手写几个 Tool Call，模拟 LLM 生成的结构化动作。
calls = [
    {
        "name": "legal calculator call",
        "call": {"tool": "calculator", "arguments": {"a": 173, "b": 294, "op": "mul"}},
    },
    {
        "name": "unknown tool hallucination",
        "call": {"tool": "delete_everything", "arguments": {}},
    },
    {
        "name": "missing parameter",
        "call": {"tool": "calculator", "arguments": {"a": 5, "op": "add"}},
    },
    {
        "name": "authorized file check",
        "call": {"tool": "file_exists", "arguments": {"path": "/workspace/data.csv"}},
    },
    {
        "name": "blocked path",
        "call": {"tool": "file_exists", "arguments": {"path": "/etc/passwd"}},
    },
    {
        "name": "lookup external state",
        "call": {"tool": "lookup_record", "arguments": {"key": "budget_usd"}},
    },
]

if __name__ == "__main__":
    print("Toy Tool Runtime")
    print("=" * 72)

    for item in calls:
        print()
        print("CASE:", item["name"])
        print("MODEL PROPOSED:")
        print(json.dumps(item["call"], ensure_ascii=False))
        result = execute_tool_call(item["call"])
        print("RUNTIME RESULT:")
        print(json.dumps(result, ensure_ascii=False))

    print()
    print("=" * 72)
    print("Key lesson:")
    print("The model proposes an action.")
    print("The runtime validates permissions and arguments.")
    print("Only the runtime executes the tool.")
