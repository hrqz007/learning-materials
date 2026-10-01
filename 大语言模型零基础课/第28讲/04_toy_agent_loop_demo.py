# 第28讲实验：Toy Agent Loop
# 目标：演示 Goal -> Observe -> State -> Decide -> Act -> Feedback -> Replan -> Stop
# 不使用真实 LLM；toy_policy 只是手写决策规则，用于看清 Agent 系统结构。

from dataclasses import dataclass, field

# -------------------------
# 1. Environment
# -------------------------
class ToyEnvironment:
    def __init__(self):
        self.files = {
            "data.txt": "10 20 30 40",
        }

    def file_exists(self, name):
        return name in self.files

    def read_numbers(self, name):
        if name not in self.files:
            return {"ok": False, "error": "file not found"}
        try:
            nums = [float(x) for x in self.files[name].split()]
            return {"ok": True, "numbers": nums}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def write_summary(self, value):
        self.files["summary.txt"] = f"mean={value}"
        return {"ok": True, "path": "summary.txt"}

    def verify_summary(self, expected):
        if "summary.txt" not in self.files:
            return {"ok": False, "reason": "summary missing"}
        actual = self.files["summary.txt"]
        target = f"mean={expected}"
        return {"ok": actual == target, "actual": actual, "expected": target}

# -------------------------
# 2. Agent State
# -------------------------
@dataclass
class AgentState:
    step: int = 0
    data_exists: bool | None = None
    numbers: list = field(default_factory=list)
    computed_mean: float | None = None
    summary_written: bool = False
    verified: bool = False
    last_error: str | None = None
    injected_error_once: bool = False
    done: bool = False

# -------------------------
# 3. Toy Policy
# -------------------------
def toy_policy(state: AgentState):
    # 这相当于“LLM / policy 根据当前 state 决定 action”
    if state.data_exists is None:
        return ("file_exists", {"name": "data.txt"})

    if state.data_exists is False:
        return ("stop", {"reason": "data.txt missing"})

    if not state.numbers:
        return ("read_numbers", {"name": "data.txt"})

    if state.computed_mean is None:
        return ("compute_mean", {})

    if not state.summary_written:
        # 故意第一次写错，模拟 Agent / model error
        value = state.computed_mean
        if not state.injected_error_once:
            value = value + 1
        return ("write_summary", {"value": value})

    if not state.verified:
        return ("verify_summary", {"expected": state.computed_mean})

    return ("stop", {"reason": "goal achieved"})

# -------------------------
# 4. Agent Loop
# -------------------------
def run_agent(max_steps=12):
    env = ToyEnvironment()
    state = AgentState()

    print("GOAL: read data.txt -> compute mean -> write summary.txt -> verify output")
    print("=" * 88)

    while not state.done and state.step < max_steps:
        state.step += 1
        action, args = toy_policy(state)

        print(f"\nSTEP {state.step}")
        print("STATE:", state)
        print("ACTION:", action, args)

        if action == "file_exists":
            result = {"exists": env.file_exists(**args)}
            state.data_exists = result["exists"]

        elif action == "read_numbers":
            result = env.read_numbers(**args)
            if result["ok"]:
                state.numbers = result["numbers"]
            else:
                state.last_error = result["error"]

        elif action == "compute_mean":
            state.computed_mean = sum(state.numbers) / len(state.numbers)
            result = {"mean": state.computed_mean}

        elif action == "write_summary":
            result = env.write_summary(**args)
            state.summary_written = result["ok"]
            if not state.injected_error_once:
                state.injected_error_once = True

        elif action == "verify_summary":
            result = env.verify_summary(**args)
            if result["ok"]:
                state.verified = True
                state.last_error = None
            else:
                # Feedback -> State update -> next round chooses repair action
                state.last_error = (
                    f"verification failed: actual={result['actual']} "
                    f"expected={result['expected']}"
                )
                state.summary_written = False  # force rewrite

        elif action == "stop":
            result = args
            state.done = True

        else:
            result = {"error": "unknown action"}
            state.last_error = "unknown action"
            state.done = True

        print("OBSERVATION:", result)

    print("\n" + "=" * 88)
    print("FINAL STATE:", state)
    print("FILES:", env.files)
    print("SUCCESS:", state.verified)

if __name__ == "__main__":
    run_agent()
