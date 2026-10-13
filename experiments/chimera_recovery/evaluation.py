"""Grade complete, pinned recovery recordings without loading models or tools.

Deterministic task oracles are distinct from generation failure diagnostics and
teacher-token agreement. A matched comparison needs the same heldout inputs,
corpus, tokenizer, template, decoding, runtime and grader pins. Code is interpreted
by a bounded AST subset; Python exec/eval and historical commands are never used.
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import operator
import re
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path

from .corpus import SHA_RE, canonical, check_arguments, digest, file_sha256, iter_jsonl, supported_schema

PIN_FIELDS = ("holdout_sha256", "corpus_sha256", "tokenizer_sha256", "template_sha256",
              "input_config_sha256", "decoding_config_sha256", "runtime_sha256")
HORIZONS = (5, 8, 16, 32)


def supported_eval_schema(schema, depth=0):
    """Captured local-tool extension: bounds, patterns and if/then/allOf.

    References, dynamic catalogs and other JSON-schema constructs are ungraded.
    These constraints are validated in place, never discarded for readiness.
    """
    if not isinstance(schema, dict) or depth > 12:
        return False
    extra = {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "minLength", "maxLength", "pattern", "allOf", "if", "then", "else"}
    base = {k: v for k, v in schema.items() if k not in extra}
    if "properties" in base:
        if not isinstance(base["properties"], dict) or not all(supported_eval_schema(v, depth + 1) for v in base["properties"].values()):
            return False
        base["properties"] = {k: {} for k in base["properties"]}
    if "items" in base:
        if not supported_eval_schema(base["items"], depth + 1):
            return False
        base["items"] = {}
    if not supported_schema(base):
        return False
    for key in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"):
        if key in schema and (type(schema[key]) not in (int, float) or not math.isfinite(schema[key])):
            return False
    for key in ("minLength", "maxLength"):
        if key in schema and (type(schema[key]) is not int or not 0 <= schema[key] <= 4096):
            return False
    if "pattern" in schema:
        if not isinstance(schema["pattern"], str) or len(schema["pattern"]) > 256:
            return False
        try:
            re.compile(schema["pattern"])
        except re.error:
            return False
    if "allOf" in schema and (not isinstance(schema["allOf"], list) or not schema["allOf"] or len(schema["allOf"]) > 16 or not all(supported_eval_schema(s, depth + 1) for s in schema["allOf"])):
        return False
    return all(supported_eval_schema(schema[k], depth + 1) for k in ("if", "then", "else") if k in schema)


def check_eval_arguments(value, schema, prefix="arguments"):
    errors = check_arguments(schema_numbers(value, schema), schema, prefix)
    if "enum" in schema and not any(json_equivalent(value, v) for v in schema["enum"]):
        errors.append(prefix + ":enum")
    if isinstance(value, (int, float)) and type(value) is not bool:
        for key, predicate in (("minimum", operator.ge), ("maximum", operator.le), ("exclusiveMinimum", operator.gt), ("exclusiveMaximum", operator.lt)):
            if key in schema and not predicate(value, schema[key]):
                errors.append(prefix + ":" + key)
    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            errors.append(prefix + ":minLength")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errors.append(prefix + ":maxLength")
        if "pattern" in schema and (len(value) > 4096 or re.search(schema["pattern"], value) is None):
            errors.append(prefix + ":pattern")
    if isinstance(value, dict):
        for key in value.keys() & schema.get("properties", {}).keys():
            errors.extend(check_eval_arguments(value[key], schema["properties"][key], prefix + "." + key))
    elif isinstance(value, list) and isinstance(schema.get("items"), dict):
        for index, sub in enumerate(value):
            errors.extend(check_eval_arguments(sub, schema["items"], f"{prefix}[{index}]"))
    for sub in schema.get("allOf", []):
        errors.extend(check_eval_arguments(value, sub, prefix))
    if "if" in schema:
        branch = "else" if check_eval_arguments(value, schema["if"], prefix) else "then"
        if branch in schema:
            errors.extend(check_eval_arguments(value, schema[branch], prefix))
    return sorted(set(errors))


class UnsupportedCode(ValueError):
    pass


class Returned(Exception):
    def __init__(self, value):
        self.value = value


class SafePython:
    """Small deterministic interpreter, no imports, IO, attributes or exec.

    Supports pure functions, bounded for loops/list comprehensions, local names,
    arithmetic/comparison, and a short list of explicitly handled collection
    operations. Unsupported valid Python is ungraded, not silently incorrect.
    """
    OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
           ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod}
    CMP = {ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt, ast.LtE: operator.le,
           ast.Gt: operator.gt, ast.GtE: operator.ge, ast.In: lambda a, b: a in b,
           ast.NotIn: lambda a, b: a not in b, ast.Is: operator.is_, ast.IsNot: operator.is_not}

    def __init__(self, source: str, budget=20000):
        if len(source) > 40000:
            raise UnsupportedCode("source exceeds bounded fixture limit")
        try:
            module = ast.parse(source)
        except SyntaxError as e:
            raise ValueError("invalid Python syntax") from e
        self.functions = {}
        self.remaining = budget
        self.depth = 0
        for node in module.body:
            if isinstance(node, ast.FunctionDef) and not node.decorator_list and not node.args.vararg and not node.args.kwarg and not node.args.kwonlyargs:
                annotations = [a.annotation for a in node.args.posonlyargs + node.args.args if a.annotation] + ([node.returns] if node.returns else [])
                for annotation in annotations:
                    if any(isinstance(n, (ast.Call, ast.Attribute)) or isinstance(n, ast.Name) and n.id not in {"int", "float", "str", "bool", "list", "tuple", "set", "dict", "object"} for n in ast.walk(annotation)):
                        raise UnsupportedCode("only safe built-in type annotations permitted")
                # Real Python evaluates defaults at definition time. Restrict
                # them even when supplied fixture args would bypass the default.
                for default in node.args.defaults:
                    if any(isinstance(n, (ast.Call, ast.Attribute, ast.Name, ast.Subscript, ast.comprehension)) for n in ast.walk(default)):
                        raise UnsupportedCode("only literal function defaults permitted")
                self.functions[node.name] = node
            elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                continue
            else:
                raise UnsupportedCode("only pure function definitions permitted")
        if not self.functions:
            raise ValueError("no function definition")

    def tick(self):
        self.remaining -= 1
        if self.remaining < 0:
            raise ValueError("bounded interpreter instruction limit")

    def bound(self, value):
        if isinstance(value, (str, list, tuple, dict, set, range)) and len(value) > 4096:
            raise ValueError("bounded collection limit")
        if isinstance(value, (int, float)) and (not math.isfinite(value) or abs(value) > 10 ** 12):
            raise ValueError("bounded numeric limit")
        return value

    def assign(self, target, value, env):
        if isinstance(target, ast.Name) and not target.id.startswith("__"):
            env[target.id] = self.bound(value)
        elif isinstance(target, (ast.Tuple, ast.List)) and len(target.elts) == len(value):
            for t, v in zip(target.elts, value):
                self.assign(t, v, env)
        else:
            raise UnsupportedCode("unsupported assignment target")

    def expr(self, node, env):
        self.tick()
        if isinstance(node, ast.Constant) and type(node.value) in (int, float, str, bool, type(None)):
            return self.bound(node.value)
        if isinstance(node, ast.Name) and node.id in env:
            return env[node.id]
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            values = [self.expr(n, env) for n in node.elts]
            return self.bound(tuple(values) if isinstance(node, ast.Tuple) else set(values) if isinstance(node, ast.Set) else values)
        if isinstance(node, ast.Dict):
            return self.bound({self.expr(k, env): self.expr(v, env) for k, v in zip(node.keys, node.values)})
        if isinstance(node, ast.BinOp) and type(node.op) in self.OPS:
            left, right = self.expr(node.left, env), self.expr(node.right, env)
            if isinstance(node.op, ast.Mult) and isinstance(left, (str, list, tuple)) and isinstance(right, int) and len(left) * max(right, 0) > 4096:
                raise ValueError("bounded multiplication limit")
            if isinstance(node.op, ast.Mult) and isinstance(right, (str, list, tuple)) and isinstance(left, int) and len(right) * max(left, 0) > 4096:
                raise ValueError("bounded multiplication limit")
            return self.bound(self.OPS[type(node.op)](left, right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd, ast.Not)):
            value = self.expr(node.operand, env)
            return not value if isinstance(node.op, ast.Not) else self.bound(-value if isinstance(node.op, ast.USub) else value)
        if isinstance(node, ast.BoolOp):
            for sub in node.values:
                value = self.expr(sub, env)
                if isinstance(node.op, ast.And) and not value or isinstance(node.op, ast.Or) and value:
                    return value
            return value
        if isinstance(node, ast.Compare):
            left = self.expr(node.left, env)
            for op, rhs in zip(node.ops, node.comparators):
                if type(op) not in self.CMP:
                    raise UnsupportedCode("unsupported comparison")
                right = self.expr(rhs, env)
                if not self.CMP[type(op)](left, right):
                    return False
                left = right
            return True
        if isinstance(node, ast.IfExp):
            return self.expr(node.body if self.expr(node.test, env) else node.orelse, env)
        if isinstance(node, ast.Subscript):
            value = self.expr(node.value, env)
            if isinstance(node.slice, ast.Slice):
                sl = node.slice
                index = slice(self.expr(sl.lower, env) if sl.lower else None, self.expr(sl.upper, env) if sl.upper else None, self.expr(sl.step, env) if sl.step else None)
            else:
                index = self.expr(node.slice, env)
            return self.bound(value[index])
        if isinstance(node, ast.ListComp) and len(node.generators) == 1 and not node.generators[0].is_async:
            gen, result = node.generators[0], []
            local = env.copy()
            for value in self.bound(self.expr(gen.iter, env)):
                self.tick()
                self.assign(gen.target, value, local)
                if all(self.expr(cond, local) for cond in gen.ifs):
                    result.append(self.expr(node.elt, local))
                self.bound(result)
            return result
        if isinstance(node, ast.Call) and not any(k.arg is None for k in node.keywords):
            args = [self.expr(a, env) for a in node.args]
            kwargs = {k.arg: self.expr(k.value, env) for k in node.keywords}
            if isinstance(node.func, ast.Name):
                name = node.func.id
                if name in self.functions:
                    return self.call(name, args, kwargs)
                builtins = {"len": len, "sum": sum, "min": min, "max": max, "abs": abs,
                            "sorted": sorted, "list": list, "tuple": tuple, "set": set,
                            "range": range, "enumerate": lambda x: list(enumerate(x)),
                            "zip": lambda *xs: list(zip(*xs)), "all": all, "any": any}
                if name in builtins:
                    if name == "range" and (kwargs or any(type(a) is not int or abs(a) > 4096 for a in args)):
                        raise ValueError("bounded range arguments")
                    return self.bound(builtins[name](*args, **kwargs))
            if isinstance(node.func, ast.Attribute) and not kwargs:
                # Attributes are never resolved generically. Only these exact
                # local collection operations are implemented explicitly.
                if isinstance(node.func.value, ast.Name) and node.func.value.id == "dict" and node.func.attr == "fromkeys" and 1 <= len(args) <= 2:
                    return self.bound(dict.fromkeys(*args))
                receiver = self.expr(node.func.value, env)
                if type(receiver) is list and node.func.attr == "append" and len(args) == 1:
                    receiver.append(args[0])
                    self.bound(receiver)
                    return None
                if type(receiver) is set and node.func.attr == "add" and len(args) == 1:
                    receiver.add(args[0])
                    self.bound(receiver)
                    return None
                if type(receiver) is dict and node.func.attr == "get" and 1 <= len(args) <= 2:
                    return receiver.get(*args)
            raise UnsupportedCode("unapproved call or collection operation")
        raise UnsupportedCode("unapproved expression: " + type(node).__name__)

    def block(self, body, env):
        for node in body:
            self.tick()
            if isinstance(node, ast.Return):
                raise Returned(self.expr(node.value, env) if node.value else None)
            if isinstance(node, ast.Assign):
                value = self.expr(node.value, env)
                for target in node.targets:
                    self.assign(target, value, env)
            elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name) and type(node.op) in self.OPS:
                value = self.expr(ast.BinOp(left=ast.Name(id=node.target.id, ctx=ast.Load()), op=node.op, right=node.value), env)
                self.assign(node.target, self.bound(value), env)
            elif isinstance(node, ast.If):
                self.block(node.body if self.expr(node.test, env) else node.orelse, env)
            elif isinstance(node, ast.For):
                for value in self.bound(self.expr(node.iter, env)):
                    self.tick()
                    self.assign(node.target, value, env)
                    self.block(node.body, env)
                self.block(node.orelse, env)
            elif isinstance(node, ast.Expr):
                self.expr(node.value, env)
            elif isinstance(node, ast.Pass):
                continue
            else:
                raise UnsupportedCode("unapproved statement: " + type(node).__name__)

    def call(self, name, args, kwargs):
        node = self.functions.get(name)
        if node is None:
            raise ValueError("requested function missing")
        self.depth += 1
        try:
            if self.depth > 12:
                raise ValueError("bounded call depth")
            names = [a.arg for a in node.args.posonlyargs + node.args.args]
            if len(args) > len(names) or set(kwargs) - set(names) or set(names[:len(args)]) & set(kwargs):
                raise ValueError("function argument mismatch")
            env = dict(zip(names, args))
            env.update(kwargs)
            for key, default in zip(names[-len(node.args.defaults):], node.args.defaults):
                if key not in env:
                    env[key] = self.expr(default, {})
            if set(names) != set(env):
                raise ValueError("missing function arguments")
            try:
                self.block(node.body, env)
            except Returned as result:
                return self.bound(result.value)
            return None
        finally:
            self.depth -= 1


def assistant_text(messages: list[dict]) -> str:
    return "\n".join(m["content"] for m in messages if m.get("role") == "assistant" and isinstance(m.get("content"), str))


def json_equivalent(left, right) -> bool:
    if type(left) is bool or type(right) is bool:
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isfinite(left) and math.isfinite(right) and left == right
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(json_equivalent(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(json_equivalent(a, b) for a, b in zip(left, right))
    return type(left) is type(right) and left == right


def schema_numbers(value, schema):
    """JSON's integral numeric values satisfy integer, while booleans do not."""
    if schema.get("type") == "integer" and type(value) is float and math.isfinite(value) and value.is_integer():
        return int(value)
    if isinstance(value, dict):
        return {k: schema_numbers(v, schema.get("properties", {}).get(k, {})) for k, v in value.items()}
    if isinstance(value, list):
        return [schema_numbers(v, schema.get("items", {})) for v in value]
    return value


def grade_arithmetic(text: str, oracle: dict) -> dict:
    """Extract the final numeric answer by a published rule, never a target match.

    Accept a JSON answer field, or a final line ending with a number/fraction.
    Longer explanations can precede it. A correct fragment followed by unrelated
    numbers cannot receive credit by searching for the expected value.
    """
    try:
        obj = json.loads(text)
        answer = obj.get("answer") if isinstance(obj, dict) else obj
    except (ValueError, TypeError):
        line = text.strip().splitlines()[-1] if text.strip() else ""
        match = re.search(r"(?<![\w.])([-+]?\d+(?:\.\d+)?(?:\s*/\s*[-+]?\d+(?:\.\d+)?)?)\s*[.!]?\s*$", line)
        answer = match.group(1) if match else None
    if answer is None or type(answer) is bool:
        return {"status": "fail", "reason": "final_numeric_answer_missing"}
    try:
        def number(value):
            parts = str(value).split("/")
            return Decimal(parts[0].strip()) if len(parts) == 1 else Decimal(parts[0].strip()) / Decimal(parts[1].strip()) if len(parts) == 2 else Decimal("NaN")
        actual, expected = number(answer), number(oracle["expected"])
        tolerance = Decimal(str(oracle.get("absolute_tolerance", "0")))
        passed = actual.is_finite() and expected.is_finite() and tolerance.is_finite() and tolerance >= 0 and abs(actual - expected) <= tolerance
        return {"status": "pass" if passed else "fail", "actual": str(actual), "expected": str(expected), "absolute_tolerance": str(tolerance)}
    except (InvalidOperation, ValueError, ZeroDivisionError, ArithmeticError):
        return {"status": "fail", "reason": "numeric_answer_invalid"}


def grade_code(text: str, oracle: dict) -> dict:
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    if len(blocks) > 1:
        return {"status": "ungraded", "reason": "multiple_code_candidates_need_review"}
    source = blocks[0] if blocks else text.strip()
    cases = oracle.get("cases") or []
    if not cases:
        return {"status": "ungraded", "reason": "no_independent_code_cases"}
    results = []
    for case in cases:
        try:
            actual = SafePython(source).call(oracle["function"], case.get("args", []), case.get("kwargs", {}))
            passed = json_equivalent(actual, case["expected"])
            results.append({"pass": passed, "actual": actual, "expected": case["expected"]})
        except UnsupportedCode as error:
            return {"status": "ungraded", "reason": str(error), "cases_completed": results}
        except (ValueError, TypeError, KeyError, IndexError, ZeroDivisionError, OverflowError) as error:
            results.append({"pass": False, "error": str(error)})
    return {"status": "pass" if all(r["pass"] for r in results) else "fail", "cases": results}


def grade_tools(messages: list[dict], tools: list[dict], oracle: dict, outcomes: list[dict]) -> dict:
    schemas = {}
    for tool in tools:
        f = tool.get("function") or {}
        if not f.get("name") or f["name"] in schemas or not supported_eval_schema(f.get("parameters") or {}):
            return {"status": "ungraded", "reason": "authentic_supported_tool_schema_required"}
        schemas[f["name"]] = f["parameters"]
    expected = oracle.get("steps") or []
    if not expected:
        return {"status": "ungraded", "reason": "independent_tool_assertions_missing"}
    observed, pending, identifiers, results, errors = [], {}, set(), {}, []
    user_turns = 0
    user_messages = []
    user_after_steps = []
    assistant_batches = []
    final_answer = ""
    for message in messages:
        role = message.get("role")
        if role == "user":
            user_turns += 1
            user_messages.append({"role": "user", "content": message.get("content")})
            user_after_steps.append(len(observed))
            if pending:
                errors.append("unanswered_tool_before_user_turn")
        elif role == "assistant":
            calls = message.get("tool_calls") or []
            if pending:
                errors.append("unanswered_tool_before_assistant_turn")
            if calls:
                batch = []
                for call in calls:
                    identifier = call.get("id")
                    fn = call.get("function") or {}
                    if not identifier or identifier in identifiers:
                        errors.append("tool_call_id_missing_or_duplicate")
                    identifiers.add(identifier)
                    if call.get("type") != "function" or fn.get("name") not in schemas:
                        errors.append("tool_name_or_type_invalid")
                    try:
                        if not isinstance(fn.get("arguments"), str):
                            raise ValueError("arguments must remain JSON string")
                        arguments = json.loads(fn["arguments"])
                        schema = schemas.get(fn.get("name"), {})
                        mismatch = check_eval_arguments(arguments, schema)
                        if mismatch:
                            errors.extend(mismatch)
                    except (ValueError, TypeError):
                        arguments = None
                        errors.append("tool_arguments_not_valid_json_string")
                    action = {"id": identifier, "name": fn.get("name"), "arguments": arguments}
                    pending[identifier] = action
                    batch.append(action)
                observed.append(batch)
                assistant_batches.append(message)
            elif isinstance(message.get("content"), str):
                final_answer = message["content"]
        elif role == "tool":
            identifier = message.get("tool_call_id")
            if identifier not in pending:
                errors.append("tool_result_not_linked_to_pending_call")
            else:
                if message.get("name") and message["name"] != pending[identifier]["name"]:
                    errors.append("tool_result_name_mismatch")
                results[identifier] = message
                del pending[identifier]
        else:
            errors.append("unexpected_recorded_message_role")
    if len(observed) != len(expected):
        errors.append("tool_batch_count_mismatch")
    verified_outcomes = {}
    for outcome in outcomes:
        identifier = outcome.get("tool_call_id")
        if identifier in verified_outcomes or outcome.get("source_verified") is not True or not SHA_RE.fullmatch(str(outcome.get("receipt_sha256") or "")):
            errors.append("runtime_outcome_receipt_invalid")
        verified_outcomes[identifier] = outcome
        if identifier not in identifiers or outcome.get("status") not in {"success", "error"}:
            errors.append("runtime_outcome_call_or_status_invalid")
    for index, (batch, step) in enumerate(zip(observed, expected)):
        requested = step.get("calls") or []
        if len(batch) != len(requested):
            errors.append(f"step_{index}_call_count_mismatch")
        if step.get("parallel") is True and len(batch) < 2:
            errors.append(f"step_{index}_parallel_batch_missing")
        # Parallel order may vary, but matching must remain unambiguous.
        unmatched = list(batch)
        for wanted in requested:
            def matches(action):
                if action["name"] != wanted.get("name"):
                    return False
                args = action["arguments"]
                if "accepted_arguments" in wanted:
                    return any(json_equivalent(args, a) for a in wanted["accepted_arguments"])
                choices = wanted.get("argument_choices")
                optional = set(wanted.get("optional_arguments") or [])
                return isinstance(args, dict) and isinstance(choices, dict) and set(choices) - optional <= set(args) <= set(choices) and all(any(json_equivalent(args[k], v) for v in choices[k]) for k in args)
            candidates = [action for action in unmatched if matches(action)]
            if not candidates:
                errors.append(f"step_{index}_requested_arguments_not_met")
                continue
            action = candidates[0]
            unmatched.remove(action)
            wanted_outcome = wanted.get("outcome")
            if wanted_outcome:
                result, receipt = results.get(action["id"]), verified_outcomes.get(action["id"])
                if result is None or receipt is None:
                    errors.append(f"step_{index}_recorded_execution_outcome_missing")
                elif receipt.get("status") != wanted_outcome.get("status") or receipt.get("result_sha256") != digest(result):
                    errors.append(f"step_{index}_execution_status_or_result_receipt_mismatch")
                elif "content" in wanted_outcome and result.get("content") != wanted_outcome["content"]:
                    errors.append(f"step_{index}_execution_content_mismatch")
                elif "json_equals" in wanted_outcome:
                    try:
                        if not json_equivalent(json.loads(result.get("content", "")), wanted_outcome["json_equals"]):
                            errors.append(f"step_{index}_execution_json_mismatch")
                    except (ValueError, TypeError):
                        errors.append(f"step_{index}_execution_json_invalid")
    needs_execution = any(call.get("outcome") for s in expected for call in s.get("calls", []))
    if pending and (needs_execution or oracle.get("require_all_results") is True):
        errors.append("unanswered_tool_at_end")
    if oracle.get("min_followup_user_turns", 0) > user_turns:
        errors.append("multiturn_user_followup_missing")
    if "followup_user_messages" in oracle and canonical(user_messages) != canonical(oracle["followup_user_messages"]):
        errors.append("multiturn_user_followup_content_mismatch")
    if "followup_after_steps" in oracle and user_after_steps != oracle["followup_after_steps"]:
        errors.append("multiturn_user_followup_timing_mismatch")
    reviews = []
    for requirement in oracle.get("schema_reports", []):
        step = requirement["step"]
        content = assistant_batches[step].get("content") if 0 <= step < len(assistant_batches) else None
        try:
            reported = json.loads(content or "")
            if not json_equivalent(reported, schemas[requirement["name"]]):
                errors.append("reported_tool_schema_mismatch")
        except (TypeError, ValueError):
            reviews.append("natural_language_schema_report_requires_review")
    final = None
    if oracle.get("final_answer"):
        final = grade_semantics({"oracle": oracle["final_answer"]}, {"messages": [{"role": "assistant", "content": final_answer}]})
        if final["status"] == "fail":
            errors.append("tool_final_answer_not_correct")
        elif final["status"] == "ungraded":
            reviews.append(final.get("reason", "tool_final_answer_requires_review"))
    return {"status": "fail" if errors else "ungraded" if reviews else "pass", "errors": sorted(set(errors)), "reviews": reviews,
            "final_answer": final, "observed_batches": observed,
            "proposed_calls_correct": not any("arguments" in e or "name_or_type" in e or "count_mismatch" in e for e in errors),
            "execution_required": needs_execution, "pending_tool_ids": list(pending), "recorded_outcome_count": len(results)}


def grade_semantics(task: dict, response: dict) -> dict:
    oracle = task.get("oracle") or {}
    kind = oracle.get("kind")
    text = assistant_text(response.get("messages") or [])
    if kind == "arithmetic":
        return grade_arithmetic(text, oracle)
    if kind == "python_function":
        return grade_code(text, oracle)
    if kind == "instruction_json":
        # Strict JSON is an explicit task requirement, not exact answer text:
        # whitespace and object member order vary; duplicates/extra prose fail.
        def unique_pairs(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("duplicate JSON member")
                result[key] = value
            return result
        try:
            value = json.loads(text, object_pairs_hook=unique_pairs,
                               parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
            errors = check_eval_arguments(value, oracle["schema"])
            passed = not errors and json_equivalent(value, oracle["expected"])
            return {"status": "pass" if passed else "fail", "actual": value,
                    "expected": oracle["expected"], "schema_errors": errors}
        except (ValueError, TypeError, KeyError):
            return {"status": "fail", "reason": "required_strict_json_missing_or_invalid"}
    if kind == "observed_values":
        expected = oracle["expected_values"]
        try:
            stripped = text.strip()
            match = re.search(r"(\[.*\])\s*[.!]?\s*$", stripped, re.S)
            value = json.loads(match.group(1) if match else stripped)
            if isinstance(value, dict) and set(value) == {"values"}:
                value = value["values"]
            if not isinstance(value, list) and len(expected) == 1:
                value = [value]
            return {"status": "pass" if json_equivalent(value, expected) else "fail", "actual": value, "expected": expected}
        except (TypeError, ValueError):
            if len(expected) == 1 and isinstance(expected[0], str) and text.strip() == expected[0]:
                return {"status": "pass", "actual": [text.strip()], "expected": expected}
            return {"status": "ungraded", "reason": "unstructured_final_values_require_review", "expected": expected}
    if kind == "tool":
        return grade_tools(response.get("messages") or [], (task.get("input") or {}).get("tools") or [], oracle, response.get("tool_outcomes") or [])
    return {"status": "ungraded", "reason": "independent_semantic_oracle_missing", "required_review": oracle.get("review_requirements", [])}


def generation_diagnostics(response: dict) -> dict:
    text = assistant_text(response.get("messages") or [])
    gen = response.get("generation") or {}
    tokens = gen.get("token_ids")
    units = tokens if isinstance(tokens, list) else re.findall(r"\S+", text)
    repeats = []
    # Consecutive repeated spans are a diagnostic, not an intelligence label.
    for size in range(1, min(32, len(units) // 3) + 1):
        covered_until = -1
        for start in range(len(units) - 3 * size + 1):
            if start < covered_until:
                continue
            pattern = units[start:start + size]
            count = 1
            while start + (count + 1) * size <= len(units) and units[start + count * size:start + (count + 1) * size] == pattern:
                count += 1
            if count >= 3:
                repeats.append({"start": start, "span": size, "repetitions": count})
                covered_until = start + size * count
    longest = max(repeats, key=lambda r: r["span"] * r["repetitions"], default=None)
    calls = sum(len(m.get("tool_calls") or []) for m in response.get("messages") or [])
    return {"empty_assistant": not text.strip() and not calls, "stop_reason": gen.get("stop_reason", "unknown"),
            "eos": gen.get("stop_reason") == "eos", "cap_reached": gen.get("stop_reason") == "length",
            "loop_candidate": longest, "loop_units": "tokens" if isinstance(tokens, list) else "whitespace_words",
            "generated_tokens": len(tokens) if isinstance(tokens, list) else gen.get("generated_tokens"),
            "protocol_complete": bool(text.strip() or calls) and gen.get("stop_reason") == "eos" and longest is None}


def token_metrics(task: dict, response: dict) -> dict:
    reference = task.get("teacher_reference") or {}
    target = reference.get("token_ids")
    generated = (response.get("generation") or {}).get("token_ids")
    result = {"teacher_forced": None, "free_run": None}
    if not isinstance(target, list) or not target or any(type(v) is not int for v in target):
        return result
    if isinstance(generated, list) and all(type(v) is int for v in generated):
        matches = sum(a == b for a, b in zip(generated, target))
        lcp = 0
        for a, b in zip(generated, target):
            if a != b:
                break
            lcp += 1
        result["free_run"] = {"reference_positions": len(target), "generated_positions": len(generated),
                              "positional_matches": matches, "positional_match_rate": matches / len(target), "lcp": lcp,
                              "exact_continuation": {str(h): (generated[:h] == target[:h]) if min(len(generated), len(target)) >= h else None for h in HORIZONS},
                              "exact_complete_reference": generated == target,
                              "scope": "teacher agreement only; alternate correct answers may differ"}
    recorded = response.get("teacher_forced")
    if recorded is not None:
        if recorded.get("context_kind") != "teacher_forced" or recorded.get("context_sha256") != reference.get("context_sha256") or recorded.get("teacher_checkpoint_sha256") != reference.get("checkpoint_sha256"):
            raise ValueError("teacher-forced context/checkpoint mismatch")
        k = recorded.get("k")
        if type(k) is not int or k < 1:
            raise ValueError("top-k width missing")
        positions, top1, topk = set(), 0, 0
        for item in recorded.get("positions") or []:
            pos, ranks = item.get("position"), item.get("topk_token_ids")
            if type(pos) is not int or not 0 <= pos < len(target) or pos in positions or not isinstance(ranks, list) or len(ranks) != k or len(set(ranks)) != k or any(type(t) is not int for t in ranks):
                raise ValueError("invalid/duplicate teacher-forced position")
            positions.add(pos)
            top1 += ranks[0] == target[pos]
            topk += target[pos] in ranks
        result["teacher_forced"] = {"k": k, "positions_covered": len(positions), "reference_positions": len(target),
                                    "position_coverage": len(positions) / len(target), "top1_matches": top1, "topk_matches": topk,
                                    "top1_rate_on_covered": top1 / len(positions) if positions else None,
                                    "topk_rate_on_covered": topk / len(positions) if positions else None,
                                    "full_position_coverage": len(positions) == len(target), "scope": "gold prefix next-token ranking, not consecutive generation"}
    return result


def validate_recording(tasks: list[dict], receipt: dict, responses: list[dict], holdout_sha256: str) -> None:
    if receipt.get("schema") != "recovery_eval_run.v1" or receipt.get("phase") not in {"baseline", "after_recovery"}:
        raise ValueError("run receipt schema/phase invalid")
    pins = receipt.get("pins") or {}
    if any(not SHA_RE.fullmatch(str(pins.get(key) or "")) for key in PIN_FIELDS) or pins["holdout_sha256"] != holdout_sha256:
        raise ValueError("run source/input pins missing or mismatched")
    if not SHA_RE.fullmatch(str((receipt.get("model") or {}).get("checkpoint_sha256") or "")):
        raise ValueError("model checkpoint pin missing")
    if receipt.get("recording_complete") is not True or receipt.get("full_answers_published") is not True:
        raise ValueError("complete full-answer recording required")
    if receipt.get("responses_sha256") != digest(responses):
        raise ValueError("response payload seal mismatch")
    ids = [t.get("id") for t in tasks]
    seen = [r.get("id") for r in responses]
    if not ids or None in ids or len(set(ids)) != len(ids) or len(set(seen)) != len(seen) or set(ids) != set(seen):
        raise ValueError("every heldout task must appear exactly once")
    by_id = {r["id"]: r for r in responses}
    training_ids = receipt.get("training_group_ids")
    if not isinstance(training_ids, list) or any(not isinstance(g, str) or not g for g in training_ids) or len(set(training_ids)) != len(training_ids):
        raise ValueError("explicit complete recovery training group inventory required")
    trained = set(training_ids)
    for task in tasks:
        if task.get("schema") != "recovery_eval_task.v1" or task.get("split") not in {"validation", "test"} or not task.get("group_id"):
            raise ValueError("task lacks explicit heldout group/split")
        if task["group_id"] in trained:
            raise ValueError("heldout source group entered training")
        response = by_id[task["id"]]
        if response.get("schema") != "recovery_eval_response.v1" or response.get("input_sha256") != digest(task.get("input")) or not isinstance(response.get("messages"), list):
            raise ValueError("recorded task input/messages mismatch")
        generation = response.get("generation") or {}
        if generation.get("stop_reason") not in {"eos", "length", "error"} or not isinstance(generation.get("max_new_tokens"), int) or generation["max_new_tokens"] < 1:
            raise ValueError("generation termination/budget receipt missing")


def evaluate(tasks: list[dict], receipt: dict, responses: list[dict], *, holdout_sha256: str) -> dict:
    validate_recording(tasks, receipt, responses, holdout_sha256)
    by_id = {r["id"]: r for r in responses}
    rows = []
    for task in tasks:
        response = by_id[task["id"]]
        semantic = grade_semantics(task, response)
        diagnostics = generation_diagnostics(response)
        rows.append({"id": task["id"], "group_id": task["group_id"], "route": task.get("route"), "split": task["split"],
                     "semantic": semantic, "generation": diagnostics, "teacher_metrics": token_metrics(task, response),
                     "successful_complete_response": semantic["status"] == "pass" and diagnostics["protocol_complete"],
                     "full_input": task["input"], "full_messages": response["messages"], "full_tool_outcomes": response.get("tool_outcomes", [])})
    report = {"schema": "recovery_eval_report.v1", "run": receipt, "grader_sha256": file_sha256(Path(__file__)),
              "counts": {"tasks": len(rows), "semantic": dict(Counter(r["semantic"]["status"] for r in rows)),
                         "empty": sum(r["generation"]["empty_assistant"] for r in rows), "caps": sum(r["generation"]["cap_reached"] for r in rows),
                         "loop_candidates": sum(r["generation"]["loop_candidate"] is not None for r in rows),
                         "successful_complete_responses": sum(r["successful_complete_response"] for r in rows)}, "rows": rows,
              "limitations": ["Only declared independent oracles receive automatic semantic grades; media/open reasoning require recorded external evaluation.",
                              "Repetition heuristics are separate generation diagnostics, never IQ.", "Teacher agreement is not semantic correctness.",
                              "Tool outcomes must be recorded and receipt-linked; evaluator executes no tool."]}
    report["report_sha256"] = digest(report)
    return report


def compare(before: dict, after: dict) -> dict:
    for report in (before, after):
        if report.get("schema") != "recovery_eval_report.v1" or report.get("report_sha256") != digest({k: v for k, v in report.items() if k != "report_sha256"}):
            raise ValueError("evaluation report seal invalid")
    if before["run"]["phase"] != "baseline" or after["run"]["phase"] != "after_recovery":
        raise ValueError("comparison phases invalid")
    keys = set(before["run"]["pins"]) | set(after["run"]["pins"])
    mismatched = [k for k in keys if before["run"]["pins"].get(k) != after["run"]["pins"].get(k)]
    if before["grader_sha256"] != after["grader_sha256"]:
        mismatched.append("grader_sha256")
    if mismatched:
        raise ValueError("unmatched comparison pins: " + ", ".join(sorted(mismatched)))
    base_sha = before["run"]["model"]["checkpoint_sha256"]
    if (after["run"].get("recovery") or {}).get("base_checkpoint_sha256") != base_sha or not SHA_RE.fullmatch(str((after["run"].get("recovery") or {}).get("training_receipt_sha256") or "")):
        raise ValueError("after-recovery lineage/training receipt missing")
    left, right = {r["id"]: r for r in before["rows"]}, {r["id"]: r for r in after["rows"]}
    if set(left) != set(right) or any(digest(left[k]["full_input"]) != digest(right[k]["full_input"]) or left[k]["group_id"] != right[k]["group_id"] for k in left):
        raise ValueError("comparison heldout inputs/groups differ")
    cases = [{"id": k, "before": left[k], "after": right[k]} for k in left]
    result = {"schema": "recovery_eval_comparison.v1", "before_report_sha256": before["report_sha256"], "after_report_sha256": after["report_sha256"],
              "matched_pins": before["run"]["pins"], "models": {"before": before["run"]["model"], "after": after["run"]["model"]},
              "before_counts": before["counts"], "after_counts": after["counts"], "full_answers_and_outcomes": cases}
    result["comparison_sha256"] = digest(result)
    return result


def publish(report: dict, output: Path) -> None:
    if output.exists():
        raise ValueError("evaluation output must be a new directory")
    output.mkdir(parents=True)
    (output / "report.json").write_bytes(canonical(report) + b"\n")
    parts = ["# Recovery evaluation: complete recorded answers\n",
             "Semantic correctness, generation termination/repetition and teacher token agreement are separate.\n"]
    for row in report.get("rows") or report.get("full_answers_and_outcomes") or []:
        parts.append("\n## " + str(row["id"]) + "\n\n```json\n" + json.dumps(row, ensure_ascii=False, indent=2) + "\n```\n")
    (output / "full_answers_and_outcomes.md").write_text("\n".join(parts), encoding="utf-8")


def prepare_bfcl(*, heldout_corpus: Path, training_corpus: Path, expected_heldout_sha256: str,
                 expected_training_sha256: str, output: Path) -> dict:
    """Derive proposal-only oracles from captured BFCL acceptable alternatives.

    Ground truth stays in the evaluator task oracle; input contains only leading
    source messages and authentic schemas. No execution/result is synthesized.
    """
    if file_sha256(heldout_corpus) != expected_heldout_sha256 or file_sha256(training_corpus) != expected_training_sha256:
        raise ValueError("heldout/training corpus pin mismatch")
    training_groups = set()
    for row in iter_jsonl(training_corpus):
        group = (row.get("provenance") or {}).get("recovery_group_id")
        if row.get("split") != "train" or not group:
            raise ValueError("training corpus explicit group/split required")
        training_groups.add(group)
    tasks, excluded = [], Counter()
    for row in iter_jsonl(heldout_corpus):
        p = row.get("provenance") or {}
        adaptation = p.get("bfcl_schema_target_adaptation") or {}
        ground = (adaptation.get("original_acceptable_targets") or {}).get("expected_tool_calls")
        if not ground:
            excluded["no_captured_bfcl_acceptable_targets"] += 1
            continue
        group = p.get("recovery_group_id")
        if row.get("split") not in {"validation", "test"} or not group or group in training_groups:
            raise ValueError("BFCL heldout group overlaps training or lacks explicit split")
        if p.get("source_verified") is not True or (row.get("quality") or {}).get("schema_complete") is not True:
            excluded["source_schema_not_verified"] += 1
            continue
        tools = row.get("tools") or []
        schemas = {t["function"]["name"]: t["function"]["parameters"] for t in tools}
        if not schemas or any(not supported_schema(s) for s in schemas.values()):
            excluded["schema_outside_supported_subset"] += 1
            continue
        calls = []
        for entry in ground:
            if not isinstance(entry, dict) or len(entry) != 1:
                break
            name, choices = next(iter(entry.items()))
            if name not in schemas or not isinstance(choices, dict) or any(not isinstance(v, list) or not v for v in choices.values()):
                break
            optional = [key for key, alternatives in choices.items() if key not in schemas[name].get("required", []) and "" in alternatives]
            calls.append({"name": name, "argument_choices": choices, "optional_arguments": optional})
        if len(calls) != len(ground):
            excluded["unsupported_acceptable_target_shape"] += 1
            continue
        messages = []
        for message in row.get("messages") or []:
            if message.get("role") == "assistant":
                break
            messages.append(message)
        if not messages or any(m.get("role") not in {"user", "system", "developer"} for m in messages):
            excluded["source_input_not_isolatable"] += 1
            continue
        tasks.append({"schema": "recovery_eval_task.v1", "id": "bfcl-heldout:" + row["id"], "group_id": group,
                      "split": row["split"], "route": row["route"], "input": {"messages": messages, "tools": tools, "media": []},
                      "oracle": {"kind": "tool", "steps": [{"parallel": len(calls) > 1, "calls": calls}], "require_all_results": False},
                      "provenance": {"source": "captured_bfcl_acceptable_targets", "source_row_id": row["id"], "row_sha256": digest(row), "source_ids": p.get("source_ids"),
                                     "original_schemas": adaptation.get("original_schemas"), "source_dates": p.get("source_dates"),
                                     "heldout_corpus_sha256": expected_heldout_sha256, "training_corpus_sha256": expected_training_sha256},
                      "evaluation_scope": "proposed call names/arguments and same-batch parallelism only; no execution outcome claimed"})
    if not tasks:
        raise ValueError("no genuine heldout BFCL tasks admitted")
    if output.exists():
        raise ValueError("task output directory must be new")
    output.mkdir(parents=True)
    path = output / "tasks.jsonl"
    path.write_bytes(b"".join(canonical(t) + b"\n" for t in tasks))
    inputs_path = output / "inputs.jsonl"
    inputs_path.write_bytes(b"".join(canonical({"id": t["id"], "group_id": t["group_id"], "split": t["split"], "input": t["input"], "input_sha256": digest(t["input"])}) + b"\n" for t in tasks))
    groups_path = output / "training_group_ids.json"
    groups_path.write_bytes(canonical(sorted(training_groups)) + b"\n")
    manifest = {"schema": "recovery_eval_tasks_manifest.v1", "tasks": len(tasks), "routes": dict(Counter(t["route"] for t in tasks)),
                "excluded": dict(excluded), "holdout_sha256": file_sha256(path), "inputs_sha256": file_sha256(inputs_path), "training_group_ids_sha256": digest(sorted(training_groups)),
                "heldout_corpus_sha256": expected_heldout_sha256, "training_corpus_sha256": expected_training_sha256,
                "task_source_groups_disjoint_from_training": True, "source_group_count": len({t["group_id"] for t in tasks}),
                "grader_sha256": file_sha256(Path(__file__)), "limitations": ["Proposal-only BFCL tasks do not establish successful tool execution, retries, or multi-turn agent behavior.",
                "Source targets remain mechanically schema-valid, not independently manually graded; acceptable alternatives are retained verbatim."]}
    manifest["manifest_sha256"] = digest(manifest)
    (output / "manifest.json").write_bytes(canonical(manifest) + b"\n")
    return manifest


def prepare_tool_seeds(*, ready: Path, execution_receipts: Path, seed_report: Path,
                       expected_ready_sha256: str, expected_receipts_sha256: str,
                       expected_report_sha256: str, output: Path, split="test") -> dict:
    """Derive executed-outcome oracles from genuine bounded local-tool episodes.

    Reads already recorded evidence only. No command/file tool is executed here.
    Operator-authored synthetic scenarios retain that label and benchmark limit.
    """
    sources = {"ready": ready, "execution_receipts": execution_receipts, "seed_report": seed_report}
    wanted = {"ready": expected_ready_sha256, "execution_receipts": expected_receipts_sha256, "seed_report": expected_report_sha256}
    for name, path in sources.items():
        if not SHA_RE.fullmatch(str(wanted[name])) or file_sha256(path) != wanted[name]:
            raise ValueError("tool seed source pin mismatch: " + name)
    report = json.loads(seed_report.read_text(encoding="utf-8"))
    for name, path in (("ready.jsonl", ready), ("execution_receipts.jsonl", execution_receipts)):
        if report.get("artifacts", {}).get(name, {}).get("sha256") != file_sha256(path):
            raise ValueError("tool seed report not bound to actual artifacts")
    if split not in {"validation", "test"}:
        raise ValueError("tool seed evaluation split must be heldout")
    rows = list(iter_jsonl(ready))
    receipts = list(iter_jsonl(execution_receipts))
    receipt_by_id = {r.get("id"): r for r in receipts}
    if len(receipt_by_id) != len(receipts) or len(rows) != report.get("episodes") or len(rows) != len(receipts):
        raise ValueError("tool seed row/receipt cardinality mismatch")
    seen = {}
    for row in rows:
        p = row.get("provenance") or {}
        row_split = row.get("split") or p.get("split")
        if row_split not in {"train", "validation", "test"} or p.get("source_year") != 2026 or p.get("source_verified") is not True or p.get("actual_tool_execution") is not True or p.get("synthetic_operator_authored") is not True:
            raise ValueError("tool seed provenance/split is not authentic operator fixture")
        if p.get("source_code_sha256") != report.get("source_code_sha256"):
            raise ValueError("tool seed source-code receipt mismatch")
        keys = [("group", p.get("group_id")), ("family", p.get("template_family")), ("case", p.get("case_id"))]
        keys += [("source", v) for v in p.get("source_ids", [])]
        keys += [("user_prompt", digest(m.get("content"))) for m in row["messages"] if m.get("role") == "user"]
        keys += [("tool_parameters", digest([call["function"]["name"], call["function"]["arguments"]])) for m in row["messages"] for call in m.get("tool_calls", [])]
        for key in keys:
            if not key[1]:
                raise ValueError("tool seed source identity missing")
            previous = seen.setdefault(key, row_split)
            if previous != row_split:
                raise ValueError("tool seed family/parameter/source crosses splits")
    training_groups = sorted({r["provenance"]["group_id"] for r in rows if (r.get("split") or r["provenance"]["split"]) == "train"})
    tasks, recording_adapters = [], []
    for row in rows:
        p = row["provenance"]
        if (row.get("split") or p["split"]) != split:
            continue
        receipt = receipt_by_id.get(row["id"])
        calls_recorded = receipt.get("calls") if receipt else None
        if not receipt or receipt.get("template_family") != p["template_family"] or receipt.get("split") != split or digest(calls_recorded) != p.get("execution_receipt_sha256"):
            raise ValueError("tool seed execution receipt join/seal mismatch")
        call_by_id = {c["call_id"]: c for c in calls_recorded}
        if len(call_by_id) != len(calls_recorded):
            raise ValueError("tool seed receipt duplicate call ID")
        tool_results = {m["tool_call_id"]: m for m in row["messages"] if m.get("role") == "tool"}
        if len(tool_results) != len(calls_recorded):
            raise ValueError("tool seed full recorded result count mismatch")
        tools = row.get("tools") or []
        schemas = {t["function"]["name"]: t["function"]["parameters"] for t in tools}
        if not schemas or not all(supported_eval_schema(s) for s in schemas.values()):
            raise ValueError("tool seed captured schema outside evaluated subset")
        prefix, steps, followups, after_steps, schema_reports = [], [], [], [], []
        runtime_records = []
        for message in row["messages"]:
            role = message.get("role")
            if not steps and role in {"system", "developer", "user"}:
                prefix.append(message)
            elif role == "user":
                followups.append({"role": "user", "content": message.get("content")})
                after_steps.append(len(steps))
            if role != "assistant" or not message.get("tool_calls"):
                continue
            batch = []
            for call in message["tool_calls"]:
                fn, identifier = call["function"], call["id"]
                args = json.loads(fn["arguments"])
                observed = call_by_id.get(identifier)
                result_message = tool_results.get(identifier)
                if not observed or not result_message or observed["tool_name"] != fn["name"] or not json_equivalent(args, observed["arguments"]) or check_eval_arguments(args, schemas.get(fn["name"], {})):
                    raise ValueError("tool seed source call/result/schema disagreement")
                result = json.loads(result_message["content"])
                if not json_equivalent(result, observed["result"]) or digest(result) != observed["result_sha256"] or type(result.get("ok")) is not bool:
                    raise ValueError("tool seed recorded outcome does not match execution receipt")
                status = "success" if result["ok"] else "error"
                batch.append({"name": fn["name"], "accepted_arguments": [args], "outcome": {"status": status, "json_equals": result}})
                runtime_records.append({"tool_call_id": identifier, "status": status, "source_verified": True,
                                        "receipt_sha256": digest(observed), "result_sha256": digest(result_message)})
            parallel = len(batch) > 1
            if parallel and any(call_by_id[c["id"]].get("concurrent_dispatch") is not True for c in message["tool_calls"]):
                raise ValueError("tool seed parallel dispatch not observed")
            if message.get("content"):
                try:
                    reported_schema = json.loads(message["content"])
                except ValueError:
                    reported_schema = None
                for name, schema in schemas.items():
                    if json_equivalent(reported_schema, schema):
                        schema_reports.append({"step": len(steps), "name": name})
            steps.append({"parallel": parallel, "calls": batch})
        if not steps or set(tool_results) != set(call_by_id) or sum(len(s["calls"]) for s in steps) != len(calls_recorded):
            raise ValueError("tool seed executed trajectory incomplete")
        values = [c["outcome"]["json_equals"]["value"] for c in steps[-1]["calls"] if c["outcome"]["status"] == "success"]
        if len(values) != len(steps[-1]["calls"]):
            raise ValueError("tool seed ends without successful observed values")
        facets = (row.get("quality") or {}).get("tool_routes") or []
        task = {"schema": "recovery_eval_task.v1", "id": row["id"], "group_id": p["group_id"], "split": split,
                "route": facets[0] if len(facets) == 1 else "tool_multiturn", "input": {"messages": prefix, "tools": tools, "media": []},
                "oracle": {"kind": "tool", "steps": steps, "require_all_results": True, "schema_reports": schema_reports,
                           "followup_user_messages": followups, "followup_after_steps": after_steps,
                           "final_answer": {"kind": "observed_values", "expected_values": values}},
                "environment": {"followup_user_messages": followups, "followup_after_steps": after_steps,
                                "tool_catalog_sha256": p["tool_catalog_sha256"], "tool_source_code_sha256": p["source_code_sha256"]},
                "provenance": {**p, "ready_sha256": expected_ready_sha256, "execution_receipts_sha256": expected_receipts_sha256,
                               "seed_report_sha256": expected_report_sha256, "row_sha256": digest(row)},
                "evaluation_scope": "operator-authored synthetic format scenarios with actual recorded local results; not a realistic benchmark or model score"}
        tasks.append(task)
        recording_adapters.append({"id": task["id"], "schema": "recovery_eval_response.v1", "input_sha256": digest(task["input"]),
                                   "messages": row["messages"][len(prefix):], "tool_outcomes": runtime_records,
                                   "scope": "source trajectory compatibility check only, never a model response"})
    if not tasks:
        raise ValueError("no heldout tool seed tasks")
    if output.exists():
        raise ValueError("tool seed task output must be new")
    output.mkdir(parents=True)
    task_path = output / "tasks.jsonl"
    task_path.write_bytes(b"".join(canonical(t) + b"\n" for t in tasks))
    inputs_path = output / "inputs.jsonl"
    inputs_path.write_bytes(b"".join(canonical({"id": t["id"], "group_id": t["group_id"], "split": t["split"], "input": t["input"], "input_sha256": digest(t["input"]), "environment": t["environment"]}) + b"\n" for t in tasks))
    # The source-evidence adapters prove oracle wiring; publish no model receipt.
    checks = [{"id": t["id"], "source_evidence_compatibility": grade_semantics(t, recording)} for t, recording in zip(tasks, recording_adapters)]
    (output / "source_evidence_compatibility.json").write_bytes(canonical({"scope": "source executed fixtures only; no model evaluated", "checks": checks}) + b"\n")
    manifest = {"schema": "recovery_eval_tasks_manifest.v1", "tasks": len(tasks), "split": split,
                "routes": dict(Counter(t["route"] for t in tasks)), "families": sorted({t["provenance"]["template_family"] for t in tasks}),
                "task_source_groups_disjoint_from_training": True, "training_group_ids": training_groups,
                "holdout_sha256": file_sha256(task_path), "inputs_sha256": file_sha256(inputs_path),
                "sources": {name: {"path": str(path.resolve()), "sha256": wanted[name]} for name, path in sources.items()},
                "source_evidence_compatible": sum(c["source_evidence_compatibility"]["status"] == "pass" for c in checks),
                "grader_sha256": file_sha256(Path(__file__)), "limitations": ["Operator-authored synthetic format coverage with genuine tool executions, not model intelligence or BFCL performance.",
                "Natural-language schema/final-value alternatives outside deterministic parsing require review.", "Model run must record new linked outcomes; copied fixture answers never count as model execution."]}
    manifest["manifest_sha256"] = digest(manifest)
    (output / "manifest.json").write_bytes(canonical(manifest) + b"\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    grade = sub.add_parser("grade")
    for key in ("tasks", "receipt", "responses", "output"):
        grade.add_argument("--" + key, type=Path, required=True)
    matched = sub.add_parser("compare")
    for key in ("before", "after", "output"):
        matched.add_argument("--" + key, type=Path, required=True)
    prepare = sub.add_parser("prepare-bfcl")
    for key in ("heldout_corpus", "training_corpus", "output"):
        prepare.add_argument("--" + key.replace("_", "-"), type=Path, required=True)
    for key in ("expected_heldout_sha256", "expected_training_sha256"):
        prepare.add_argument("--" + key.replace("_", "-"), required=True)
    seeds = sub.add_parser("prepare-tool-seeds")
    for key in ("ready", "execution_receipts", "seed_report", "output"):
        seeds.add_argument("--" + key.replace("_", "-"), type=Path, required=True)
    for key in ("expected_ready_sha256", "expected_receipts_sha256", "expected_report_sha256"):
        seeds.add_argument("--" + key.replace("_", "-"), required=True)
    seeds.add_argument("--split", choices=("validation", "test"), default="test")
    args = parser.parse_args()
    if args.mode == "grade":
        report = evaluate(list(iter_jsonl(args.tasks)), json.loads(args.receipt.read_text(encoding="utf-8")), list(iter_jsonl(args.responses)), holdout_sha256=file_sha256(args.tasks))
    elif args.mode == "compare":
        report = compare(json.loads(args.before.read_text(encoding="utf-8")), json.loads(args.after.read_text(encoding="utf-8")))
    else:
        fields = vars(args).copy()
        fields.pop("mode")
        result = prepare_bfcl(**fields) if args.mode == "prepare-bfcl" else prepare_tool_seeds(**fields)
        print(json.dumps(result, indent=2))
        return
    publish(report, args.output)
    print(json.dumps({"counts": report.get("counts"), "report_sha256": report.get("report_sha256"), "comparison_sha256": report.get("comparison_sha256")}, indent=2))


if __name__ == "__main__":
    main()
