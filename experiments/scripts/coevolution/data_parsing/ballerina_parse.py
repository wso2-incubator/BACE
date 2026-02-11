import json
import re


def convert_raw_to_jsonl(input_file, output_file) -> None:
    with open(input_file, "r") as f:
        raw_data = json.load(f)

    with open(output_file, "w") as f:
        for idx, item in enumerate(raw_data):
            prompt = item["prompt"]
            test_code = item["test"]

            # 1. Robust Function Name Extraction
            # Handles generic return types and potential spacing issues
            func_match = re.search(
                r"function\s+(\w+)\s*\(.*?\)\s*returns\s+.*?(?:\{|=)", prompt, re.DOTALL
            )
            func_name = func_match.group(1) if func_match else "unknown"

            # 2. Cleaner Starter Code Extraction
            # Finds the last occurrence of "function funcName" to avoid grabbing comments
            lines = prompt.split("\n")
            start_index = -1
            for i in range(len(lines) - 1, -1, -1):
                if lines[i].strip().startswith(f"function {func_name}"):
                    start_index = i
                    break

            if start_index != -1:
                # Only take the function signature line and close it
                # We intentionally drop previous comments to keep starter code clean
                starter_code = lines[start_index].rstrip().rstrip("{").strip() + " {\n}"
            else:
                starter_code = f"function {func_name}(...) {{\n}}"

            # 3. Enhanced Public Test Extraction
            public_tests = []
            # Regex now supports:
            # // func(...) returns val
            # // func(...) should return val
            # // func(...) -> val
            comment_regex = (
                f"//\\s+{func_name}\\((.*?)\\)\\s*(?:returns|should return|->)\\s*(.*)"
            )
            for match in re.finditer(comment_regex, prompt):
                public_tests.append(
                    {"input": match.group(1).strip(), "output": match.group(2).strip()}
                )

            # 4. Advanced Private Test Extraction
            private_tests = []

            # Helper to resolve variable references in test code
            # e.g., int[] input = [1, 2]; ... func(input)
            def resolve_variable(var_name, code_chunk):
                # Search backwards for "Type var_name = value;"
                var_regex = f"=\\s*(.*?);\n.*?{re.escape(var_name)}"
                # This is a simple heuristic; a full parser would be better but overkill here
                # We look for the assignment immediately preceding usage
                lines = code_chunk.split("\n")
                for line in lines:
                    if f"{var_name} =" in line or f"{var_name}=" in line:
                        # Extract value after equals
                        parts = line.split("=", 1)
                        if len(parts) > 1:
                            return parts[1].strip().rstrip(";")
                return var_name

            # Split by individual test functions usually marked by @test:Config
            # We assume one assertion per test function usually, but handle multiple
            test_blocks = re.split(
                r"@test:Config\s*\{.*?\}", test_code, flags=re.DOTALL
            )

            for block in test_blocks:
                if not block.strip():
                    continue

                # Find all calls to the function
                # captured groups: 1=args
                call_matches = list(re.finditer(f"{func_name}\\((.*?)\\)", block))

                for call_match in call_matches:
                    args = call_match.group(1).strip()

                    # If arg looks like a variable (no commas, braces, brackets), try to resolve it
                    if re.match(r"^[a-zA-Z_]\w*$", args):
                        args = resolve_variable(args, block)

                    expected = None

                    # Look for assertion associated with this call
                    # Case A: boolean return (assertTrue/False)
                    if "assertTrue" in block and call_match.start() < block.find(
                        "assertTrue"
                    ):
                        expected = "true"
                    elif "assertFalse" in block and call_match.start() < block.find(
                        "assertFalse"
                    ):
                        expected = "false"

                    # Case B: assertEquals(result, expected)
                    # We search for assertEquals occurring AFTER the function call
                    else:
                        assert_match = re.search(
                            r"test:assertEquals\(\s*\w+,\s*(.*?)(?:,.*)?\);",
                            block[call_match.end() :],
                        )
                        if assert_match:
                            expected = assert_match.group(1).strip()

                    if expected:
                        private_tests.append({"input": args, "output": expected})

            # 5. Construct Object
            problem = {
                "question_title": func_name,
                "question_id": f"HumanEval/{idx}",
                "question_content": prompt,
                "starter_code": starter_code,
                "public_test_cases": public_tests,
                "private_test_cases": private_tests,
            }

            f.write(json.dumps(problem) + "\n")


# Usage:
convert_raw_to_jsonl("data/humaneval_bal_raw.json", "data/humaneval_ballerina.jsonl")
