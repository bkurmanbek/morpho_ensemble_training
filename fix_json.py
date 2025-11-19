"""
JSON Diagnostic and Repair Script
==================================

Diagnoses and attempts to fix JSON parsing errors in data files.
"""

import json
import re
import sys


def diagnose_json(filepath: str):
    """Diagnose JSON parsing issues"""

    print(f"Diagnosing: {filepath}")
    print("=" * 60)

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    print(f"File size: {len(content):,} characters")

    # Try to parse
    try:
        data = json.loads(content)
        print(f"✓ JSON is valid! Contains {len(data)} items")
        return True
    except json.JSONDecodeError as e:
        print(f"✗ JSON Error: {e.msg}")
        print(f"  Line: {e.lineno}, Column: {e.colno}")
        print(f"  Position: {e.pos}")

        # Show context around error
        start = max(0, e.pos - 200)
        end = min(len(content), e.pos + 200)

        context = content[start:end]
        error_in_context = e.pos - start

        print(f"\nContext around error (position {e.pos}):")
        print("-" * 60)
        print(context[:error_in_context] + ">>>ERROR HERE<<<" + context[error_in_context:])
        print("-" * 60)

        # Try to identify the specific line
        lines = content[:e.pos].split('\n')
        print(f"\nLast 5 lines before error:")
        for i, line in enumerate(lines[-5:], start=len(lines)-4):
            print(f"  {i}: {line[:100]}{'...' if len(line) > 100 else ''}")

        return False


def find_all_errors(filepath: str):
    """Find multiple JSON errors by attempting incremental parsing"""

    print(f"\nSearching for all errors in: {filepath}")
    print("=" * 60)

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    errors = []

    # Check for common issues

    # 1. Unescaped control characters
    control_chars = re.finditer(r'[\x00-\x1f]', content)
    for match in control_chars:
        if match.group() not in '\n\r\t':
            line_num = content[:match.start()].count('\n') + 1
            errors.append(f"Line {line_num}: Unescaped control character at position {match.start()}")

    # 2. Unescaped quotes in strings (basic check)
    # This is a simplified check - real issues might be more complex

    # 3. Check for truncation
    stripped = content.rstrip()
    if not stripped.endswith(']') and not stripped.endswith('}'):
        errors.append(f"File appears truncated - doesn't end with ] or }}")
        errors.append(f"Last 50 chars: {stripped[-50:]}")

    # 4. Check bracket balance
    open_brackets = content.count('[')
    close_brackets = content.count(']')
    open_braces = content.count('{')
    close_braces = content.count('}')

    if open_brackets != close_brackets:
        errors.append(f"Bracket mismatch: [ = {open_brackets}, ] = {close_brackets}")
    if open_braces != close_braces:
        errors.append(f"Brace mismatch: {{ = {open_braces}, }} = {close_braces}")

    if errors:
        print("Found potential issues:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("No obvious structural issues found")

    return errors


def attempt_fix(filepath: str, output_path: str = None):
    """Attempt to fix common JSON issues"""

    if output_path is None:
        output_path = filepath.replace('.json', '_fixed.json')

    print(f"\nAttempting to fix: {filepath}")
    print(f"Output will be saved to: {output_path}")
    print("=" * 60)

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    original_len = len(content)

    # Fix 1: Remove/escape control characters
    def escape_control_chars(s):
        result = []
        for char in s:
            if ord(char) < 32 and char not in '\n\r\t':
                result.append(f'\\u{ord(char):04x}')
            else:
                result.append(char)
        return ''.join(result)

    content = escape_control_chars(content)

    # Fix 2: Try to fix truncation by adding closing brackets
    stripped = content.rstrip()

    # Count brackets to determine what's missing
    open_brackets = stripped.count('[') - stripped.count(']')
    open_braces = stripped.count('{') - stripped.count('}')

    if open_brackets > 0 or open_braces > 0:
        print(f"File appears truncated. Missing: {open_braces} }}, {open_brackets} ]")

        # Add closing brackets
        closing = '}' * open_braces + ']' * open_brackets
        content = stripped + closing
        print(f"Added closing: {closing}")

    # Fix 3: Try to fix unterminated strings
    # This is tricky - we'll try a simple approach

    # Test if fixed
    try:
        data = json.loads(content)
        print(f"✓ Fix successful! JSON now valid with {len(data)} items")

        # Save fixed version
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"Saved fixed JSON to: {output_path}")
        return True

    except json.JSONDecodeError as e:
        print(f"✗ Simple fixes didn't work: {e.msg}")
        print(f"  Error at line {e.lineno}, position {e.pos}")

        # Try more aggressive fix for unterminated string
        print("\nAttempting aggressive fix for unterminated string...")

        # Find the problematic position and try to close the string
        error_pos = e.pos

        # Look backwards to find the opening quote
        search_start = max(0, error_pos - 1000)
        segment = content[search_start:error_pos]

        # Find last unclosed quote
        in_string = False
        last_quote_pos = -1

        for i, char in enumerate(segment):
            if char == '"' and (i == 0 or segment[i-1] != '\\'):
                if not in_string:
                    last_quote_pos = search_start + i
                in_string = not in_string

        if in_string and last_quote_pos >= 0:
            # Insert closing quote at error position
            content = content[:error_pos] + '"' + content[error_pos:]
            print(f"Inserted closing quote at position {error_pos}")

            # Try again
            try:
                data = json.loads(content)
                print(f"✓ Aggressive fix successful! JSON now valid with {len(data)} items")

                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)

                print(f"Saved fixed JSON to: {output_path}")
                return True

            except json.JSONDecodeError as e2:
                print(f"✗ Aggressive fix also failed: {e2.msg}")

        return False


def extract_valid_portion(filepath: str, output_path: str = None):
    """Extract valid portion of JSON up to the error"""

    if output_path is None:
        output_path = filepath.replace('.json', '_partial.json')

    print(f"\nExtracting valid portion from: {filepath}")
    print("=" * 60)

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find the last complete object
    # Strategy: find the last "}," and close the array there

    try:
        data = json.loads(content)
        print("File is already valid!")
        return
    except json.JSONDecodeError as e:
        error_pos = e.pos

    # Search backwards from error for last complete item
    search_content = content[:error_pos]

    # Find last complete object (ends with },)
    last_complete = search_content.rfind('},')

    if last_complete > 0:
        # Extract up to and including that object, then close array
        valid_content = search_content[:last_complete+1] + ']'

        try:
            data = json.loads(valid_content)
            print(f"✓ Extracted {len(data)} valid items")

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            print(f"Saved partial data to: {output_path}")

            # Calculate what was lost
            total_objects = content.count('"word":')
            print(f"Original had ~{total_objects} items, extracted {len(data)}")
            print(f"Lost ~{total_objects - len(data)} items ({(total_objects - len(data))/total_objects*100:.1f}%)")

            return True

        except json.JSONDecodeError as e:
            print(f"✗ Could not extract valid portion: {e.msg}")
            return False
    else:
        print("Could not find last complete object")
        return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fix_json.py <json_file> [action]")
        print("\nActions:")
        print("  diagnose  - Diagnose JSON issues (default)")
        print("  fix       - Attempt to fix the JSON")
        print("  extract   - Extract valid portion")
        print("\nExample:")
        print("  python fix_json.py all_structured_kazakh_data.json diagnose")
        print("  python fix_json.py all_structured_kazakh_data.json fix")
        sys.exit(1)

    filepath = sys.argv[1]
    action = sys.argv[2] if len(sys.argv) > 2 else 'diagnose'

    if action == 'diagnose':
        diagnose_json(filepath)
        find_all_errors(filepath)
    elif action == 'fix':
        attempt_fix(filepath)
    elif action == 'extract':
        extract_valid_portion(filepath)
    else:
        print(f"Unknown action: {action}")
        sys.exit(1)
