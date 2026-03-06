"""
This logic is largely copied from the Hendrycks' MATH release (math_equivalence), and borrowed from:
- https://github.com/microsoft/ProphetNet/tree/master/CRITIC
- https://github.com/openai/prm800k
- https://github.com/microsoft/ToRA/blob/main/src/eval/grader.py
- https://github.com/deepseek-ai/DeepSeek-Math/blob/main/evaluation/eval/eval_utils.py
"""

import re
import regex
import multiprocessing
from math import isclose
from typing import Union
from collections import defaultdict

from sympy import simplify, N
from sympy.parsing.sympy_parser import parse_expr
from sympy.parsing.latex import parse_latex
from latex2sympy2 import latex2sympy

from .parser import strip_string
# from parser import choice_answer_clean, strip_string

from .math_normalization import check_sympy_equivalence

import signal
from concurrent.futures import ThreadPoolExecutor


def normalize_latex_expression(expr):
    """
    Normalize LaTeX expressions to handle common formatting differences.
    """
    if not isinstance(expr, str):
        return expr
        
    # Convert to string and strip
    expr = str(expr).strip()
    
    # Normalize different minus signs
    expr = expr.replace('−', '-')  # Unicode minus to ASCII minus
    expr = expr.replace('–', '-')  # En dash to ASCII minus
    expr = expr.replace('—', '-')  # Em dash to ASCII minus
    
    # Normalize fraction commands (\tfrac, \dfrac -> \frac)
    expr = re.sub(r'\\[td]frac', r'\\frac', expr)
    
    # Handle missing braces in fractions (e.g., \frac32 -> \frac{3}{2})
    # Pattern: \frac followed by digits or simple expressions without braces
    def fix_frac_braces(match):
        frac_content = match.group(1)
        # Split into numerator and denominator
        # Handle cases like: 32, 3}{2, etc.
        if '{' in frac_content and '}' in frac_content:
            return match.group(0)  # Already has braces, don't change
        
        # Handle common patterns like \frac32, \frac 32, etc.
        frac_content = frac_content.strip()
        
        # Common fraction patterns
        common_fractions = {
            '12': ('1', '2'), '32': ('3', '2'), '52': ('5', '2'), '72': ('7', '2'),
            '13': ('1', '3'), '23': ('2', '3'), '43': ('4', '3'), '53': ('5', '3'),
            '14': ('1', '4'), '34': ('3', '4'), '54': ('5', '4'), '74': ('7', '4'),
            '15': ('1', '5'), '25': ('2', '5'), '35': ('3', '5'), '45': ('4', '5'),
        }
        
        if frac_content in common_fractions:
            num, denom = common_fractions[frac_content]
            return f"\\frac{{{num}}}{{{denom}}}"
        
        # Try to split by finding reasonable break points
        for i in range(1, len(frac_content)):
            num_part = frac_content[:i]
            denom_part = frac_content[i:]
            if num_part and denom_part:
                # Check if this is a reasonable split (numbers, simple expressions)
                if (num_part.replace('-', '').replace('+', '').replace('.', '').isdigit() and
                    denom_part.replace('-', '').replace('+', '').replace('.', '').isdigit()):
                    return f"\\frac{{{num_part}}}{{{denom_part}}}"
        
        # Fallback: if it looks like a single digit fraction, split it
        if len(frac_content) == 2 and frac_content.isdigit():
            return f"\\frac{{{frac_content[0]}}}{{{frac_content[1]}}}"
        
        return match.group(0)  # Return original if can't parse
    
    # Handle \frac without braces and also \tfrac, \dfrac, etc.
    expr = re.sub(r'\\[td]?frac([^{}\s]+)', fix_frac_braces, expr)
    expr = re.sub(r'\\frac\s+([^{}\s]+)', fix_frac_braces, expr)
    
    # Normalize spacing commands
    spacing_normalizations = [
        (r'\\,', ' '),          # \, -> space
        (r'\\;', ' '),          # \; -> space  
        (r'\\:', ' '),          # \: -> space
        (r'\\!', ''),           # \! -> nothing (negative space)
        (r'\\quad', ' '),       # \quad -> space
        (r'\\qquad', ' '),      # \qquad -> space
        (r'\\ ', ' '),          # \ -> space
    ]
    
    for pattern, replacement in spacing_normalizations:
        expr = re.sub(pattern, replacement, expr)
    
    # Normalize different bracket styles
    bracket_normalizations = [
        (r'\\left\(', '('),
        (r'\\right\)', ')'),
        (r'\\left\[', '['),
        (r'\\right\]', ']'),
        (r'\\left\\{', '{'),
        (r'\\right\\}', '}'),
        (r'\\Bigl?\(', '('),
        (r'\\Bigr?\)', ')'),
        (r'\\bigl?\(', '('),
        (r'\\bigr?\)', ')'),
        (r'\\Big[lr]?\[', '['),
        (r'\\Big[lr]?\]', ']'),
        (r'\\big[lr]?\[', '['),
        (r'\\big[lr]?\]', ']'),
    ]
    
    for pattern, replacement in bracket_normalizations:
        expr = re.sub(pattern, replacement, expr)
    

    
    # Normalize different text commands
    text_normalizations = [
        (r'\\text\{([^}]+)\}', r'\1'),  # Remove \text{}
        (r'\\mathrm\{([^}]+)\}', r'\1'),  # Remove \mathrm{}
        (r'\\mathbf\{([^}]+)\}', r'\1'),  # Remove \mathbf{}
        (r'\\textbf\{([^}]+)\}', r'\1'),  # Remove \textbf{}
    ]
    
    for pattern, replacement in text_normalizations:
        expr = re.sub(pattern, replacement, expr)
    

    
    # Normalize multiple spaces to single space
    expr = re.sub(r'\s+', ' ', expr)
    
    # Remove \circ and degree symbols entirely (don't convert them)
    expr = re.sub(r'\^\s*\\circ', '', expr)  # Remove ^\circ
    expr = re.sub(r'\^\\circ', '', expr)     # Remove ^\circ
    expr = re.sub(r'°', '', expr)            # Remove degree symbol
    
    # Remove trailing punctuation that doesn't affect mathematical meaning
    expr = expr.rstrip('.,;')
    
    return expr.strip()

def timeout_handler(signum, frame):
    raise TimeoutError("Function execution timed out")


def choice_answer_clean(pred: str):
    pred = pred.strip("\n").rstrip(".").rstrip("/").strip(" ").lstrip(":")
    # Clean the answer based on the dataset
    tmp = re.findall(r"\b(A|B|C|D|E)\b", pred.upper())
    if tmp:
        pred = tmp
    else:
        pred = [pred.strip().strip(".")]
    pred = pred[-1]
    # Remove the period at the end, again!
    pred = pred.rstrip(".").rstrip("/")
    return pred


def parse_digits(num):
    num = regex.sub(",", "", str(num))
    try:
        return float(num)
    except:
        if num.endswith("%"):
            num = num[:-1]
            if num.endswith("\\"):
                num = num[:-1]
            try:
                return float(num) / 100
            except:
                pass
    return None


def is_digit(num):
    # paired with parse_digits
    return parse_digits(num) is not None


def str_to_pmatrix(input_str):
    input_str = input_str.strip()
    matrix_str = re.findall(r"\{.*,.*\}", input_str)
    pmatrix_list = []

    for m in matrix_str:
        m = m.strip("{}")
        pmatrix = r"\begin{pmatrix}" + m.replace(",", "\\") + r"\end{pmatrix}"
        pmatrix_list.append(pmatrix)

    return ", ".join(pmatrix_list)


single_choice_patterns = [
    r"^\(A\)", r"^\(B\)", r"^\(C\)", r"^\(D\)", r"^\(E\)",  # (A) (B) (C) (D) (E)
    r"^A\.", r"^B\.", r"^C\.", r"^D\.", r"^E\.",            # A. B. C. D. E.
    r"^A\)", r"^B\)", r"^C\)", r"^D\)", r"^E\)",            # A) B) C) D) E)
    r"^\*\*A\*\*", r"^\*\*B\*\*", r"^\*\*C\*\*", r"^\*\*D\*\*", r"^\*\*E\*\*",  # **A** **B** **C** **D** **E**
    r"^A:", r"^B:", r"^C:", r"^D:", r"^E:",                 # A: B: C: D: E:
]


def math_equal(
    prediction: Union[bool, float, str],
    reference: Union[float, str],
    include_percentage: bool = True,
    is_close: bool = True,
    timeout: bool = True,
    depth: int = 0,
    max_depth: int = 5
) -> bool:
    """
    Exact match of math if and only if:
    1. numerical equal: both can convert to float and are equal
    2. symbolic equal: both can convert to sympy expression and are equal
    """
    
    if depth > max_depth:
        return False

    if prediction is None or reference is None:
        return False
    
    # Early normalization of LaTeX expressions to handle formatting differences
    if isinstance(prediction, str) and isinstance(reference, str):
        normalized_pred = normalize_latex_expression(prediction)
        normalized_ref = normalize_latex_expression(reference)
        
        # Check if normalized versions are equal
        if normalized_pred.strip().lower() == normalized_ref.strip().lower():
            return True
            
        # Also try the original logic with normalized versions
        prediction = normalized_pred
        reference = normalized_ref
    
    if str(prediction.strip().lower()) == str(reference.strip().lower()):
        return True
    if (
        reference in ["A", "B", "C", "D", "E"]
        and choice_answer_clean(prediction) == reference
    ):
        return True
    
    for pattern in single_choice_patterns:
        if regex.match(pattern, prediction):
            # Remove the pattern from the beginning of the prediction and strip the result
            prediction_cleaned = regex.sub(pattern, "", prediction, count=1).strip()
            # Recursively call math_equal to check if the cleaned prediction matches the reference
            if math_equal(prediction_cleaned, reference, include_percentage, is_close, timeout=timeout, depth=depth+1, max_depth=max_depth):
                return True
            
    if "," in prediction and "," in reference:
        # Split by comma and normalize each part
        pred_parts = [normalize_latex_expression(part.strip()) for part in prediction.split(",")]
        ref_parts = [normalize_latex_expression(part.strip()) for part in reference.split(",")]

        if len(pred_parts) == len(ref_parts):
            # Try both ordered and unordered comparison
            # First try ordered (maintains original order)
            if all(
                math_equal(pred_parts[i], ref_parts[i], include_percentage, is_close, timeout=timeout, depth=depth+1, max_depth=max_depth)
                for i in range(len(pred_parts))
            ):
                return True
                
            # Then try unordered (sort both lists and compare)
            pred_parts_sorted = sorted(pred_parts)
            ref_parts_sorted = sorted(ref_parts)
            
            if all(
                math_equal(pred_parts_sorted[i], ref_parts_sorted[i], include_percentage, is_close, timeout=timeout, depth=depth+1, max_depth=max_depth)
                for i in range(len(pred_parts_sorted))
            ):
                return True
    
    

    try:  # 1. numerical equal
        if is_digit(prediction) and is_digit(reference):
            prediction = parse_digits(prediction)
            reference = parse_digits(reference)
            # number questions
            if include_percentage:
                gt_result = [reference / 100, reference, reference * 100]
            else:
                gt_result = [reference]
            for item in gt_result:
                try:
                    if is_close:
                        if numeric_equal(prediction, item):
                            return True
                    else:
                        if item == prediction:
                            return True
                except Exception:
                    continue
            return False
    except:
        pass

    if not prediction and prediction not in [0, False]:
        return False

    # 2. symbolic equal
    reference = str(reference).strip()
    prediction = str(prediction).strip()

    ## pmatrix (amps)
    if "pmatrix" in prediction and not "pmatrix" in reference:
        reference = str_to_pmatrix(reference)

    ## deal with [], (), {}
    pred_str, ref_str = prediction, reference
    if (
        prediction.startswith("[")
        and prediction.endswith("]")
        and not reference.startswith("(")
    ) or (
        prediction.startswith("(")
        and prediction.endswith(")")
        and not reference.startswith("[")
    ):
        pred_str = pred_str.strip("[]()")
        ref_str = ref_str.strip("[]()")
    for s in ["{", "}", "(", ")"]:
        ref_str = ref_str.replace(s, "")
        pred_str = pred_str.replace(s, "")
    if pred_str.lower() == ref_str.lower():
        return True
    
    
    ## unordered [a, b] vs. [c, d]
    # if (
    # regex.match(r"(\(|\[).+(\)|\])", prediction) is not None
    # and regex.match(r"(\(|\[).+(\)|\])", reference) is not None
    # ):
    #     pred_parts = prediction[1:-1].split(",")
    #     ref_parts = reference[1:-1].split(",")

    #     if len(pred_parts) == len(ref_parts):
    #         # 对两个列表的每个元素进行排序后逐个比较，使用 math_equal 递归判断是否相等
    #         pred_parts_sorted = sorted(pred_parts, key=lambda x: x.strip())
    #         ref_parts_sorted = sorted(ref_parts, key=lambda x: x.strip())
            
    #         if all(
    #             [
    #                 math_equal(pred_parts_sorted[i], ref_parts_sorted[i], include_percentage, is_close, timeout=timeout)
    #                 for i in range(len(pred_parts_sorted))
    #             ]
    #         ):
    #             return True
    

    ## [a, b] vs. [c, d], return a==c and b==d
    if (
        regex.match(r"(\(|\[).+(\)|\])", prediction) is not None
        and regex.match(r"(\(|\[).+(\)|\])", reference) is not None
    ):
        pred_parts = prediction[1:-1].split(",")
        ref_parts = reference[1:-1].split(",")
        if len(pred_parts) == len(ref_parts):
            if all(
                [
                    math_equal(
                        pred_parts[i], ref_parts[i], include_percentage, is_close, timeout=timeout, depth=depth+1, max_depth=max_depth
                    )
                    for i in range(len(pred_parts))
                ]
            ):
                return True
    if (
        (
            prediction.startswith("\\begin{pmatrix}")
            or prediction.startswith("\\begin{bmatrix}")
        )
        and (
            prediction.endswith("\\end{pmatrix}")
            or prediction.endswith("\\end{bmatrix}")
        )
        and (
            reference.startswith("\\begin{pmatrix}")
            or reference.startswith("\\begin{bmatrix}")
        )
        and (
            reference.endswith("\\end{pmatrix}") or reference.endswith("\\end{bmatrix}")
        )
    ):
        pred_lines = [
            line.strip()
            for line in prediction[
                len("\\begin{pmatrix}") : -len("\\end{pmatrix}")
            ].split("\\\\")
            if line.strip()
        ]
        ref_lines = [
            line.strip()
            for line in reference[
                len("\\begin{pmatrix}") : -len("\\end{pmatrix}")
            ].split("\\\\")
            if line.strip()
        ]
        matched = True
        if len(pred_lines) == len(ref_lines):
            for pred_line, ref_line in zip(pred_lines, ref_lines):
                pred_parts = pred_line.split("&")
                ref_parts = ref_line.split("&")
                if len(pred_parts) == len(ref_parts):
                    if not all(
                        [
                            math_equal(
                                pred_parts[i],
                                ref_parts[i],
                                include_percentage,
                                is_close,
                                timeout=timeout,
                                depth=depth+1, 
                                max_depth=max_depth
                            )
                            for i in range(len(pred_parts))
                        ]
                    ):
                        matched = False
                        break
                else:
                    matched = False
                if not matched:
                    break
        else:
            matched = False
        if matched:
            return True

    if prediction.count("=") == 1 and reference.count("=") == 1:
        pred = prediction.split("=")
        pred = f"{pred[0].strip()} - ({pred[1].strip()})"
        ref = reference.split("=")
        ref = f"{ref[0].strip()} - ({ref[1].strip()})"
        if symbolic_equal(pred, ref) or symbolic_equal(f"-({pred})", ref):
            return True
    elif (
        prediction.count("=") == 1
        and "=" not in reference
    ):
        if math_equal(
            prediction.split("=")[1], reference, include_percentage, is_close, timeout=timeout, depth=depth+1, max_depth=max_depth
        ):
            return True

    if timeout:
        # For Windows, call symbolic_equal directly to avoid multiprocessing issues
        import platform
        if platform.system() == "Windows":
            try:
                if symbolic_equal(prediction, reference):
                    return True
            except Exception:
                # If symbolic_equal fails, continue to return False
                pass
        else:
            # For non-Windows, use the multiprocessing timeout approach
            if call_with_timeout(symbolic_equal_process, prediction, reference):
                return True
        # try:
        #     if call_with_timeout(symbolic_equal, prediction, reference, timeout=1):
        #         return True
        # except TimeoutError:
        #     return False
    else:
        if symbolic_equal(prediction, reference):
            return True

    return False


def math_equal_process(param):
    return math_equal(param[-2], param[-1])


def numeric_equal(prediction: float, reference: float):
    # Note that relative tolerance has significant impact
    # on the result of the synthesized GSM-Hard dataset
    # if reference.is_integer():
    #     return isclose(reference, round(prediction), abs_tol=1e-4)
    # else:
    # prediction = round(prediction, len(str(reference).split(".")[-1]))
    
    # return isclose(reference, prediction, rel_tol=1e-4)
    return isclose(reference, prediction, abs_tol=1e-4)


def symbolic_equal(a, b):
    # Apply normalization before parsing
    if isinstance(a, str):
        a = normalize_latex_expression(a)
    if isinstance(b, str):
        b = normalize_latex_expression(b)
    
    def _parse(s):
        for f in [parse_latex, parse_expr, latex2sympy]:
            try:
                return f(s.replace("\\\\", "\\"))
            except:
                try:
                    return f(s)
                except:
                    pass
        return s

    a = _parse(a)
    b = _parse(b)

    # direct equal
    try:
        if str(a) == str(b) or a == b:
            return True
    except:
        pass

    # simplify equal
    try:
        if a.equals(b) or simplify(a - b) == 0:
            return True
    except:
        pass

    # equation equal
    try:
        if (abs(a.lhs - a.rhs)).equals(abs(b.lhs - b.rhs)):
            return True
    except:
        pass

    try:
        if numeric_equal(float(N(a)), float(N(b))):
            return True
    except:
        pass

    # matrix
    try:
        # if a and b are matrix
        if a.shape == b.shape:
            _a = a.applyfunc(lambda x: round(x, 3))
            _b = b.applyfunc(lambda x: round(x, 3))
            if _a.equals(_b):
                return True
    except:
        pass

    return False


def symbolic_equal_process(a, b, output_queue):
    result = symbolic_equal(a, b)
    output_queue.put(result)


def call_with_timeout(func, *args, timeout=3, **kwargs):
    """
    Call a function with timeout using multiprocessing, with Windows compatibility fixes
    """
    import platform
    
    # On Windows, use a simpler approach to avoid handle issues
    if platform.system() == "Windows":
        try:
            # For Windows, we'll use a simplified timeout approach
            # This avoids the complex multiprocessing handle issues
            import signal
            import time
            
            def timeout_handler(signum, frame):
                raise TimeoutError("Function execution timed out")
            
            # Try the function directly first (most cases should work)
            try:
                return func(*args, **kwargs)
            except Exception:
                # If direct call fails, return False (safer fallback)
                return False
                
        except Exception:
            # If anything goes wrong, return False as a safe fallback
            return False
    
    # For non-Windows systems, use the original multiprocessing approach
    try:
        output_queue = multiprocessing.Queue()
        process_args = args + (output_queue,)
        process = multiprocessing.Process(target=func, args=process_args, kwargs=kwargs)
        process.start()
        process.join(timeout)

        if process.is_alive():
            process.terminate()
            process.join()
            return False

        return output_queue.get()
    except Exception:
        # If multiprocessing fails, return False as fallback
        return False

# def call_with_timeout(func, *args, timeout=1, **kwargs):
#     # Register the signal function handler
#     signal.signal(signal.SIGALRM, timeout_handler)
#     # Set the alarm
#     signal.alarm(timeout)

#     try:
#         result = func(*args, **kwargs)
#         signal.alarm(0)  # Disable the alarm if function completes in time
#         return result
#     except TimeoutError:
#         return False
#     finally:
#         # Ensure the alarm is disabled
#         signal.alarm(0)

# def call_with_timeout(func, *args, timeout=1, **kwargs):
#     with ThreadPoolExecutor(max_workers=1) as executor:
#         future = executor.submit(func, *args, **kwargs)
#         try:
#             result = future.result(timeout=timeout)  # Wait for result with a timeout
#             return result
#         except TimeoutError:
#             return False  # Timeout occurred



def check_is_correct(pred, gt, timeout=True):
    return math_equal(strip_string(pred), strip_string(gt), timeout=timeout)


def math_equal_simple(pred, gt):
    pred = strip_string(pred)
    gt = strip_string(gt)
    flag = False
    
    try:
        pred_expr = latex2sympy(pred)
    except:
        pred_expr = pred
        flag = True
        
    try:  
        gt_expr = latex2sympy(gt)
    except:
        gt_expr = gt
        flag = True
        
    if flag == True:
        return pred == gt
    
    try:
        if abs(N(pred_expr) - N(gt_expr)) <= 1e-5:
            return True
    except:
        return False

    return False
    

def check_is_correct_simple(pred, gt, timeout=True):
    if timeout:
        return call_with_timeout(math_equal_simple, pred, gt, timeout=1)
    else:
        return math_equal_simple(pred, gt)

def _test_math_equal():
    print("Testing improvements to handle LaTeX formatting differences:")
    
    # Test cases focusing on the three main issues: \, spacing, \circ/degree, and \frac braces
    test_cases = [
        # Missing braces in fractions
        ("-\\frac 32", "−\\tfrac{3}{2}"),
        ("-3,\\;\\frac32", "-3, \\frac{3}{2}"),
        
        # Spacing differences (\, removal)
        ("\\left( -\\frac{3}{10}, -\\frac{10}{3} \\right)", "\\Bigl(-\\frac{3}{10},\\,-\\frac{10}{3}\\Bigr)"),
        ("(4,-2,3)", "(4,\\,-2,\\;3)"),
        ("4,14,6, 15", "4,\\,6,\\,14,\\,15"),
        ("(-3,4)", "(-3,\\,4)"),
        ("(5,-5)", "(5,\\,-5)"),
        
        # Degree symbols (should be removed entirely)
        ("45, 135", "45°,135°"),
        ("45, 135", "45^\\circ, 135^\\circ"),
        
        # Fraction braces with spacing
        ("\\left( \\frac{1}{\\sqrt{2}}, \\frac{1}{\\sqrt{2}} \\right)", "\\Bigl(\\tfrac1{\\sqrt2},\\,\\tfrac1{\\sqrt2}\\Bigr)"),
    ]
    
    for i, (pred, gt) in enumerate(test_cases, 1):
        result = math_equal(strip_string(pred), strip_string(gt), timeout=False)
        print(f"Test {i}: {'PASS' if result else 'FAIL'}")
        print(f"  Prediction: {pred}")
        print(f"  Ground Truth: {gt}")
        if not result:
            print(f"  Normalized Pred: {normalize_latex_expression(pred)}")
            print(f"  Normalized GT: {normalize_latex_expression(gt)}")
        print()


if __name__ == "__main__":
    _test_math_equal()
