"""
eval_intergrals.py

This script evaluates whether an LLM-generated solution for an integral is correct.
It:
  1. Prompts the LLM to solve an integral (provided as text) and return the antiderivative 
     wrapped in <answer> ... </answer> tags (if applicable) as a valid Python sympy expression.
  2. Extracts the candidate solution from within the <answer> tags (for evaluation) while also
     retaining the full raw LLM response.
  3. Computes the actual solution using sympy (based solely on the input integral).
  4. Verifies the candidate solution by differentiating it and comparing against the original integrand
     at several random points (to 5 decimal places).
  5. Returns the candidate solution, the computed solution, the full raw LLM response, and a correctness flag.
  
Usage example:
    python eval_intergrals.py
"""

import re
import random
import sympy as sp
import asyncio
from utils.inference import generate_text  # Ensure this function is available

def extract_candidate_solution(text: str) -> str:
    """
    Extracts a candidate solution from the LLM response by looking for text wrapped in <answer> and </answer> tags.
    
    If these tags are not found, the entire text is returned.
    """
    match = re.search(r"<answer>(.*?)</answer>", text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()

def preprocess_candidate_solution(solution: str) -> str:
    """
    Pre-processes the candidate solution string to convert common LaTeX-style expressions
    to Sympy-friendly syntax.
    
    This function:
      - Removes LaTeX math delimiters such as "\(" and "\)", as well as "$" signs.
      - Replaces "\arctan" with "atan" (and you can add more replacements as needed).
      - Removes any trailing "+ C".
    """
    # Remove LaTeX delimiters
    solution = solution.replace(r"\(", "").replace(r"\)", "")
    solution = solution.replace("$", "")
    # Replace LaTeX-style commands with sympy equivalents
    solution = solution.replace("\\arctan", "atan")
    solution = solution.replace("\\ln", "log")
    # Remove any occurrence of "+ C" or " + constant"
    solution = re.sub(r"\+?\s*C", "", solution)
    solution = solution.strip()
    return solution

async def evaluate_llm_solution(integral_text: str, num_tests: int = 5, model: str = "gpt-4o-mini", tol: float = 1e-5):
    """
    Uses the LLM to generate a solution for the given integral and then verifies its correctness.
    
    Parameters:
        integral_text: A string of the form "integrate(<integrand>, x)"
        num_tests: Number of random test points for verification.
        tol: Tolerance (to 5 decimal places) for numerical verification.
    
    Returns:
        A tuple (candidate_solution, computed_solution, raw_llm_solution, is_correct) where:
            - candidate_solution: The solution extracted from the LLM's output (from within <answer> tags) after preprocessing.
            - computed_solution: The antiderivative computed by sympy (or an error message if integration failed).
            - raw_llm_solution: The full raw response returned by the LLM.
            - is_correct: True if the candidate solution appears to be correct.
    """
    x = sp.symbols('x')
    
    # First, extract the integrand and compute the actual solution using sympy.
    integrand_pattern = r"integrate\((.+),\s*x\)"
    m = re.search(integrand_pattern, integral_text)
    if not m:
        raise ValueError("Could not extract the integrand from the provided integral text.")
    integrand_expr = sp.sympify(m.group(1))
    computed_expr = sp.integrate(integrand_expr, x)
    if computed_expr is None:
        computed_expr = sp.integrate(integrand_expr, x, meijerg=True)
    if computed_expr is None:
        computed_solution = "Could not compute a closed-form antiderivative."
    else:
        computed_solution = str(computed_expr)
    
    # Construct a prompt asking the LLM to solve the integral.
    # We now instruct the model to output the answer as a valid Python sympy expression (e.g. "atan(x)").
    prompt = (
        f"Solve the following integral and return ONLY your answer wrapped in <answer> and </answer> tags.\n"
        f"Output your answer as a valid Python sympy expression without LaTeX formatting (for example, use 'atan(x)' not '\\arctan(x)').\n\n"
        f"{integral_text}\n\n"
        "Do not include any additional commentary or constants of integration."
    )
    
    # Get the full LLM response.
    raw_llm_solution = await generate_text(model, prompt)
    
    # Extract and preprocess the candidate solution.
    candidate_solution = extract_candidate_solution(raw_llm_solution)
    print("Extracted Candidate Solution (raw):", candidate_solution)
    candidate_solution = preprocess_candidate_solution(candidate_solution)
    print("Preprocessed Candidate Solution:", candidate_solution)
    
    try:
        candidate_expr = sp.sympify(candidate_solution, locals={'C': 0})
    except Exception as e:
        print("Error parsing the candidate solution expression:", e)
        # Even if candidate fails, return the computed solution from the integrand.
        return candidate_solution, computed_solution, raw_llm_solution, False

    # Differentiate the candidate solution.
    derived_expr = sp.diff(candidate_expr, x)
    
    # Verify the candidate solution by comparing its derivative with the original integrand at random points.
    is_correct = True
    for _ in range(num_tests):
        test_val = random.uniform(-1000, 1000)
        try:
            dv = float(derived_expr.subs({x: test_val}).evalf())
            iv = float(integrand_expr.subs({x: test_val}).evalf())
        except Exception as conv_e:
            print(f"Skipping x={test_val} due to conversion error: {conv_e}")
            continue
        if abs(dv - iv) > tol:
            print(f"Mismatch at x={test_val}: derivative={dv} vs integrand={iv}")
            is_correct = False
            break

    return candidate_solution, computed_solution, raw_llm_solution, is_correct

# Main entry point.
if __name__ == "__main__":
    # Change the integral_text as needed.
    integral_text = "integrate(sec(x)**2, x)"
    
    # Run the evaluation in an asyncio event loop.
    candidate, computed, raw_llm, correct = asyncio.run(evaluate_llm_solution(integral_text, model="meta-llama/Llama-3.2-3B-Instruct"))
    
    print("\n--- Evaluation Summary ---")
    print("Input Integral:", integral_text)
    print("Extracted Candidate Solution:", candidate)
    print("Computed (Actual) Solution:", computed)
    print("Full LLM Response:\n", raw_llm)
    print("Solution Correct:", correct)
