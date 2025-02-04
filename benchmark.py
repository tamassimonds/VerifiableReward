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
  
Additionally, this script loads a list of integrals from base_questions.py, benchmarks the model’s performance
across many integrals, prints progress as well as an overall accuracy summary, and optionally saves a new Python file
with only the integrals that were answered incorrectly.

Usage example:
    python eval_intergrals.py
"""

import re
import random
import sympy as sp
import asyncio
from utils.inference import generate_text  # Ensure this function is available
from questions.intergration_mit_bee_questions import BASE_QUESTIONS as QUESTIONS

# ---------------- Configuration ----------------
SAVE_INCORRECT_QUESTIONS = True  # Set to True to save a new file with the integrals answered incorrectly.
INCORRECT_FILENAME = "incorrect_mit_bee_questions.py"  # Name of the file to save incorrect questions.

# ---------------- Helper Functions ----------------

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
    return solution.strip()

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
    
    # Extract the integrand and compute the actual solution using sympy.
    integrand_pattern = r"integrate\((.+),\s*x\)"
    m = re.search(integrand_pattern, integral_text)
    if not m:
        raise ValueError("Could not extract the integrand from the provided integral text.")
    try:
        integrand_expr = sp.sympify(m.group(1))
    except Exception as e:
        return "", "Error parsing integrand: " + str(e), "", False
    print("INTERGRATING")
    computed_expr = sp.integrate(integrand_expr, x)
    print("COMPUTED")
    if computed_expr is None:
        computed_expr = sp.integrate(integrand_expr, x, meijerg=True)
    if computed_expr is None:
        computed_solution = "Could not compute a closed-form antiderivative."
    else:
        computed_solution = str(computed_expr)
    
    # Construct a prompt asking the LLM to solve the integral.
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
    candidate_solution = preprocess_candidate_solution(candidate_solution)
    
    try:
        candidate_expr = sp.sympify(candidate_solution, locals={'C': 0})
    except Exception as e:
        # If candidate parsing fails, return false.
        return candidate_solution, computed_solution, raw_llm_solution, False

    # Differentiate the candidate solution.
    derived_expr = sp.diff(candidate_expr, x)
    
    # Verify the candidate solution by comparing its derivative with the original integrand at random points.
    is_correct = True
    for _ in range(num_tests):
        test_val = random.uniform(-100, 100)  # use a moderate range
        try:
            dv = float(derived_expr.subs({x: test_val}).evalf())
            iv = float(integrand_expr.subs({x: test_val}).evalf())
        except Exception as conv_e:
            # If an evaluation error occurs, skip this test point.
            continue
        if abs(dv - iv) > tol:
            is_correct = False
            break

    return candidate_solution, computed_solution, raw_llm_solution, is_correct

# -------------- Benchmarking Code --------------

async def benchmark_integrals(batch_size: int = 10, model: str = "gpt-4o-mini"):
    total = len(QUESTIONS)
    correct_count = 0
    results = []
    print(f"Starting benchmark on {total} integrals in batches of {batch_size}...\n")

    # Process the integrals in batches.
    for batch_start in range(0, total, batch_size):
        batch = QUESTIONS[batch_start:batch_start+batch_size]
        tasks = []
        for integral_text in batch:
            tasks.append(evaluate_llm_solution(integral_text, model=model))
        
        # Await the results concurrently for this batch.
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process and print the results for the current batch.
        for i, res in enumerate(batch_results):
            integral_text = batch[i]
            if isinstance(res, Exception):
                candidate, computed, raw_response, is_correct = ("", "", str(res), False)
            else:
                candidate, computed, raw_response, is_correct = res
            
            if is_correct:
                correct_count += 1
            results.append({
                "integral": integral_text,
                "candidate": candidate,
                "computed": computed,
                "raw_response": raw_response,
                "correct": is_correct
            })
            
            # Active printing for progress.
            print("--------------------------------------------------")
            print(f"Integral: {integral_text}")
            print(f"Candidate Solution: {candidate}")
            print(f"Computed Solution: {computed}")
            print(f"LLM Correct: {is_correct}")
            print("--------------------------------------------------\n")
        current_accuracy = (correct_count / (batch_start + len(batch))) * 100
        print(f"Batch {batch_start // batch_size + 1}: Current accuracy = {current_accuracy:.2f}%\n")
    
    overall_accuracy = (correct_count / total) * 100
    print("\n--- Benchmarking Complete ---")
    print(f"Total integrals tested: {total}")
    print(f"Total correct: {correct_count}")
    print(f"Overall accuracy: {overall_accuracy:.2f}%")
    return results

def save_incorrect_questions(results, filename: str):
    """
    Given the benchmarking results, extract the integrals that were answered incorrectly
    and save them to a Python file containing a list variable `QUESTIONS_INCORRECT`.
    """
    incorrect_integrals = [res["integral"] for res in results if not res["correct"]]
    content = (
        "# This file is auto-generated and contains the integrals that were answered incorrectly.\n"
        "QUESTIONS_INCORRECT = [\n"
    )
    for integral in incorrect_integrals:
        content += f"    {repr(integral)},\n"
    content += "]\n"
    
    try:
        with open(filename, "w") as f:
            f.write(content)
        print(f"Incorrect questions saved to {filename}")
    except Exception as e:
        print(f"Failed to write incorrect questions to {filename}: {e}")

# ---------------- Main Entry Point ----------------

if __name__ == "__main__":
    async def main():
        results = await benchmark_integrals(batch_size=10, model="gpt-4o")
        if SAVE_INCORRECT_QUESTIONS:
            save_incorrect_questions(results, INCORRECT_FILENAME)
    
    asyncio.run(main())
