"""
self_improve.py

An experimental pipeline that:
- Accepts an integral (or uses a preset one) and a set of difficulty targets.
- For each requested difficulty (e.g., easier, equivalent, or harder) it generates several variants.
- Each LLM prompt now generates up to 10 variants at once. If the user requests more than 10 variants,
  the work is split into multiple concurrent calls.
- Each variant is verified symbolically via Sympy.
- A second LLM prompt asks for a difficulty evaluation (easier/harder/equivalent) as a double-check.
- The antiderivative solution (i.e. the integration result) is computed, and it is evaluated at three random points.
- Points that produce complex values are skipped.
- Variants judged as "harder" are filtered out when not desired.
- All results are saved to "variants.json".
"""

import asyncio
import json
import math
import random
import re
from datetime import datetime

import sympy as sp

MODEL = "gpt-4o"

# Import our LLM-based generation function from the provided utils.inference module.
from utils.inference import generate_text


def verify_integral(integral_str: str) -> bool:
    """
    Verify the integral by checking that the derivative of the antiderivative equals the original integrand.
    Uses symbolic simplification to check for an exact zero difference.
    """
    x = sp.symbols('x')
    try:
        pattern = r"integrate\((.+),\s*x\)"
        match = re.search(pattern, integral_str)
        if not match:
            return False

        integrand_str = match.group(1)
        integrand = sp.sympify(integrand_str)
        antideriv = sp.integrate(integrand, x)
        diff_expr = sp.simplify(sp.diff(antideriv, x) - integrand)
        return diff_expr == 0
    except Exception as e:
        print("Error verifying integral:", e)
        return False


def compute_solution_and_evals(integral_str: str, num_points: int = 3, lower: float = -10, upper: float = 10, tol: float = 1e-6):
    """
    Given an integral string of the form "integrate(<integrand>, x)", compute the antiderivative (solution)
    and evaluate that solution at up to `num_points` random values of x.
    If an evaluation produces a complex number (with non-negligible imaginary part), it is skipped.
    
    Returns:
        solution_str: The antiderivative as a string.
        evaluations: A dictionary mapping each random x value (rounded) to the numerical evaluation.
    """
    x = sp.symbols('x')
    try:
        pattern = r"integrate\((.+),\s*x\)"
        match = re.search(pattern, integral_str)
        if not match:
            return None, {}
        integrand_str = match.group(1)
        integrand = sp.sympify(integrand_str)
        antideriv = sp.integrate(integrand, x)
        solution_str = str(antideriv)
        evaluations = {}
        attempts = 0
        max_attempts = num_points * 10
        while len(evaluations) < num_points and attempts < max_attempts:
            attempts += 1
            test_val = random.uniform(lower, upper)
            eval_val = antideriv.evalf(subs={x: test_val})
            if hasattr(eval_val, "as_real_imag"):
                re_val, im_val = eval_val.as_real_imag()
                if abs(im_val) < tol:
                    evaluations[round(test_val, 3)] = float(re_val)
            else:
                evaluations[round(test_val, 3)] = float(eval_val)
        return solution_str, evaluations
    except Exception as e:
        print("Error computing solution/evaluations:", e)
        return None, {}


def parse_variants(text: str) -> list:
    """
    Parse the LLM response text and extract a list of variant dictionaries.
    The expected format for each variant is:
    
    ====
    Variant <number>:
    Reasoning: <explanation>
    Variant: integrate(<integrand>, x)
    ====
    """
    variants = []
    # Split on delimiter "===="
    blocks = re.split(r"====\s*", text)
    for block in blocks:
        if "Variant:" in block and "Reasoning:" in block:
            reasoning_match = re.search(r"Reasoning:\s*(.*?)\s*Variant:", block, re.DOTALL)
            variant_match = re.search(r"Variant:\s*(integrate\([^,]+,\s*x\))", block)
            if variant_match:
                variant_expr = variant_match.group(1).strip()
                reasoning_text = reasoning_match.group(1).strip() if reasoning_match else ""
                variants.append({"reasoning": reasoning_text, "variant": variant_expr})
    return variants


async def process_single_variant(original_integral: str, difficulty: str, variant_data: dict) -> dict:
    """
    Process one variant dictionary:
      - Verify the integral.
      - If verified, compute its solution and numerical evaluations.
      - Ask the LLM for a difficulty evaluation.
    """
    variant_integral = variant_data.get("variant")
    if not variant_integral:
        return None

    verification = verify_integral(variant_integral)
    solution, evaluations = (None, {})
    if verification:
        solution, evaluations = compute_solution_and_evals(variant_integral)

    # Second prompt: evaluate the variant relative to the original.
    evaluation = "unknown"
    prompt_evaluation = (
        f"Original integral: {original_integral}\n"
        f"Variant integral: {variant_integral}\n"
        "Based on the changes, determine whether the variant is 'easier', 'harder', or 'equivalent' to the original. "
        "Answer with one word: easier, harder, or equivalent."
    )
    evaluation_response = await generate_text(MODEL, prompt_evaluation)
    if evaluation_response:
        evaluation = evaluation_response.strip().split()[0].lower()

    return {
        "original": original_integral,
        "requested_difficulty": difficulty,
        "variant": variant_integral,
        "reasoning": variant_data.get("reasoning"),
        "variant_response": None,  # the raw response text is not preserved here
        "verification_passed": verification,
        "evaluation": evaluation,
        "transformations_used": variant_data.get("transformations_used", []),
        "solution": solution,
        "evaluations": evaluations,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }


async def generate_variant_chunk(integral_str: str, difficulty: str, count: int) -> list:
    """
    Generate a chunk (up to 10) of variants in a single LLM call.
    The prompt instructs the LLM to produce `count` variants in the specified format.
    After receiving the response, each variant is parsed and then further processed.
    """
    # Choose transformation instructions from the list.
    transformations = TRANSFORMATIONS_BY_DIFFICULTY.get(difficulty.lower(), [])
    if not transformations:
        transformations = ["make a small change"]

    # Randomly choose a set of transformation ideas (for display in the prompt).
    num_choices = random.choice([3, 5])
    chosen_transforms = random.sample(transformations, min(num_choices, len(transformations)))
    transforms_text = " and ".join(chosen_transforms)

    # List of personas.
    personas = [
        "a calculus professor who loves elegant simplifications",
        "a creative mathematician who enjoys unusual substitutions",
        "a student who prefers working with polynomials and rational functions",
        "a theoretical physicist who likes trigonometric and exponential forms",
        "an engineer who favors practical, computational approaches",
        "a number theorist fascinated by prime numbers and rational coefficients",
        "a geometry enthusiast who thinks in terms of geometric transformations"
    ]
    personas_str = ", ".join(personas)

    prompt_variant = (
        f"Assume you can adopt various mathematical personas such as {personas_str}.\n\n"
        f"Given the integral: {integral_str}\n"
        f"Your task is to generate {count} variant(s) that are {difficulty} than the original.\n\n"
        f"1. Analyze the original integral and identify its key characteristics.\n"
        f"2. Consider the following transformation ideas: {transforms_text}. You may use them or devise your own modifications.\n"
        f"3. For each variant, provide a brief reasoning from the perspective of a distinct persona and then present the variant in valid Python sympy syntax.\n\n"
        "Return your answer in the following exact format for each variant:\n"
        "====\n"
        "Variant <number>:\n"
        "Reasoning: <your explanation>\n"
        "Variant: integrate(<integrand>, x)\n"
        "====\n"
        "Ensure each variant is clearly separated by the delimiter '===='."
    )

    response_text = await generate_text(MODEL, prompt_variant, temperature=1.0)
    parsed_variants = parse_variants(response_text)

    # Attach the chosen transforms to each variant (for record keeping).
    for variant in parsed_variants:
        variant["transformations_used"] = chosen_transforms

    # Process each variant concurrently (verify, compute solution, and LLM evaluation).
    tasks = [
        process_single_variant(integral_str, difficulty, variant)
        for variant in parsed_variants
    ]
    processed_variants = await asyncio.gather(*tasks)
    # Remove any None results.
    return [v for v in processed_variants if v is not None]


# Define the transformation instructions for each difficulty level.
TRANSFORMATIONS_BY_DIFFICULTY = {
    "easier": [
        "remove a complicated term",
        "simplify the denominator",
        "reduce an exponent",
        "lower a coefficient",
        "remove a factor",
        "eliminate a radical",
        "drop a subexpression",
        "simplify a trigonometric component",
        "convert a product to a simpler sum",
        "replace a complex fraction with a simpler one",
        "remove nested functions",
        "reduce the degree of a polynomial",
        "simplify composite trigonometric functions",
        "remove logarithmic terms",
        "eliminate absolute value terms",
        "reduce the number of terms in the expression",
        "replace transcendental functions with simpler algebraic ones",
        "change a function to an easier one"
    ],
    "equivalent": [
        "change coefficient values slightly",
        "alter constant terms",
        "modify an exponent slightly",
        "rewrite the integrand in a different form without changing overall complexity",
        "exchange similar functions (e.g., sin to cos)",
        "adjust parameters while keeping the integral equivalent",
        "rearrange the order of terms",
        "use trigonometric identities to rewrite expression",
        "substitute equivalent exponential forms",
        "change variables while maintaining complexity",
        "distribute terms differently",
        "factor common terms differently",
        "rewrite using alternate algebraic forms",
        "swap numerator and denominator with reciprocal",
        "use alternate but equivalent radical forms",
        "rewrite using different logarithmic properties",
        "apply algebraic identities that preserve complexity"
    ],
    "harder": [
        "introduce an additional polynomial factor",
        "increase the exponent",
        "add a non-linear term",
        "include a higher degree term",
        "insert a logarithmic factor",
        "complicate the denominator",
        "introduce a composite trigonometric function",
        "add a product of functions",
        "embed an extra constant factor that makes the expression less trivial"
    ]
}


async def process_integral(integral_str: str, difficulties: list, num_variants: int = 3) -> list:
    """
    Generate a batch of variants for the given integral and for each difficulty.
    If more than 10 variants are requested per difficulty, the work is split into multiple LLM calls.
    A buffer multiplier is used to allow for duplicates or filtering before trimming to the requested number.
    """
    final_results = []
    seen_variants = set()
    buffer_multiplier = 2  # request extra variants to allow filtering
    tasks = []

    for difficulty in difficulties:
        total_to_request = num_variants * buffer_multiplier
        num_chunks = math.ceil(total_to_request / 10)
        for i in range(num_chunks):
            count = 10 if (i < num_chunks - 1) else (total_to_request - 10 * (num_chunks - 1))
            tasks.append((difficulty, generate_variant_chunk(integral_str, difficulty, count)))

    # Launch all chunk calls concurrently.
    chunk_results = await asyncio.gather(*[t[1] for t in tasks])
    # Organize results by difficulty.
    difficulty_dict = {d: [] for d in difficulties}
    for idx, (difficulty, _) in enumerate(tasks):
        for variant in chunk_results[idx]:
            variant_expr = variant.get("variant")
            # Filter duplicates and (if not desired) variants that are evaluated as "harder"
            if (variant_expr 
                and variant_expr not in seen_variants 
                and not (variant.get("evaluation", "") == "harder" and difficulty != "harder")):
                seen_variants.add(variant_expr)
                difficulty_dict[difficulty].append(variant)
    
    for difficulty in difficulties:
        # Trim each difficulty's list to the requested number.
        final_results.extend(difficulty_dict[difficulty][:num_variants])
    
    return final_results


async def main():
    # Example: process a single integral.
    base_integral = "integrate(1/(x**2 - x + 1), x)"
    # Request variants for each difficulty level.
    difficulties = ["easier", "equivalent", "harder"]
    print("Processing integral:", base_integral)
    variants = await process_integral(base_integral, difficulties, num_variants=3)
    
    # Save the variants to a JSON file.
    with open("variants.json", "w") as outfile:
        json.dump(variants, outfile, indent=2)
    
    # Print a summary.
    for idx, v in enumerate(variants, start=1):
        print(f"\n--- Variant {idx} ---")
        print("Requested difficulty:", v["requested_difficulty"])
        print("Transformations used:", v["transformations_used"])
        print("Variant integral:", v["variant"])
        print("Verification passed:", v["verification_passed"])
        print("LLM evaluation:", v["evaluation"])
        print("Solution (antiderivative):", v["solution"])
        print("Evaluations at random points:", v["evaluations"])


if __name__ == "__main__":
    asyncio.run(main())
