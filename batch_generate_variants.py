import asyncio
import json
from datetime import datetime
from pathlib import Path
import random

from base_datasets.train_questions import TRAIN_QUESTIONS2 as QUESTIONS
from generate_variants import process_integral

VARIANTS_PER_INTEGRAL = 30  # Number of variants to generate for each integral
DIFFICULTIES = ["equivalent", "harder"]  # Desired variant difficulties

async def main():
    # Create output directory if it doesn't exist
    output_dir = Path("variant_results")
    output_dir.mkdir(exist_ok=True)

    all_results = []
    total_integrals = len(QUESTIONS)
    shuffled_questions = QUESTIONS.copy()
    random.shuffle(shuffled_questions)

    # Process each integral sequentially
    for idx, integral in enumerate(shuffled_questions, 1):
        # Process one integral at a time
        result = await process_integral(
            integral,
            difficulties=DIFFICULTIES,
            num_variants=VARIANTS_PER_INTEGRAL
        )
        all_results.append(result)

        # Save the result immediately after processing this integral
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = output_dir / f"variants_question_{idx}_{timestamp}.json"
        with open(filename, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Question {idx} result saved to {filename}")

    # Print summary statistics
    total_variants = sum(len(result) for result in all_results)
    print(f"\nSummary:")
    print(f"Total integrals processed: {total_integrals}")
    print(f"Total variants generated: {total_variants}")
    print(f"Average variants per integral: {total_variants / total_integrals:.2f}")

if __name__ == "__main__":
    asyncio.run(main())
