"""
batch_generate_variants.py

This script processes multiple integrals from incorrect_questions.py in batches,
generating easier and equivalent variants for each integral using the generate_variants.py
functionality.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict
from base_questions import BASE_QUESTIONS as QUESTIONS
from generate_variants import process_integral
import random
BATCH_SIZE = 1  # Number of integrals to process concurrently
VARIANTS_PER_INTEGRAL = 30  # Number of variants to generate for each integral
DIFFICULTIES = ["easier", "equivalent"]  # We want easier and equivalent variants

async def process_batch(integrals: List[str]) -> List[Dict]:
    """Process a batch of integrals concurrently."""
    tasks = [
        process_integral(
            integral,
            difficulties=DIFFICULTIES,
            num_variants=VARIANTS_PER_INTEGRAL
        )
        for integral in integrals
    ]
    results = await asyncio.gather(*tasks)
    return results

async def main():
    # Create output directory if it doesn't exist
    output_dir = Path("variant_results")
    output_dir.mkdir(exist_ok=True)
    
    # Process integrals in batches
    all_results = []
    total_integrals = len(QUESTIONS)
    shuffled_questions = QUESTIONS.copy()  # Create a copy of the list
    random.shuffle(shuffled_questions)  # Shuffle the copy
    for i in range(0, total_integrals, BATCH_SIZE):
        batch = shuffled_questions[i:i + BATCH_SIZE]
        print(f"Processing batch {i//BATCH_SIZE + 1}/{(total_integrals + BATCH_SIZE - 1)//BATCH_SIZE}")
        print(f"Integrals {i+1}-{min(i+BATCH_SIZE, total_integrals)} of {total_integrals}")
        
        batch_results = await process_batch(batch)
        all_results.extend(batch_results)
        
        # Save intermediate results after each batch
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        batch_filename = output_dir / f"variants_batch_{i//BATCH_SIZE + 1}_{timestamp}.json"
        with open(batch_filename, "w") as f:
            json.dump(batch_results, f, indent=2)
        
        print(f"Batch results saved to {batch_filename}")
    
    # Save final combined results
    final_filename = output_dir / f"variants_all_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    with open(final_filename, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nAll results saved to {final_filename}")
    
    # Print summary statistics
    total_variants = sum(len(result) for result in all_results)
    print(f"\nSummary:")
    print(f"Total integrals processed: {total_integrals}")
    print(f"Total variants generated: {total_variants}")
    print(f"Average variants per integral: {total_variants/total_integrals:.2f}")

if __name__ == "__main__":
    asyncio.run(main()) 