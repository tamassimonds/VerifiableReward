"""
Script to generate easier variants of all incorrect questions using the existing generate_variants.py
infrastructure. Saves each question and its variants to individual JSON files in dataset/<question_id>.json
"""

import asyncio
import json
import os
import hashlib
import uuid  # Add this import at the top with other imports
import random  # Add this import at the top
from incorrect_questions import BASE_QUESTIONS_INCORRECT
from generate_variants import process_integral
from datetime import datetime  # Add this import

BATCH_SIZE = 10  # Number of questions to process concurrently

def generate_question_id(question: str) -> str:
    """Generate a random unique ID for a question."""
    return str(uuid.uuid4())[:8]  # Using first 8 chars of UUID4

async def save_variants(question: str, variants: list):
    """Save variants for a question to its own JSON file."""
    os.makedirs("dataset", exist_ok=True)
    
    question_id = generate_question_id(question)
    
    # Count valid variants, skipping error entries
    valid_variants = 0
    for v in variants:
        if isinstance(v, dict) and "error" not in v:
            if v.get("variant") and v.get("verification_passed"):
                valid_variants += 1
    
    data = {
        "question_id": question_id,
        "original": question,
        "variants": variants,
        "num_valid_variants": valid_variants
    }
    
    filename = f"dataset/question_{question_id}.json"
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"Saved variants for question {question_id} to {filename}")
    return question_id

async def process_batch(questions, batch_num):
    """Process a batch of questions concurrently."""
    # Process questions sequentially with delay
    results = []
    question_ids = []
    for i, question in enumerate(questions):
        question_id = generate_question_id(question)
        print(f"Processing question {question_id}: {question}")
        try:
            if i > 0:  # Add delay for all requests except the first one
                await asyncio.sleep(1)  # 1 second delay
            result = await process_integral(
                question,
                difficulties=["easier"],
                num_variants=30
            )
        except Exception as e:
            print(f"Error processing question {question_id}: {e}")
            result = [{"error": str(e)}]
        
        # Save immediately after processing
        question_id = await save_variants(question, result)
        results.append(result)
        question_ids.append(question_id)
    
    return results, question_ids

async def generate_all_variants():
    """Generate variants for all incorrect questions in batches."""
    total_valid_variants = 0
    processed_questions = {}
    
    remaining_questions = BASE_QUESTIONS_INCORRECT.copy()
    total_batches = (len(BASE_QUESTIONS_INCORRECT) + BATCH_SIZE - 1) // BATCH_SIZE
    
    while remaining_questions:
        batch_size = min(BATCH_SIZE, len(remaining_questions))
        batch = random.sample(remaining_questions, batch_size)
        for q in batch:
            remaining_questions.remove(q)
            
        current_batch = total_batches - (len(remaining_questions) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"\nProcessing batch {current_batch}/{total_batches}")
        
        results, question_ids = await process_batch(batch, current_batch)
        
        # Count valid variants and store question mapping
        for result, question_id, question in zip(results, question_ids, batch):
            processed_questions[question_id] = question
            if isinstance(result, list):  # Make sure result is a list
                valid_variants = sum(
                    1 for v in result 
                    if isinstance(v, dict) 
                    and "error" not in v 
                    and v.get("variant") 
                    and v.get("verification_passed", False)
                )
                total_valid_variants += valid_variants
        
        print(f"Completed batch. Total valid variants so far: {total_valid_variants}")

    print(f"\nGeneration complete. Total valid variants: {total_valid_variants}")
    
    # Create a summary file
    summary = {
        "total_questions": len(BASE_QUESTIONS_INCORRECT),
        "total_valid_variants": total_valid_variants,
        "question_mapping": processed_questions,
        "timestamp": datetime.utcnow().isoformat() + "Z"  # Fixed datetime usage
    }
    
    with open("dataset/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    asyncio.run(generate_all_variants()) 