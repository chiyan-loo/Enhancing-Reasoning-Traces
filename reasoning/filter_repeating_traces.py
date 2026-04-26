import json
import os
import re
from collections import Counter

def is_repeating(text, threshold=2, min_len=100):
    """
    Checks if there are repeating sentences or lines in the text.
    
    Args:
        text: The text to check.
        threshold: How many times a sentence can appear before being considered repeating.
        min_len: Minimum length of a sentence to be considered for repetition check.
                 Short sentences are ignored.
    """
    # Split by common sentence endings and newlines
    # This regex splits by . ! ? followed by space/newline, or just newlines
    sentences = re.split(r'(?<=[.!?])\s+|\n+', text)
    
    # Clean and filter sentences
    cleaned_sentences = [s.strip() for s in sentences if len(s.strip()) >= min_len]
    
    if not cleaned_sentences:
        return False
    
    counts = Counter(cleaned_sentences)
    
    for sentence, count in counts.items():
        if count >= threshold:
            return True
            
    return False

def filter_jsonl(input_path, output_path):
    print(f"Filtering {input_path}...")
    
    total_samples = 0
    filtered_samples = 0
    
    if not os.path.exists(input_path):
        print(f"Error: Input file {input_path} not found.")
        return

    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            total_samples += 1
            try:
                sample = json.loads(line)
                # Check for repetition in 'model_response'
                trace = sample.get('model_response', '')
                
                if is_repeating(trace):
                    filtered_samples += 1
                    continue
                
                f_out.write(json.dumps(sample) + '\n')
            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON on line {total_samples}")
                continue

    print(f"Finished.")
    print(f"Total samples: {total_samples}")
    print(f"Filtered out: {filtered_samples} ({filtered_samples/total_samples*100:.2f}%)")
    print(f"Remaining samples: {total_samples - filtered_samples}")
    print(f"Filtered file saved to: {output_path}")

if __name__ == "__main__":
    # Default paths based on the requested file
    input_file = "reasoning/MATH_traces_Qwen2.5-3B-Instruct_alt_3alt_2000_correct.jsonl"
    output_file = "reasoning/MATH_traces_Qwen2.5-3B-Instruct_alt_3alt_2000_correct_filtered_2.jsonl"
    
    filter_jsonl(input_file, output_file)
