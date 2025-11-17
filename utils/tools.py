def word_count_checker(text: str) -> int:
    print(f"Calculating word count for text of length {text} {len(text)} {len(text.split())}", flush=True)
    return len(text.split())