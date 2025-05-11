import os
from pathlib import Path
import textwrap
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

def initialize_summarizer():
    """Initialize and return the summarization pipeline."""
    try:
        model_name = "facebook/bart-large-cnn"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        return pipeline("summarization", model=model, tokenizer=tokenizer), tokenizer
    except Exception as e:
        print(f"Error initializing summarizer: {e}")
        raise

def chunk_text(text, tokenizer, max_token_length=1024):
    """Split long text into chunks that fit the model's max token length."""
    # Tokenize the entire text first
    tokens = tokenizer.tokenize(text)
    
    chunks = []
    for i in range(0, len(tokens), max_token_length):
        chunk_tokens = tokens[i:i + max_token_length]
        chunk_text = tokenizer.convert_tokens_to_string(chunk_tokens)
        chunks.append(chunk_text)
    
    return chunks

def summarize_text(text, summarizer, tokenizer):
    """Generate summary for the given text."""
    try:
        chunks = chunk_text(text, tokenizer)
        summaries = []
        
        for chunk in chunks:
            summary = summarizer(
                chunk,
                max_length=150,
                min_length=40,
                do_sample=False,
                truncation=True  # Ensure truncation is enabled
            )[0]['summary_text']
            summaries.append(summary)
        
        return " ".join(summaries)
    except Exception as e:
        print(f"Error during summarization: {e}")
        raise

def main():
    print("=" * 50)
    print("🧠 Advanced Text Summarizer (Fixed Version)")
    print("=" * 50)
    
    try:
        # Setup summarizer
        summarizer, tokenizer = initialize_summarizer()
        print("Model loaded successfully")

        # Get input file
        input_path = input("Enter the path to the text file: ").strip()
        input_path = Path(input_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"File not found: {input_path}")
        
        # Read and process file
        with open(input_path, 'r', encoding='utf-8') as file:
            article = file.read()

        print("\n📄 Original Text (First 500 characters):")
        print(textwrap.shorten(article, width=500, placeholder="..."))

        print("\n⏳ Generating summary...")
        summary = summarize_text(article, summarizer, tokenizer)

        print("\n📝 Generated Summary:")
        print("\n" + summary)

    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
    except Exception as e:
        print(f"⚠️ An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()