Text Summarizer Tool
A powerful command-line text summarization tool that leverages Hugging Face’s `transformers` library and the pre-trained `facebook/bart-large-cnn` model to generate concise, readable summaries from lengthy documents.

---

##  Features

* 📄 Summarizes long text files using state-of-the-art NLP models.
* 🧩 Automatically chunks long documents to fit the model’s input limitations.
* ⚠️ Handles model initialization and errors gracefully.
* 📦 Uses the `transformers` library for easy integration with pre-trained models.
* 🧪 No training required – just inference.

---

## 📁 File Overview

### `text_summarizer.py`

Main script for summarizing a plain-text file. It includes:

* `initialize_summarizer()`: Loads the tokenizer and model.
* `chunk_text(text, tokenizer, max_token_length=1024)`: Splits long texts into manageable chunks based on token length.
* `summarize_text(text, summarizer, tokenizer)`: Generates and concatenates summaries of the chunks.
* `main()`: CLI-based workflow for reading a text file and printing the summary.

---

##  Requirements

* Python 3.7+
* Install dependencies with:

```bash
pip install transformers
```

> Optional but recommended: Set up a virtual environment.

---

## Usage

1. Save your input text in a `.txt` file.
2. Run the script:

```bash
python text_summarizer.py
```

3. Enter the path to your `.txt` file when prompted.

4. View the first 500 characters of the original text and the generated summary.

---

## 🧩 How It Works

### 1. Model Initialization

The script uses `facebook/bart-large-cnn`, a fine-tuned sequence-to-sequence model specifically for summarization tasks.

```python
from transformers import pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
```

### 2. Tokenization & Chunking

To handle documents longer than 1024 tokens, the script chunks the input text intelligently:

```python
tokens = tokenizer.tokenize(text)
chunks = [tokens[i:i + max_len] for i in range(0, len(tokens), max_len)]
```

### 3. Inference

Each chunk is summarized individually and concatenated into a final result:

```python
summary = summarizer(chunk, max_length=150, min_length=40)[0]['summary_text']
```

---

## ⚠️ Limitations

* Currently designed for plain text input (`.txt` files).
* Summary quality may degrade for very fragmented input or unusual formatting.
* Does not support streaming input or batch processing yet.

---

## Example

Input:

```
The Industrial Revolution, which took place from the 18th to 19th centuries...

sample document is uploaded article.txt
```

Output:

```
The Industrial Revolution was a transformative era marked by significant technological advancements, urbanization, and changes in labor systems.
```

---

## Future Improvements

* GUI or web interface.
* PDF or DOCX input support.
* Batch summarization and export features.
* Language detection and multilingual summarization.

---

## 🧑‍💻 Author

Developed by Wagmare Sanjana
📧 Contact: wagmaresanjana5@gmail.com
---

## 📜 License

This project is licensed under the MIT License
