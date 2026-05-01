# Project Full Flow Explanation

Here is the **full end-to-end flow of your project**, in viva-friendly order, with **what each library, function, and tool does** and **what its responsibility is**.

This explanation follows the order the project actually works:

1. training flow
2. inference flow
3. role of every important library and tool

## 1. Project Goal

Your project builds a system for:

- **input**: Computer Science research paper body text
- **output 1**: generated abstract
- **output 2**: extracted keywords

The project has **2 main notebooks**:

- `NLP_Project_15_epochs.ipynb`
  This is the **training notebook**.
- `Project_26_infer.ipynb`
  This is the **inference notebook**.

## 2. Training Notebook Full Flow

File:
`NLP_Project_15_epochs.ipynb`

### Step 1: Install required libraries

Cell 1 runs:

```python
!pip install -q transformers datasets accelerate evaluate sentencepiece scikit-learn beautifulsoup4 lxml rouge_score
```

Responsibility:
This installs all the packages needed for:

- transformer model loading
- dataset creation
- training
- evaluation
- HTML parsing
- keyword extraction

### Step 2: Import libraries

Cell 2 imports:

```python
import re
import requests
from bs4 import BeautifulSoup
from datasets import Dataset
import evaluate
import numpy as np
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
```

Responsibility of each:

- `re`
  Used for text cleaning with regular expressions.
- `requests`
  Used to download web pages from arXiv.
- `BeautifulSoup`
  Used to parse HTML pages and extract text.
- `Dataset`
  Used to convert Python records into train/validation datasets.
- `evaluate`
  Used to compute ROUGE scores.
- `numpy`
  Used for array handling and metric processing.
- `AutoTokenizer`
  Loads the tokenizer for the transformer model.
- `AutoModelForSeq2SeqLM`
  Loads the transformer summarization model.
- `DataCollatorForSeq2Seq`
  Prepares batches correctly for seq2seq training.
- `Seq2SeqTrainer`
  Handles the fine-tuning process.
- `Seq2SeqTrainingArguments`
  Stores training hyperparameters.

### Step 3: Define project configuration

Still in Cell 2:

```python
ARXIV_CATEGORIES = ['cs.AI', 'cs.CL', 'cs.LG', 'cs.CV']
MODEL_NAME = 'sshleifer/distilbart-cnn-12-6'
MAX_SOURCE_TOKENS = 1024
MAX_TARGET_TOKENS = 160
TRAIN_SAMPLE_LIMIT = 40
VAL_SAMPLE_LIMIT = 10
MAX_PER_CATEGORY = 15
OUTPUT_DIR = '/content/cs_transformer_summarizer'
```

Responsibility:
These variables control:

- which papers are collected
- which transformer model is used
- how long source and target text can be
- how many samples are used
- where the trained model will be saved

Important meaning:

- `ARXIV_CATEGORIES`
  Limits the project to Computer Science papers.
- `MODEL_NAME`
  Chooses the pretrained summarization model.
- `MAX_SOURCE_TOKENS`
  Maximum token length for paper body text.
- `MAX_TARGET_TOKENS`
  Maximum token length for abstract labels.
- `OUTPUT_DIR`
  Save location of the fine-tuned model.

### Step 4: Collect paper links

Cell 3 defines:

```python
def fetch_recent_paper_links(categories, max_per_category=10):
```

Responsibility:
This function finds paper HTML links for the chosen categories.

What it does:

- loops over each arXiv category
- calls the arXiv API
- parses the XML response
- extracts the paper ID for each entry
- converts each paper ID into an arXiv HTML link

In the final working inference flow, this uses **arXiv API/XML-based paper discovery** instead of list-page scraping.

The goal is:

- get links to paper HTML pages

### Step 5: Download HTML of a paper

Cell 3 defines:

```python
def download_arxiv_html(arxiv_html_url):
```

Responsibility:
This function downloads the HTML page of a paper.

What it does:

- sends HTTP request using `requests`
- parses HTML using `BeautifulSoup`
- removes script and style tags
- returns clean soup object

### Step 6: Extract abstract and body text

Cell 3 defines:

```python
def extract_abstract_and_body(soup):
```

Responsibility:
This is one of the most important functions.

What it does:

- finds the abstract section from HTML
- extracts the abstract text
- removes abstract from the soup
- collects all main paragraph text from the body
- joins paragraphs into one body string

Output:

- `abstract`
- `body_text`

Important viva point:
This means the project is training on:

- `input = body_text`
- `target = abstract`

Not abstract-to-abstract rewriting.

### Step 7: Clean body text

Cell 3 defines:

```python
def clean_text(text):
```

Responsibility:
Removes noise from the extracted text.

What it does:

- removes URLs
- removes extra spaces
- normalizes text

### Step 8: Build training records

Cell 3 defines:

```python
def build_records(sample_limit):
```

Responsibility:
This function creates the actual training examples.

What it does:

- gets paper links
- downloads each paper
- extracts abstract and body
- cleans the body
- stores each paper as:

```python
{
    'link': link,
    'source_text': body_text,
    'target_summary': abstract,
}
```

This is the core dataset creation step.

Then this line runs it:

```python
records = build_records(TRAIN_SAMPLE_LIMIT + VAL_SAMPLE_LIMIT)
```

So now you have data records ready.

### Step 9: Split into train and validation

Cell 4:

```python
train_records = records[:TRAIN_SAMPLE_LIMIT]
val_records = records[TRAIN_SAMPLE_LIMIT:TRAIN_SAMPLE_LIMIT + VAL_SAMPLE_LIMIT]
```

Responsibility:

- training set teaches the model
- validation set checks performance during training

### Step 10: Convert to dataset objects

Cell 4:

```python
train_dataset = Dataset.from_list(train_records)
val_dataset = Dataset.from_list(val_records)
```

Responsibility:
Converts plain Python records into Hugging Face dataset objects.

### Step 11: Load tokenizer and model

Cell 4:

```python
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
```

Responsibility:

- `tokenizer` converts text into token IDs
- `model` is the transformer that learns summarization

### Step 12: Tokenize inputs and targets

Cell 4 defines:

```python
def preprocess_batch(batch):
```

Responsibility:
Converts raw text into model-ready tokens.

What it does:

- tokenizes `batch['source_text']` as model input
- tokenizes `batch['target_summary']` as labels
- stores labels inside `model_inputs['labels']`

Then:

```python
tokenized_train = train_dataset.map(preprocess_batch, ...)
tokenized_val = val_dataset.map(preprocess_batch, ...)
```

This creates the tokenized dataset.

### Step 13: Prepare batching and metric tool

Cell 4:

```python
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
rouge = evaluate.load('rouge')
```

Responsibility:

- `data_collator`
  Makes batches correctly for seq2seq training.
- `rouge`
  Loads the ROUGE metric tool.

### Step 14: Define metric computation

Cell 4 defines:

```python
def compute_metrics(eval_pred):
```

Responsibility:
Computes summary evaluation after generation.

What it does:

- decodes predicted token IDs back to text
- decodes label token IDs back to text
- computes:
  - ROUGE-1
  - ROUGE-2
  - ROUGE-L

### Step 15: Define training configuration

Cell 5:

```python
training_args = Seq2SeqTrainingArguments(...)
```

Responsibility:
Stores the model training settings.

Examples:

- learning rate
- batch size
- number of epochs
- evaluation strategy
- save strategy

### Step 16: Create trainer

Cell 5:

```python
trainer = Seq2SeqTrainer(...)
```

Responsibility:
This object manages:

- training loop
- evaluation loop
- batching
- generation during validation
- saving

### Step 17: Train the model

Cell 5:

```python
train_result = trainer.train()
metrics = trainer.evaluate()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
```

Responsibility:

- `trainer.train()`
  Fine-tunes the transformer on your paper-to-abstract dataset.
- `trainer.evaluate()`
  Measures validation performance.
- `save_model`
  Saves the fine-tuned model.
- `save_pretrained`
  Saves the tokenizer too.

At this point, training is complete.

### Step 18: Quick sample generation

Cell 6 defines:

```python
def generate_summary(text):
```

Responsibility:
Generates a summary for any input paper text.

What it does:

- tokenizes source text
- calls `model.generate(...)`
- decodes generated token IDs into text

Then it runs sample outputs on validation papers.

This is just a quick training-side sanity check.

## 3. Inference Notebook Full Flow

File:
`Project_26_infer.ipynb`

### Step 1: Install required libraries

Cell 1 installs the same core libraries.

Reason:
Inference still needs:

- transformer model
- tokenizer
- HTML parsing
- evaluation
- TF-IDF keyword extraction

### Step 2: Import libraries

Cell 2 imports:

```python
import re
import requests
from bs4 import BeautifulSoup
from collections import Counter
import evaluate
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
```

New important thing here:

- `Counter`
  Used to count frequent terms for reference keyword extraction.
- `TfidfVectorizer`
  Used for keyword extraction.

### Step 3: Define inference configuration

Cell 2 sets:

```python
ARXIV_CATEGORIES
MODEL_DIR
FALLBACK_MODEL
MAX_SOURCE_TOKENS
MAX_TARGET_TOKENS
MAX_KEYWORDS
NUM_SAMPLE_PAPERS
```

Responsibility:

- `MODEL_DIR`
  Where the trained model is loaded from
- `FALLBACK_MODEL`
  Backup summarization model if saved model is missing
- `MAX_KEYWORDS`
  Number of keywords to return
- `NUM_SAMPLE_PAPERS`
  Number of validation sample papers to test

### Step 4: Reuse paper fetching and extraction functions

Cell 3 defines again:

- `fetch_recent_paper_links`
- `download_arxiv_html`
- `extract_abstract_and_body`
- `clean_text`

Responsibility:
Same as training notebook, but now used for inference-time paper collection.

### Step 5: Keyword extraction function

Cell 4 defines:

```python
def extract_keywords_tfidf(text, top_n=MAX_KEYWORDS):
```

Responsibility:
Extracts keywords from paper body text.

What it does:

- splits text into sentence-like segments
- runs TF-IDF vectorization
- scores terms
- filters bad short terms
- ranks keywords
- returns top terms

This is your keyword module.

### Step 6: Reference term extraction

Cell 4 defines:

```python
def extract_reference_terms(text, top_n=MAX_KEYWORDS):
```

Responsibility:
Creates simple reference keywords from text using term frequency.

Used for:

- extracting reference terms from original abstract
- extracting terms from generated abstract

### Step 7: Keyword metric calculation

Cell 4 defines:

```python
def compute_set_metrics(predicted_items, reference_items):
```

Responsibility:
Measures overlap between two keyword sets.

What it computes:

- precision
- recall
- F1
- overlap terms

Used for:

- extracted keywords vs reference keywords
- extracted keywords vs generated abstract terms

### Step 8: Load trained model

Cell 5:

```python
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
```

Responsibility:
Loads the fine-tuned summarization model and tokenizer.

If this fails:

- it loads the fallback pretrained model

### Step 9: Summary generation function

Cell 6 defines:

```python
def generate_summary(source_text):
```

Responsibility:
This is the actual summarization function used in inference.

What it does:

- tokenizes source text
- generates summary with beam search
- decodes output into readable abstract

### Step 10: Full single-paper inference pipeline

Cell 6 defines:

```python
def summarize_cs_paper(link):
```

Responsibility:
This is the main end-to-end inference function.

What it does step by step:

1. download paper HTML
2. extract abstract and body
3. clean body text
4. generate abstract using transformer
5. extract keywords using TF-IDF
6. extract reference keywords from original abstract
7. extract generated abstract terms
8. compute ROUGE
9. compute keyword overlap metrics
10. package everything into one result dictionary

This function is the heart of inference.

### Step 11: Pretty print output

Cell 6 defines:

```python
def print_result(result, index):
```

Responsibility:
Prints sample output in demo-friendly format.

It shows:

- sample paper number
- paper link
- generated abstract
- extracted keywords
- original abstract
- reference keywords
- generated abstract terms
- ROUGE scores
- keyword F1 scores

### Step 12: Run inference on multiple papers

Cell 7:

```python
paper_links = fetch_recent_paper_links(...)
sample_links = paper_links[:NUM_SAMPLE_PAPERS]
results = []
```

Then loop:

```python
for index, link in enumerate(sample_links, start=1):
```

Responsibility:
Runs full inference paper by paper.

### Step 13: Collect final averages

Cell 7 computes:

```python
avg_rouge1
avg_rouge2
avg_rougeL
avg_keyword_ref_f1
avg_keyword_summary_f1
```

Responsibility:
Creates final project validation results.

Then it prints:

- average summary metrics
- average keyword metrics

This becomes your final demo and viva result.

## 4. The Simplest End-to-End Explanation

If you want to explain the whole project in one flow:

### Training

1. collect Computer Science papers from arXiv
2. download each paper’s HTML page
3. extract the original abstract and body text
4. clean the body text
5. use body text as input and abstract as target
6. tokenize both with one transformer tokenizer
7. fine-tune a transformer summarization model
8. save the trained model

### Inference

1. load the saved transformer model
2. fetch new CS papers
3. extract body text and original abstract
4. generate a new abstract from the body text
5. extract keywords from the body text using TF-IDF
6. compare outputs with the original abstract using ROUGE and keyword F1

## 5. One Important Clarification For Viva

The project does **not** do this:

- abstract in
- abstract out

It does this:

- **paper body text in**
- **generated abstract out**

The extracted original abstract is only:

- training target during fine-tuning
- reference for evaluation during inference

## 6. Responsibilities of Main Libraries

Use this as a quick viva memory list:

- `requests`
  Downloads arXiv pages.
- `BeautifulSoup`
  Extracts abstract and body text from HTML.
- `re`
  Cleans text using regex.
- `Dataset`
  Converts records to train/validation datasets.
- `AutoTokenizer`
  Converts text into transformer token IDs.
- `AutoModelForSeq2SeqLM`
  Transformer model that learns and generates summaries.
- `Seq2SeqTrainer`
  Handles training and evaluation.
- `evaluate`
  Computes ROUGE metrics.
- `numpy`
  Computes averages of metrics.
- `TfidfVectorizer`
  Extracts keywords using TF-IDF.
- `Counter`
  Extracts reference frequent terms for comparison.

## 7. Most Important Functions To Remember

Training notebook:

- `fetch_recent_paper_links`
- `download_arxiv_html`
- `extract_abstract_and_body`
- `clean_text`
- `build_records`
- `preprocess_batch`
- `compute_metrics`
- `generate_summary`

Inference notebook:

- `fetch_recent_paper_links`
- `download_arxiv_html`
- `extract_abstract_and_body`
- `clean_text`
- `extract_keywords_tfidf`
- `extract_reference_terms`
- `compute_set_metrics`
- `generate_summary`
- `summarize_cs_paper`
- `print_result`


