\# Text Summarisation using T5-small



An NLP project exploring \*\*abstractive text summarisation\*\* using Google's pretrained T5-small model and the CNN/DailyMail dataset.



\## Overview



Text summarisation is the process of generating a shorter version of a document while preserving its key information.



This project uses \*\*T5 (Text-to-Text Transfer Transformer)\*\*, an encoder-decoder Transformer model, to explore abstractive summarisation. Unlike extractive summarisation, which selects existing sentences, abstractive summarisation aims to generate new text that captures the meaning of the original content.



\## Objectives



\- Understand the T5 architecture and its application to summarisation.

\- Load and preprocess the CNN/DailyMail dataset.

\- Tokenise articles and their corresponding reference summaries.

\- Fine-tune T5-small for abstractive summarisation.

\- Evaluate generated summaries using the ROUGE metrics.



\## Tech Stack



\- \*\*Language:\*\* Python

\- \*\*Deep Learning:\*\* PyTorch

\- \*\*NLP:\*\* Hugging Face Transformers

\- \*\*Dataset:\*\* Hugging Face Datasets

\- \*\*Evaluation:\*\* Evaluate, ROUGE

\- \*\*Model:\*\* T5-small



\## Dataset



This project uses the CNN/DailyMail dataset, version `3.0.0`.



Each dataset example contains:

\- `article`: The original news article.

\- `highlights`: The reference summary.

\- `id`: The example identifier.



Dataset loading:



```python

from datasets import load\_dataset



dataset = load\_dataset(

&#x20;   "abisee/cnn\_dailymail",

&#x20;   "3.0.0",

&#x20;   streaming=True

)

```



Streaming is used to access the dataset without downloading the entire dataset at once.



\## Preprocessing



The preprocessing pipeline currently includes:



1\. Loading the CNN/DailyMail dataset in streaming mode.

2\. Adding the `summarize:` task prefix required by T5.

3\. Tokenising articles with a maximum input length of 512 tokens.

4\. Tokenising reference summaries with a maximum target length of 150 tokens.

5\. Truncating sequences that exceed the configured maximum lengths.

6\. Replacing target padding token IDs with `-100` so they are ignored during loss calculation.



Example:



```python

input\_text = "summarize: " + article



inputs = tokenizer(

&#x20;   input\_text,

&#x20;   max\_length=512,

&#x20;   truncation=True,

&#x20;   padding="max\_length",

&#x20;   return\_tensors="pt"

)

```



\## Project Structure



```text

text-summarization-t5/

│

├── data/

├── models/

├── utils/

│

├── main.py

├── requirements.txt

├── README.md

└── .gitignore

```



\## Setup and Installation



Clone the repository:



```bash

git clone https://github.com/sharan6422/text-summarization-t5.git

cd text-summarization-t5

```



Install dependencies:



```bash

pip install -r requirements.txt

```



Run the project:



```bash

python main.py

```



\## Current Progress



\- \[x] Project structure and dependencies

\- \[x] CNN/DailyMail dataset loading

\- \[x] T5-small tokenizer setup

\- \[x] Article and summary tokenisation

\- \[x] Batched preprocessing pipeline

\- \[x] Label padding and masking

\- \[ ] Load pretrained T5-small model

\- \[ ] Generate baseline summaries

\- \[ ] Fine-tune T5-small

\- \[ ] Evaluate using ROUGE

\- \[ ] Analyse generated summaries



\## Evaluation



The ROUGE evaluation metrics will be used to compare generated summaries against the reference summaries.



Planned metrics:



\- ROUGE-1

\- ROUGE-2

\- ROUGE-L



Evaluation results will be added after model inference and fine-tuning are implemented.



\## Future Improvements



\- Experiment with different input and output sequence lengths.

\- Compare baseline and fine-tuned model performance.

\- Analyse the impact of training data size on summarisation quality.

\- Investigate model performance on longer articles.



\## Acknowledgements



\- \[Hugging Face Transformers](https://huggingface.co/docs/transformers)

\- \[CNN/DailyMail Dataset](https://huggingface.co/datasets/abisee/cnn\_dailymail)

\- \[T5: Text-to-Text Transfer Transformer](https://arxiv.org/abs/1910.10683)



