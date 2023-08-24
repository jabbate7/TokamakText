# TokamakText
![](assets/chatd3d.png)
### Pip Environment
1. Make pip environemnt
2. `python3 -m pip install chromadb python-dotenv openai flask h5py`


### Setting up dotenv
You should make a file in the root of this repo called `.env` that contains the following variables:
```
LLM_NAME= ${your_llm_model_name} # (e.g "gpt-3.5-turbo-16k" or "NousResearch/Nous-Hermes-Llama2-13b")

OPENAI_API_KEY=${your_openai_api_key} # Only if you are using OpenAI models

CACHE_DIR=${your_cache_dir} # Only if you are using Huggingface models
```

### Running HuggingFace Models
To run huggingface models, you'll need to install the torch and transformers libraries. A conda `environment.yml` file is provided for this purpose.



### Set up OPEN AI Keys
Go to OPEN AI website and set-up billing information and get API key. You can set usage limits to make sure you don't get charged more than $1 or so. 

Copy OPENAI key to a file you make called `.env` in the `rag/` directory with the line: `OPENAI_API_KEY="YOUR_KEY"`

### Convert PDFs to txt file
This is needed to embed miniproposals and any papers from journals. In the main directory run `python mass_pdf_to_txt.py pdf_dir txt_dir` and it will grab all the pdfs from `pdf_dir` and output them in `txt_dir` with the same name. This is not error checked well so there should not be non-pdfs in `pdf_dir` and `txt_dir` has to exist. 


### Set-up embeddings
Move to `rag/` directory and do everything else there. Run `python3 embed_data.py h5_file` to turn a h5 file into a file full of embeddings. Make sure the embeddings end up in a `db/` directory in the `rag/` directory, not the main directory. 

### Run rag server
Run `python3 rag_server.py` and make queries to the local website. By default, this will use the OpenAI API, but you can run a huggingface model (e.g. LLAMA) instead by typing `python3 rag_server.py --model_name huggingface`.
