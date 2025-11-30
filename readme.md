# RAG Project

0. make venv
python -m venv .venv

1. enter venv
.\.venv\Scripts\activate 

2. install requirements
pip install -r requirements.txt

3. pipeline should be runnable.


Structure
rag-qa/
│
├── config/
│   └── paths.py         # Path, detects working directory of colab or local
│
├── src/
│   ├── data.py          # check/download dataset
│   ├── training.py      # check/train model
│   ├── evaluate.py
│   └── utils.py
│
└── pipeline.ipynb       
