# Poster ideas

comare with or without tokenizer

top k retriever comparison


compare 2 llms at generator part


visualize pipeline
concrete exampel with question, retrieved chunks, generated answer, expected answer.

evaluation of whole pipeline

compare acc on test and train dataset

# Future ideas

- Llama 3.1 (run locally)
- few shot prompt
- try out sparse embedding
- clean up code & repo

# Open questions

- How to evaluate?
- Why FAISS?
- How does F1 evaluation work?
- How does the re-ranker work (similarity/quality/relevance/...?)
- embed all dataset splits or only a few?
- Are train and test dataset disjoint?
- wich LaTeX formatting? IEEE?

# Poster improvements

- grouped bar chart
- example question (maybe in pipeline)
- add chunking size 
- (dataset description)
- what to do next
- enhace motivation part
- re-ranking in extra box in pipeline

# F1 evaluation

1. normalize prediction & golden answer:
    - lowercasing
    - remove punctuation
    - remove "a"/"an"/"the"
    - normalize white spaces
2. tokenize prediction & golden answer
3. count same tokens
4. calculate precision & recall:
    - $precision = \frac{num_{same}}{|tokens_{prediction}|}$
    - $recall = \frac{num_{same}}{|tokens_{golden}|}$
5. Calculate F1 score: $F1 = 2 \cdot precision \cdot \frac{recall}{precision + recall}$ ($0$ if $num_{same}=0$)

# How to use Ollama

1. install Ollama & the desired model
2. run Ollama server: `ollama serve`
3. set `USE_OLLAMA = True`, the Ollama host & port in first cell of `pipeline.ipynb`  
Done. the rest should work automatically