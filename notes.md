Retriever gibt malmal die gleiche Antwort öfter?
Tokenizer verwenden


wir appenden alle inputtexte
kommt ein text bei mehrern Fragen vor, dann haben wir den Text auch mehrfach im Trainidsdatensatz
das dann uniquen wäre eine Idee
-> dumplikate werden jetzt entfernt


1. Test retreiver
mit test datensatz:
für jede Frage
    schaue ob die title zu tieln aus den files passen
    das vom model retrieve ergebnis muss in der musterlösung enthalten sein
    -> das wird gemacht im evaluate_rag_retrieve

2. Embeding Parameter evaluieren: text length and overlap

3. vergleich der beiden mit dem test retriever



dann generator und fertig


---------------
matthias
branch aufräumen, refactor




lucas
append matthias pipeline on lucas code


next:
compare embedding and retrieve

---------
plakat

comare with or without tokenizer

top k retriever comparison


compare 2 llms at generator part


visualize pipeline
concrete exampel with question, retrieved chunks, generated answer, expected answer.

evaluation of whole pipeline

compare acc on test and train dataset