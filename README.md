Assignments and Default Project for Course: **Stanford University_CS224N_2021**: Natural Language Processing with Deep Learning

Each assignment (a1–a5) contains two parts: coding and written.

Written part: Analysis and insights about the model and the problem

Coding part: Implement fundamental blocks of neural models taught in the videos from starter code.

## a1: Assignment 1: Exploring Word Vectors

Written: Analysis and reasoning about the word embeddings from the GloVe model and co-occurrence-based word vectors.

Coding:  **GLOVE** Rebuild co-occurrence-based word vectors and explore the word embeddings.

## a2: Assignment 2: Understanding Word2Vec

Written: Gradient computations and reporting the results of the Word2Vec model.

Coding: Rebuild the **WORD2VEC** model and its variation

## a3: Assignment 3: Dependency Parsing

Written: Insights and tips about training neural models; understand the operation of the dependency parser and errors from the neural dependency parsing model.

Coding: Rebuild the **NEURAL** **DEPENDENCY PARSER** model: parser model (deep learning model for predicting the next transition)

, parser transitions: operate transitions (RIGHT ARC, LEFT ARC, ADD).
## a4: Assignment 4: Neural Machine Translation with RNNs

Written: Insights about different implementations of attention operations and masked parts in the sentence; 

detection and correction of errors from the model; understanding BLEU.

Coding: Implement fundamental building blocks of the **SEQ2SEQ** model.
## a5 : Assignment 5 : Self-Attention, Transformers, and Pretraining
Written: + Gaining mathematical intuitions about the advantages of multi-headed attention over single one. 
         + Empirical insights about the benefits of pretrained models.

Coding: Reimplement Self-attention, pretraining and fine-tuning LLM, **MINGPT**.

## Project : Default Project : BERT and downstream tasks
    + Implement key elements of minBERT : multi-head self-attention, bert layers, forward, embeddings
    + Using BERT embeddings for sentiment analysis, paraphrase detection, semantic textual similarity
    + Extensions : experiments on additional pretraining, contrastive learning, additional input features (POS, Name Entity Type...)
    Note : The code in repository is not provided with extensions.
    Results :
    PART 1 : BERT and Sentiment Analysis task (Without Extensions): 
        Pretraining for SST : Dev Accuracy : 0.423 / Baseline : 0.39
        Pretraining for CFIMDB : Dev Accuracy : 0.824 / Baseline : 0.78
        Finetuning for SST : Dev Accuracy : 0.52 / Baseline : 0.51
        Finetuning for CFIMDB : Dev Accuracy : 0.97 / Baseline : 0.966
    # PART 2 : (Not uploaded here - I am discussing with my supervisor whether my extensions in this part can be turned into one publication (5/1/2024)) : My finding for richer and more robust BERT embeddings
    scored much better than the baseline and other two tasks.
    Multitasks results :
    dev sentiment acc :: 0.532
    dev paraphrase acc :: 0.625
    dev sts corr :: 0.489

## Key skills
+ Mathematics (Linear Algebra, Statistics, Calculus)
+ PyTorch
+ Hugging Face
+ LaTeX

