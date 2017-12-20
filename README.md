# Summarizer
This repo contains two summarizers that has been implemented in the course EDAN70 by me (Hampus Londögård dat13hlo) and Hannah Lindblad (elt13hli). 

## tfidf
This summarizer utilize the tfidf-technique and expands by using a stemmer and a little cleaning. 

## centroid
This summarizer can actually utilize two different scoring methods, just swap the scoring-method used at the end of "summarize"-method to the other and try! 

## How to run
You need to download GloVe (6B tokens version, Wikipedia 2014 + Gigaword 5) from https://nlp.stanford.edu/projects/glove/ and put the glove.6B.300d.txt-file in the subfolder glove.

Then run ./convertGlove.sh from glove/

Then you can run centroid
