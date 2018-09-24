#!/usr/bin/env python3 

import os
import sys
import gensim
import logging

# The Jaccard distance between two sets. Taken as the size of the
# intersection of two sets divided by the size of their union
def jaccard(s1, s2):
    # Get intersection of sets
    intersect = [i for i in s1 if i in s2]

    # Divide size of intersection by size of union
    return float(len(intersect)) / (len(s1) + len(s2) - len(intersect))

# Get the Jaccard distance for gradually increasing subsets of the
# topics we are comparing. Accounts for elements further down the list
# being less important to the definition of a term
def average_jaccard(s1, s2):
    jd = []

    # Iterate over subsets. We assume that sets are of equal length
    for i in range(1, len(s1)+1):
        # Get jaccard distance of this subset and store
        jd.append(jaccard(s1[:i], s2[:i]))

    # Get the average Jaccard distance
    return sum(jd) / len(s1)

def get_term_similarity(term, m1, m2, depth):    
    # Make sure the input term occurs in both models
    if term in m1.wv.vocab and term in m2.wv.vocab:
        # Get the top N mode similar terms from each model
        s1 = [ s[0] for s in m1.most_similar(positive=[term], topn=depth) ]
        s2 = [ s[0] for s in m2.most_similar(positive=[term], topn=depth) ]

        # Compute average jaccard on this set and return result
        return average_jaccard(s1, s2)

    # If term does not appear in one of the models then similarity is zero
    return 0.0

# Compute the similarity between two models as a single floating point
# value. Values closer to 1.0 indicate strong levels of similarity
def get_model_agreement(m1, m2, depth):
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    # Get union of terms in both models
    terms = set([*m1.wv.vocab] + [*m2.wv.vocab])
    vocab_size = len(terms)
    sims = []
    for i, term in enumerate(terms):
        if i % 10000 == 0:
            logger.info("{}/{}".format(i+1, vocab_size))
        sims.append(get_term_similarity(term, m1, m2, depth))
    
    return sum(sims) / len(sims)

def main():   
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program) 
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    model1 = gensim.models.Word2Vec.load(sys.argv[1])    
    model2 = gensim.models.Word2Vec.load(sys.argv[2])

    logger.info("models loaded")
    print(get_model_agreement( model1, model2, 10 ))

if __name__ == "__main__":
    main()