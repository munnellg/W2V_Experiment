#!/usr/bin/env python3
import os
import sys
import gensim
import logging

def main():
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    if len(sys.argv) < 2:
        print("usage: {} MODEL".format(sys.argv[0]))
        exit()

    # Load the Word2Vec model from user input
    model = gensim.models.Word2Vec.load(sys.argv[1])
    
    # while user hasn't quit, print words that are most similar to their query
    query = input("Input term >>> ")
    while query != "\quit":
        print(model.most_similar(positive=[query.lower()]))
        query = input("Input term >>> ")

    
if __name__ == "__main__":
    main()
