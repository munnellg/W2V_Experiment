#!/usr/bin/env python3
 
import logging
import os.path
import sys
import multiprocessing
from nltk.tokenize import word_tokenize 

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

if __name__ == '__main__':
    # Set up some logging stuff. This thing is going to run for hours so it's
    # helpful to see some output so we know something is happening
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))
 
    # check and process input arguments
    if len(sys.argv) < 3:
        print ("usage: {} input output".format(sys.argv[0]))
        sys.exit(1)

    # get the input and output files from the user arguments
    inp, output = sys.argv[1:3]
    
    # train word2vec on the input file
    model = Word2Vec(LineSentence(inp), size=200, window=10, min_count=10,
            workers=multiprocessing.cpu_count(), sample= 1E-3)
    
    # save word2vec to the output file
    model.save(output)
