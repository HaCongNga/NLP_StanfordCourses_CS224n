# Calculate the accuracy of a baseline that simply predicts "London" for every
#   example in the dev set.
# Hint: Make use of existing code.
# Your solution here should only be a few lines.
from tqdm import tqdm
import utils
import argparse
argp = argparse.ArgumentParser()
argp.add_argument('--eval_corpus_path', default=None)
args = argp.parse_args()
def all_london(eval_corpus_path) :
    with open(eval_corpus_path, 'r', encoding='utf-8') as f :
        num_lines = len(f.readlines())
        predictions = ["London"] * num_lines
        total, correct = utils.evaluate_places(eval_corpus_path, predictions)
        if total > 0:
            return correct/total*100, total
print(all_london(args.eval_corpus_path))