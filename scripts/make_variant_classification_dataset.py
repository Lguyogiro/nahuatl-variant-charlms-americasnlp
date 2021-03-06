"""
Once you've created the variant dataset with `create_datasets.py`, use this
script to generate a dataset that combines variants, which can be used for the
variant identification task. In our experiments, we only use the test set since
we don't actually train a model for this task (we choose the variant whose
language model has the lowest perplexity). However, this dataset could be
used for a separate variant id model/experiment as well.

"""
from pathlib import Path
import random


unit = "char"
eval_size = 1104
dev_size = 552
test_size = 552


path_to_train = Path('..', 'data', 'char', 'normalized', 'train')
path_to_combined = Path('..', 'data', 'variant_id', 'char', 'normalized')
path_to_eval = Path('..', 'data', 'char', 'normalized', 'eval')


if __name__ == '__main__':
    all_train = []
    for lang_file in path_to_train.glob('*txt'):
        lang = lang_file.name.split('.')[0]
        with lang_file.open() as f:
            all_train.extend(
                ["{}\t{}".format(line.strip('\n'), lang)
                 for line in f]
            )
    random.shuffle(all_train)

    with (path_to_combined / 'train.tsv').open('w') as fout:
        fout.write('\n'.join(all_train))

    all_dev, all_test = [], []
    for lang_file in path_to_eval.glob('*txt'):
        lang = lang_file.name.split('.')[0]
        with lang_file.open() as f:
            lines = ["{}\t{}".format(line.strip('\n'), lang)
                     for line in f]
            all_dev.extend(lines[:dev_size])
            all_test.extend(lines[dev_size:dev_size + test_size])
    random.shuffle(all_dev)
    random.shuffle(all_test)

    with (path_to_combined / 'dev.tsv').open('w') as fout:
        fout.write('\n'.join(all_dev))

    with (path_to_combined / 'test.tsv').open('w') as fout:
        fout.write('\n'.join(all_test))



