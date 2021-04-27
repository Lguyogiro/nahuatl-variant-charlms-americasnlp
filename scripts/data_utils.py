def load_tsv_data(path):
    texts, labels = [], []
    with open(path) as f:
        for i, row in enumerate(f):
            text, label = row.strip('\n').rsplit('\t', 1)
            texts.append(text)
            labels.append(label)
    return texts, labels


def get_data_maps(training_text, training_labels):
    ch2id = {'<pad>': 0, '<unk>': 1}
    label2id = {}
    id2label = {}

    cntr = 2
    for train_ex in training_text:
        for ch in train_ex.split():
            if ch not in ch2id:
                ch2id[ch] = cntr
                cntr += 1
    cntr = 0
    for label in training_labels:
        if label not in label2id:
            label2id[label] = cntr
            id2label[cntr] = label
            cntr += 1

    return ch2id, label2id, id2label