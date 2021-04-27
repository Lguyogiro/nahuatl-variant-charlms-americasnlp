# A lot of this code came from
# https://github.com/yunjey/pytorch-tutorial
#
import argparse
import numpy as np
from pathlib import Path
import torch
from torch import nn, zeros
from torch.nn.utils import clip_grad_norm_

ALL_LANG_CODES = ['azz', 'nch', 'ncj', 'ncl', 'ngu', 'nhe', 'nhi', 'nhw',
                  'nhx', 'nhy', 'nsu']

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 30
MAX_SEQ_LENGTH = 50
EMBEDDING_SIZE = 100
HIDDEN_SIZE = 1024
NUM_LAYERS = 1
NUM_EPOCHS = 50
LR = 0.0001
DROPOUT = 0.4
CRITERION = nn.CrossEntropyLoss()


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __len__(self):
        return len(self.word2idx)


class Corpus(object):
    def __init__(self):
        self.dictionary = Dictionary()

    def fit(self, path):
        self.dictionary.add_word('<unk>')

        with open(path, 'r') as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    self.dictionary.add_word(word)

    def get_data(self, path, batch_size=20):
        # Tokenize the file content
        num_tokens = 0
        with open(path, 'r') as f:
            for line in f:
                words = line.split() + ['<eos>']
                num_tokens += len(words)

        ids = torch.LongTensor(num_tokens)

        token = 0
        with open(path, 'r') as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    unk_idx = self.dictionary.word2idx['<unk>']
                    ids[token] = self.dictionary.word2idx.get(word, unk_idx)
                    token += 1

        num_batches = ids.size(0) // batch_size
        ids = ids[:num_batches * batch_size]
        return ids.view(batch_size, -1)


class RNNLM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers,
                 dropout=DROPOUT):
        super(RNNLM, self).__init__()

        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.drop = nn.Dropout(dropout)
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers,
                            batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, h):
        # Embed word ids to vectors
        x = self.drop(self.embed(x))

        # Forward propagate LSTM
        out, (h, c) = self.lstm(x, h)

        # Reshape output to (batch_size*sequence_length, hidden_size)
        out = out.reshape(out.size(0) * out.size(1), out.size(2))

        # Decode hidden states of all time steps
        out = self.linear(out)
        return out, (h, c)


class Trainer(object):
    def __init__(self, model, corpus, criterion):
        self.model = model
        self.criterion = criterion
        self.corpus = corpus

        (self.train_losses, self.train_ppis,
         self.eval_losses, self.eval_ppis) = [], [], [], []

    @staticmethod
    def detach(states):
        return [state.detach() for state in states]

    def train(self, data, num_epochs, n_batches, batch_size,
              optimizer, seq_length, early_stopping=0,
              model_save_path=Path('old_nhi_char_lm.model'),
              eval_data=None):

        self.model.zero_grad()

        for epoch in range(num_epochs):
            self.model.train()
            # Set initial hidden and cell states
            states = (
                zeros(self.model.num_layers, batch_size, self.model.hidden_size).to(DEVICE),
                zeros(self.model.num_layers, batch_size, self.model.hidden_size).to(DEVICE)
            )
            batch_losses = []
            for i in range(0, data.size(1) - seq_length, seq_length):
                # Get mini-batch inputs and targets
                inputs = data[:, i:i + seq_length].to(DEVICE)
                targets = data[:, (i + 1):(i + 1) + seq_length].to(DEVICE)

                # Forward pass
                states = self.detach(states)
                outputs, states = self.model(inputs, states)
                loss = self.criterion(outputs, targets.reshape(-1))
                batch_losses.append(loss.to('cpu').item())

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(self.model.parameters(), 0.5)
                optimizer.step()

                step = (i + 1) // seq_length
                #if step % 100 == 0:
                #    print('Epoch [{}/{}], Step[{}/{}], '
                #          'Loss: {:.4f}, Perplexity: {:5.2f}'
                #          .format(epoch + 1, num_epochs, step, n_batches,
                #                  loss.item(), np.exp(loss.item())))

            epoch_loss = np.mean(batch_losses)
            epoch_ppi = np.exp(epoch_loss)

            print('========\n'
                  'Train: Epoch [{}/{}], Loss: {:.4f}, '
                  'Perplexity: {:5.2f}\n'
                  '========'.format(epoch + 1, num_epochs, epoch_loss,
                                    np.exp(epoch_loss)))

            self.train_losses.append(epoch_loss)
            self.train_ppis.append(epoch_ppi)

            if eval_data is not None:
                loss, ppi = self.eval(eval_data, batch_size, self.criterion,
                                      seq_length)
                self.eval_losses.append(loss)
                self.eval_ppis.append(ppi)

                print('========\n'
                      'Validation: Epoch [{}/{}], Loss: {:.4f}, '
                      'Perplexity: {:5.2f}\n'
                      '========'
                      .format(epoch + 1, num_epochs, loss, ppi))
            self.model.to('cpu')
            bundle = {
                "corpus": self.corpus,
                "model": self.model,
                "train_losses": self.train_losses,
                "train_ppis": self.train_ppis,
                "eval_losses": self.eval_losses if self.eval_losses else [],
                "eval_ppis": self.eval_ppis if self.eval_ppis else []
            }
            epoch_model_path = Path(model_save_path.parent,
                                    f'EPOCH_{epoch}_{model_save_path.name}')
            torch.save(bundle, str(epoch_model_path))
            
            self.model.to(DEVICE)
        return self.model

    def per_sample_ppi(self, texts, criterion, corpus):
        ppis = []
        self.model.eval()
        states = (
            zeros(self.model.num_layers, 1, self.model.hidden_size).to(DEVICE),
            zeros(self.model.num_layers, 1, self.model.hidden_size).to(DEVICE)
        )
        with torch.no_grad():

            for i in range(len(texts)):
                if i % 250 == 0:
                    print(i)

                sent = texts[i].split()
                label = sent[1:] + ['<eos>']

                encoded_sent = torch.zeros(1, len(sent)).long().to(DEVICE)
                encoded_label = torch.zeros(1, len(sent)).long().to(DEVICE)

                unk_idx = corpus.dictionary.word2idx['<unk>']
                # import pdb;pdb.set_trace()
                for j, word in enumerate(sent):
                    encoded_sent[0, j] = corpus.dictionary.word2idx.get(word, unk_idx)

                for j, word in enumerate(label):
                    encoded_label[0, j] = corpus.dictionary.word2idx.get(word, unk_idx)

                # Forward pass
                outputs, states = self.model(encoded_sent, states)
                loss = criterion(outputs, encoded_label.reshape(-1)).to('cpu').item()
                ppi = np.exp(loss)
                ppis.append(ppi)
        return ppis

    def eval(self, data, batch_size, criterion, seq_length):
        states = (
            zeros(self.model.num_layers, batch_size, self.model.hidden_size).to(DEVICE),
            zeros(self.model.num_layers, batch_size, self.model.hidden_size).to(DEVICE)
        )

        self.model.eval()
        with torch.no_grad():
            add_loss = 0.0
            batches_processed = 0
            for i in range(0, data.size(1) - seq_length, seq_length):
                batches_processed += 1

                #
                # Get mini-batch inputs and targets
                #
                inputs = data[:, i:i + seq_length].to(DEVICE)
                targets = data[:, (i + 1):(i + 1) + seq_length].to(DEVICE)

                # Forward pass
                outputs, states = self.model(inputs, states)
                loss = criterion(outputs, targets.reshape(-1))
                add_loss += loss.to('cpu').item()

                step = (i + 1) // seq_length
                #if step % 100 == 0:
                #    print('Loss: {:.4f}, Perplexity: {:5.2f}'
                #          .format(loss.item(), np.exp(loss.item())))

        avg_loss = add_loss / batches_processed
        ppi = np.exp(avg_loss)
        #print("Eval Perplexity: {}".format(ppi))
        return avg_loss, ppi


def run_lang(train_lang_code, model_path_template, datatype, orthography):
    train_path = Path('..', 'data', datatype, orthography, 'train',
                      f"{train_lang_code}.txt")
    val_path = Path('..', 'data', datatype, orthography, 'dev',
                    f"{train_lang_code}.txt")

    corpus = Corpus()
    corpus.fit(train_path)
    vocab_size = len(corpus.dictionary)

    training_data = corpus.get_data(train_path, batch_size=BATCH_SIZE)
    val_data = corpus.get_data(val_path, batch_size=BATCH_SIZE)

    num_batches = training_data.size(1) // MAX_SEQ_LENGTH
    model = RNNLM(vocab_size, EMBEDDING_SIZE,
                  HIDDEN_SIZE, NUM_LAYERS).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    trainer = Trainer(model, corpus, criterion=CRITERION)
    _ = trainer.train(training_data, NUM_EPOCHS, num_batches,
                      BATCH_SIZE, optimizer, MAX_SEQ_LENGTH,
                      eval_data=val_data,
                      model_save_path=model_path_template)

    best_ppi_epoch = np.argmin(trainer.eval_ppis)
    self_ppi = trainer.eval_ppis[best_ppi_epoch]

    path_to_best_model = Path.joinpath(
        model_path_template.parent,
        f'EPOCH_{best_ppi_epoch}_{model_path_template.name}'
    )

    print("Perplexity Results\n---------")
    print(f'{train_lang_code}-{train_lang_code}: {self_ppi}')

    bundle = torch.load(path_to_best_model)
    corpus = bundle['corpus']
    model = bundle['model'].to(DEVICE)
    trainer = Trainer(model, corpus, criterion=CRITERION)

    for lang_code in ALL_LANG_CODES:
        val_path = Path('..', 'data', datatype, orth, 'test',
                        f"{lang_code}.txt")
        eval_data = corpus.get_data(val_path, batch_size=BATCH_SIZE)
        avg_loss, val_ppi = trainer.eval(eval_data, BATCH_SIZE, CRITERION,
                                         MAX_SEQ_LENGTH)

        print(f'{train_lang_code}-{lang_code}: {val_ppi}')

    #
    # Get perplexity for all variant data with this lang's model
    #
    path_to_variant_id_eval_data = Path("..", "data", "variant_id", datatype,
                                        orth, 'test.tsv')
    with path_to_variant_id_eval_data.open() as f:
        text = [line.split('\t')[0] for line in f]
        ppis = trainer.per_sample_ppi(text, CRITERION, corpus)
        outpath = f"EVAL_PPs-{train_lang_code}.txt"
        with open(outpath, 'w') as f:
            f.write('\n'.join([str(p) for p in ppis]))


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('data_type', default='char')
    argparser.add_argument('--orthography', default='original')
    argparser.add_argument('--lang_code', default='nhi')
    argparser.add_argument('--model_path')
    args = argparser.parse_args()

    model_path = Path(args.model_path)
    datatype = args.data_type
    orth = args.orthography
    lang_code = args.lang_code

    run_lang(lang_code, model_path, datatype, orth)

