import json
import pandas as pd
import random
import re
from collections import defaultdict
from string import punctuation
from orthography import normalize_orthography


TRAIN_PERCENT = 0.85

punctuation = ''.join([ch for ch in punctuation if ch != '.'])
punctuation += '¿¡'
punct_pattern = re.compile('[^a-z áéíñóú.!]+')


def preprocess(text: str) -> str:
    """
    Normalize the scraped text. This should be changed/ignored for other data
    sources (e.g. texts that use the apostrophe to mark the saltillo would
    split those words into two. Also underlined text, used in the mecayapan
    bibles, isn't correctly proocessed with this function).

    :param text: raw scraped text
    :return: cleaned text
    """
    text = text.lower()
    text = re.sub(punct_pattern, ' ', text)
    text = re.sub(r'([.!])', r' \1 ', text)
    text = re.sub('  +', ' ', text).strip(' \n\t')
    return text


def words2chars(lines):
    return [' '.join(['_' if ch == ' ' else ch for ch in
                      list(line.strip('\n'))])
            for line in lines]


def write_lang_files(df):
    lang_cols = [col for col in df.columns
                 if col not in ('chapter', 'verse', 'split')]
    for col in lang_cols:
        train_df = df[df.split == 'train']
        val_df = df[df.split == 'val']

        original_text_train = train_df[col].tolist()
        with open('data/original/train/{}.txt'.format(col), 'w') as fout:
            fout.write('\n'.join(original_text_train))

        with open('data/char/original/train/{}.txt'.format(col), 'w') as fout:
            fout.write('\n'.join(words2chars(original_text_train)))

        original_text_eval = val_df[col].tolist()
        with open('data/original/eval/{}.txt'.format(col), 'w') as fout:
            fout.write('\n'.join(original_text_eval))
        with open('data/char/original/eval/{}.txt'.format(col), 'w') as fout:
            fout.write('\n'.join(words2chars(original_text_eval)))

        normalized_text_train = [
            ' '.join([normalize_orthography(w) for w in sent.split()])
            for sent in original_text_train
        ]
        normalized_text_eval = [
            ' '.join([normalize_orthography(w) for w in sent.split()])
            for sent in original_text_eval
        ]
        with open('data/normalized/train/{}.txt'.format(col), 'w') as fout:
            fout.write('\n'.join(normalized_text_train))
        with open('data/char/normalized/train/{}.txt'.format(col), 'w') as fout:
            fout.write('\n'.join(words2chars(normalized_text_train)))

        with open('data/normalized/eval/{}.txt'.format(col), 'w') as fout:
            fout.write('\n'.join(normalized_text_eval))
        with open('data/char/normalized/eval/{}.txt'.format(col), 'w') as fout:
            fout.write('\n'.join(words2chars(normalized_text_eval)))


def create_combined_df(raw_scraped_data):
    ch2verse_keep = defaultdict(list)

    inter_chapters = set([])
    for l in raw_scraped_data:
        chapters = set(raw_scraped_data[l])
        if not inter_chapters:
            inter_chapters = chapters
        else:
            inter_chapters = inter_chapters.intersection(chapters)

    for ch in inter_chapters:
        if not all(ch in raw_scraped_data[lang] for lang in raw_scraped_data):
            continue
        maxes = []
        for lang in raw_scraped_data:
            verse_nums = []
            chapter = raw_scraped_data[lang][ch]
            for verse in chapter:
                try:
                    verse_nums.append(int(verse))
                except ValueError:
                    vstart, vstop = verse.split('-')
                    verse_nums.extend(list(range(int(vstart), int(vstop) + 1)))
            maxes.append(max(verse_nums))

        all_lang_verses = [raw_scraped_data[l][ch] for l in raw_scraped_data]
        verses_in_common = []
        for i in range(1, max(maxes) + 1):
            if all(str(i) in verses for verses in all_lang_verses) and all(
                    verses[str(i)] != '' for verses in all_lang_verses):
                ch2verse_keep[ch].append(i)
                verses_in_common.append(i)

    combined_rows = []
    for ch, verses in ch2verse_keep.items():
        for verse in verses:
            row = {'chapter': ch, 'verse': verse}
            row.update(
                {lang: preprocess(raw_scraped_data[lang][ch][str(verse)])
                 for lang in raw_scraped_data}
            )
            combined_rows.append(row)

    splits = ['train' if random.random() < TRAIN_PERCENT else 'val'
              for _ in range(len(combined_rows))]

    df = pd.DataFrame(combined_rows, columns=list(combined_rows[0].keys()))
    df['split'] = splits
    return df


if __name__ == '__main__':
    with open('data/SCRAPED_nuevo_testamento_variantes.json') as f:
        res = json.load(f)

    combined_df = create_combined_df(res)

    print("Writing combined file...")
    combined_df.to_csv('data/original_combined_variant_data.csv', index=False)

    print("Writing individual language files...")
    write_lang_files(combined_df)
