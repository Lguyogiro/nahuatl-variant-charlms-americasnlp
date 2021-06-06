"""
Scrape bible text from html if a translation is available for each Nahuatl
variant liste in variantes.csv.
"""
import json
import re
import requests
from bs4 import BeautifulSoup


base_directory_url = "https://scriptureearth.org/data/{}/sab/"
verse_id_regex = re.compile('T(?P<verse_id>[0-9]+(-[0-9]+)?)[a-z]?')
chapter_name_regex = re.compile(r'([A-Z0-9][A-Z][A-Z]-[0-9][0-9][0-9])\.html')


def get_page_urls(idx_page, lang_id):
    for td in idx_page.find_all('td'):
        a = td.a
        if a is None:
            continue
        page_link = a.attrs['href']
        yield "{}{}".format(base_directory_url.format(lang_id), page_link)


def compile_verses_for_page(page_soup):
    verses = {}

    current_verse = []
    current_verse_num = None

    for i, verse_div in enumerate(page_soup.find_all('div', {'class': 'txs'})):
        try:
            verse_num = re.match(verse_id_regex,
                                 verse_div.attrs['id']).group('verse_id')
        except Exception as e:
            print(e, verse_div.attrs['id'])
            continue

        verse_num_span = verse_div.find('span', {'class': 'v'})
        if verse_num_span is not None:
            _ = verse_num_span.extract()

        verse_text = verse_div.text.replace(u'\xa0', u' ')

        if current_verse_num is None:
            current_verse.append(verse_text)
            current_verse_num = verse_num

        elif verse_num == current_verse_num:
            current_verse.append(verse_text)

        else:
            verses[current_verse_num] = ' '.join(current_verse)
            current_verse = [verse_text]
            current_verse_num = verse_num
    return verses


def scrape_nuevo_testamento_for_variant(lang_id):
    idx_page = BeautifulSoup(
        requests.get(base_directory_url.format(lang_id)).text,
        "lxml"
    )
    page2verses = {}

    for page_url in get_page_urls(idx_page, lang_id):
        try:
            res = requests.get(page_url)
        except Exception as e:
            print(e)
            continue

        res.encoding = 'utf-8'
        res_text = res.text
        page_soup = BeautifulSoup(res_text, 'lxml')
        verses = compile_verses_for_page(page_soup)
        if verses:
            chapter = re.search(chapter_name_regex, page_url).group(1)
            page2verses[chapter] = verses

    return page2verses


def main():
    results = {}
    with open('../data/variantes.csv') as f:
        for line in f:
            lang_id, lang_name = line.strip('\n').split(',')
            ch2verses = scrape_nuevo_testamento_for_variant(lang_id)
            if ch2verses:
                results[lang_id] = ch2verses
                print("Completed scraping for {}".format(lang_name))
            else:
                print("No data available for {}".format(lang_id))

    with open('SCRAPED_nuevo_testamento_variantes.json', 'w') as fout:
        json.dump(results, fout)


if __name__ == '__main__':
    main()
