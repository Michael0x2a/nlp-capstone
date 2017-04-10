from typing import List, Tuple
import json
import sys
from splinter import Browser 

Paragraph = str
Original = List[Paragraph]
Modern = List[Paragraph]

class Section:
    def __init__(self, name: str, comparisons: List[Tuple[Original, Modern]]) -> None:
        self.name = name
        self.comparisons = comparisons

def save_sections(path: str, title: str, url: str, sections: List[Section]) -> None:
    paragraphs = []
    for section in sections:
        for original, modern in section.comparisons:
            paragraphs.append({
                    'section_name': section.name,
                    'original': '$NEWLINE$'.join(original),
                    'translated': '$NEWLINE$'.join(modern)
            })
    obj = {
            'metadata': {
                'title': title,
                'original_source': url,
            },
            'paragraphs': paragraphs,
    }
    with open(path, 'w') as stream:
        json.dump(obj, stream, sort_keys=True, indent=4)

def on_citations_page(url):
    return url.endswith('citing/') or url.endswith('citing.html') or url.endswith('citing')

def scrape(root_url: str) -> List[Section]:
    browser = Browser('chrome', executable_path='./drivers/chromedriver-linux64')
    browser.visit(root_url)

    sections = []
    while not on_citations_page(browser.url):
        sections.append(scrape_current_page(browser))
        try:
            browser.find_by_css('.right.arrow-nav')[0].click()
        except:
            break

    return sections
    

def scrape_current_page(browser) -> None:
    dropdown = browser.find_by_css('.dropdownMenu select')
    option_index = int(dropdown['selectedIndex'])
    section_name = dropdown.find_by_tag('option')[option_index].text

    comparison = browser.find_by_id('noFear-comparison')[0]
    rows = list(comparison.find_by_tag('tr'))

    out = []
    for row in rows:
        original_paragraphs = [item.text for item in row.find_by_css('.noFear-left *')]
        modern_paragraphs = [item.text for item in row.find_by_css('.noFear-right *')]

        out.append((original_paragraphs, modern_paragraphs))

    return Section(section_name, out)

def scrape_and_save(path: str, title: str, url: str) -> None:
    save_sections(path, title, url, scrape(url))

if __name__ == '__main__':
    #scrape_and_save(
    #        'data/antony-and-cleopatra.json',
    #        'Antony and Cleopatra',
    #        'http://nfs.sparknotes.com/antony-and-cleopatra/page_2.html')

    #scrape_and_save(
    #        'data/midsummers-night-dream.json',
    #        "Midsummer's Night Dream",
    #        'http://nfs.sparknotes.com/msnd/page_2.html')

    arg = sys.argv[1]

    '''
    if arg == "3":
        scrape_and_save(
                'data/julius-caeser.json',
                'Julius Caesar',
                'http://nfs.sparknotes.com/juliuscaesar/page_2.html')

        scrape_and_save(
                'data/king-lear.json',
                'King Lear',
                'http://nfs.sparknotes.com/lear/page_2.html')

        scrape_and_save(
                'data/macbeth.json',
                'Macbeth',
                'http://nfs.sparknotes.com/macbeth/page_2.html')

        scrape_and_save(
                'data/the-merchant-of-venice.json',
                'The Merchant of Venice',
                'http://nfs.sparknotes.com/merchant/page_2.html')

        scrape_and_save(
                'data/much-ado-about-nothing.json',
                'Much Ado About Nothing',
                'http://nfs.sparknotes.com/muchado/page_2.html')

        scrape_and_save(
                'data/othello.json',
                'Othello',
                'http://nfs.sparknotes.com/othello/page_2.html')

        scrape_and_save(
                'data/the-taming-of-the-shrew.json',
                'The Taming of the Shrew',
                'http://nfs.sparknotes.com/shrew/page_2.html')

        scrape_and_save(
                'data/the-tempest.json',
                'The Tempest',
                'http://nfs.sparknotes.com/tempest/page_2.html')

        scrape_and_save(
                'data/twelfth-night.json',
                'Twelfth Night',
                'http://nfs.sparknotes.com/twelfthnight/page_2.html')
    '''

    if arg == "1":
        scrape_and_save(
                'data/as-you-like-it.json',
                'As You Like It',
                'http://nfs.sparknotes.com/asyoulikeit/page_2.html')

    if arg == "2":
        scrape_and_save(
                'data/the-comedy-of-errors.json',
                'The Comedy of Errors',
                'http://nfs.sparknotes.com/errors/page_2.html')

    if arg == "3":
        scrape_and_save(
                'data/hamlet.json',
                'Hamlet',
                'http://nfs.sparknotes.com/hamlet/page_2.html')

    if arg == "4":
        scrape_and_save(
                'data/henry-iv-part1.json',
                'Henry IV, part 1',
                'http://nfs.sparknotes.com/henry4pt1/page_3.html')

    if arg == "5":
        scrape_and_save(
                'data/henry-iv-part2.json',
                'Henry IV, part 2',
                'http://nfs.sparknotes.com/henry4pt2/page_265.html')

    if arg == "6":
        scrape_and_save(
                'data/henry-v.json',
                'Henry V',
                'http://nfs.sparknotes.com/henryv/page_2.html')

    if arg == "7":
        scrape_and_save(
                'data/sonnets.json',
                'Sonnets',
                'http://nfs.sparknotes.com/sonnets/sonnet_1.html')


