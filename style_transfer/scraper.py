from typing import List, Tuple
import json
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
        browser.find_by_css('.right.arrow-nav')[0].click()

    return sections
    

def scrape_current_page(browser) -> None:
    dropdown = browser.find_by_css('.dropdownMenu select')
    option_index = int(dropdown['selectedIndex'])
    section_name = dropdown.find_by_tag('option')[option_index].text

    comparison = browser.find_by_id('noFear-comparison')[0]
    rows = list(comparison.find_by_tag('tr'))

    out = []
    for row in rows:
        cells = row.find_by_tag('td')

        original_paragraphs = [item.text for item in cells[1].find_by_css('*')]
        modern_paragraphs = [item.text for item in cells[2].find_by_css('*')]

        out.append((original_paragraphs, modern_paragraphs))

    return Section(section_name, out)

def scrape_and_save(path: str, title: str, url: str) -> None:
    save_sections(path, title, url, scrape(url))

if __name__ == '__main__':
    scrape_and_save(
            'data/antony-and-cleopatra.json',
            'Antony and Cleopatra',
            'http://nfs.sparknotes.com/antony-and-cleopatra/page_2.html')

    #scrape_and_save(
    #        'data/midsummers-night-dream.json',
    #        "Midsummer's Night Dream",
    #        'http://nfs.sparknotes.com/msnd/page_2.html')


