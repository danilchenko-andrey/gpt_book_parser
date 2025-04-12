# coding: utf-8
import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import json
import re
from typing import Optional, Dict
import click
import time

def parse_book_page(html: str) -> dict:
    soup = BeautifulSoup(html, 'html.parser')
    book_data = {}

    # Title
    title_tag = soup.find('title')
    book_data['title'] = title_tag.get_text(strip=True) if title_tag else None

    # Image URL
    image_tag = soup.find('meta', {'itemprop': 'image'})
    book_data['image_url'] = "https:" + image_tag['content'] if image_tag else None

    # Short annotation
    short_annot = soup.find('meta', {'name': 'description'})
    book_data['short_annotation'] = short_annot['content'] if short_annot else None

    # Annotation
    book_data['annotation'] = None
    annotation_section = soup.find('div', id='annotation')
    if annotation_section:
        paragraphs = annotation_section.find_all(['div', 'p'], recursive=True)
        for p in paragraphs:
            text = p.get_text(strip=True)
            if text and not text.lower().startswith("аннотация"):
                book_data['annotation'] = text
                break

    # Author, Publisher, Year, Series
    book_data['author'] = None
    book_data['publisher'] = None
    book_data['year'] = None
    book_data['series'] = None

    features = soup.find_all('div', string=re.compile("Автор|Издательство|Серия"))
    for feature in features:
        label = feature.get_text(strip=True)
        parent = feature.find_parent()
        if not parent:
            continue
        links = parent.find_all('a')
        text_spans = parent.find_all('span')

        if "Автор" in label and links:
            book_data['author'] = links[0].get_text(strip=True)

        elif "Издательство" in label:
            if links:
                book_data['publisher'] = links[0].get_text(strip=True)
            if text_spans:
                raw_text = text_spans[0].get_text()
                year_match = re.search(r'\b(19|20)\d{2}\b', raw_text)
                if year_match:
                    book_data['year'] = year_match.group(0)

        elif "Серия" in label and links:
            book_data['series'] = links[0].get_text(strip=True)

    return book_data

#https://www.labirint.ru/search/978-5-93898-740-1/?stype=0
def get_book_data_labirint(isbn):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/132.0.0.0 YaBrowser/25.2.0.0 Safari/537.36"
        }

        search_url = f"https://www.labirint.ru/search/{isbn}/?stype=0"
        # print(search_url)
        response = requests.get(search_url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")

        product_link_tag = soup.select_one("a.product-card__name")
        if not product_link_tag:
            return None, None
        product_url = "https://www.labirint.ru" + product_link_tag["href"]
    
        img = soup.select_one("img.book-img-cover")
        if not img:
            return None, None
        data_src = img.get("data-src")
        return product_url, data_src
    except Exception as e:
        print(f"Error: {e}")
        return None, None


@click.command()
@click.option("--input-file", default="output.xlsx", help="Input Excel file.")
@click.option("--output-file", default="output_labirint.xlsx", help="Output Excel file.")
def main(input_file: str, output_file: str):
    click.echo(f"Processing file: {input_file}")
    df = pd.read_excel(input_file)
    # Проходим по строкам таблицы и заполняем столбцы K (цена) и I (вес)
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        isbn = row['ISBN']
        if pd.notna(isbn):
            url, img = get_book_data_labirint(str(isbn).strip())
            if url and img:
                df.at[index, "labirint_url"] = url
                df.at[index, "labirint_img"] = img
            time.sleep(0.1)
            if index % 10 == 0:
                df.to_excel(output_file, index=False)

    # Сохраняем обновленную таблицу
    df.to_excel(output_file, index=False)

    print(f"Файл сохранен: {output_file}")


if __name__ == "__main__":
    main()
