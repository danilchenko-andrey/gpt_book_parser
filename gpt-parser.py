# coding: utf-8
import json
from openai import OpenAI
from dotenv import load_dotenv
import click
import pandas as pd
from tqdm import tqdm
import time
from typing import Optional
import os
import hashlib


load_dotenv()


CACHE_FILE = ".chatgpt_cache.json"


def load_cache():
    if os.path.exists(CACHE_FILE):
        click.echo(f"Loading cache from {CACHE_FILE}")
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            click.echo(f"Cache loaded with {len(data)} entries.")
            return data
    return {}


cache = load_cache()


def save_cache(cache):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=4)

def get_cached_response(prompt):
    cache_key = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return cache.get(cache_key)

def cache_response(prompt, response):
    cache_key = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    cache[cache_key] = response
    save_cache(cache)


def ask_chatgpt(client: OpenAI, prompt: str, model: str = "gpt-4o-mini-search-preview-2025-03-11") -> Optional[dict]:
    """
    Function to send a prompt to ChatGPT and get a response.
    
    Args:
        client (OpenAI): The OpenAI client instance.
        prompt (str): The prompt to send to ChatGPT.
        model (str): The model to use (default is "gpt-4").
    
    Returns:
        dict: The response from ChatGPT as a dictionary, or None if an error occurs.
    """
    try:
        # Check if the response is cached
        cached_response = get_cached_response(prompt)
        if cached_response:
            output_txt = cached_response
        else:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                {"role": "system", "content": "Ты – помощник по поиску книг"},
                {"role": "user", "content": prompt},
                ]
            )
            output_txt = completion.choices[0].message.content
            if "```json" in output_txt:
                output_txt = output_txt.split("```json")[1].split("```")[0]
            # Cache the response
            cache_response(prompt, output_txt)

        return json.loads(output_txt)
    except json.JSONDecodeError as e:
        click.echo(f"Error decoding JSON response: {e}", err=True)
        click.echo(f"Response: {completion.choices[0].message.content}", err=True)
        return None
    except Exception as e:
        click.echo(f"Error while communicating with ChatGPT: {e}", err=True)
        return None

def generate_prompt(isbn: str, title: Optional[str], publisher: Optional[str]) -> str:
    """
    Generate a prompt to ask ChatGPT for book information based on ISBN.
    
    Args:
        isbn (str): The ISBN of the book.
        title (str): The title of the book (optional).
        publisher (str): The publisher of the book (optional).
    
    Returns:
        str: The generated prompt.
    """
    example = """{
  "title": "Зимовье зверей: русская народная сказка",
  "author": "Русская народная сказка (обработано М. Булатовой)",
  "description": "Русская народная сказка, обработанная М. Булатовой, из серии \"Лучшее детям\".",
  "year": "2024",
  "publisher": "Мелик-Пашаев",
  "weight": "200",
  "number_of_pages": "16",
  "url": "https://knigamir.com/catalog/literatura-dlya-detey_ID118/zimove-zverey-russkaya-narodnaya-skazka_ID1997895/"
}"""
    schema = """"{type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "author": {"type": "string"},
                        "description": {"type": "string"},
                        "year": {"type": "integer"},
                        "publisher": {"type": "string"},
                        "weight": {"type": "integer"},
                        "number_of_pages": {"type": "integer"},
                        "url": {"type": "string"},
                    }
                }"""
    return f"""Найди книгу, используя ее isbn. Иногда могут быть указаны Title и Publisher. Определи название, автора, издательство, аннотацию (исключи рекламные тексты о площадке), год издания и вес в граммах. Также найди ссылку на сайт с книгой.
Выведи ответ в формате json с соответсвующими полями title, author, description, year, publisher, weight, number_of_pages, url. Если не получилось найти информацию, то впиши NOT_FOUND в поле. Используй только подтвержденную информацию с сайта, это очень важно, чтобы вся информация была из источника. Ответ должен содержать только JSON.
Schema: {schema}

Пример: {example}


ISBN: {isbn}{" Title: " + title if title else ""}{" Publisher: " + publisher if publisher else ""}
"""

def get_book_info(client: OpenAI, isbn: str, title: Optional[str], publisher: Optional[str] ) -> Optional[dict]:
    """
    Get book information by querying ChatGPT with a generated prompt.
    
    Args:
        isbn (str): The ISBN of the book.
        title (str): The title of the book (optional).
        publisher (str): The publisher of the book (optional).
    
    Returns:
        dict: The book information as a dictionary, or None if an error occurs.
    """
    prompt = generate_prompt(isbn, title, publisher)
    response = ask_chatgpt(client, prompt)
    return response   


@click.command()
@click.option("--input-file", default="Тильда поставка март.xlsx", help="Input Excel file.")
@click.option("--output-file", default="output.xlsx", help="Output Excel file.")
def main(input_file: str, output_file: str):
    """
    Main function to run the script.
    """
    openai_client = OpenAI()

    df = pd.read_excel(input_file)
    df = df[~df["ISBN"].isna()]
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        isbn = row["ISBN"]
        if pd.notna(isbn):
            try:
                book_info = get_book_info(openai_client, str(isbn).strip(), title=row.get("Title"), publisher=row.get("Brand"))
                if book_info is None:
                    continue
                for k, v in book_info.items():
                    df.at[index, "gpt_" + k] = v
                if index % 10 == 0:
                    df.to_excel(output_file, index=False)
                time.sleep(0.5)
            except Exception as e:
                click.echo(f"Error processing ISBN {isbn}: {e}", err=True)
    df.to_excel(output_file, index=False)
    click.echo(f"Processing completed. Output saved to {output_file}.")
    save_cache(cache)
    click.echo(f"Cache saved with {len(cache)} entries.")


if __name__ == "__main__":
    main()
