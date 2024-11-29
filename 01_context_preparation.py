import random
from datasets import load_dataset
import json
from typing import List, Dict, Tuple
from tqdm import tqdm
import re
from datetime import datetime
import os
from transformers import GPT2TokenizerFast
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import logging
import argparse

@dataclass
class DatasetConfig:
    """Конфигурация для источника данных"""
    name: str
    dataset: str
    subset: str = None
    text_field: str = "text"
    proportion: float = 0.25

class TextProcessor:
    def __init__(self,
                 target_tokens: int = 300,
                 min_tokens: int = 200,
                 max_tokens: int = 400,
                 min_words: int = 50):
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self.target_tokens = target_tokens
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.min_words = min_words

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def clean_text(self, text: str, source: str = None) -> str:
        """Очистка текста с учетом источника"""
        if not isinstance(text, str):
            return ""

        # Базовая очистка для всех источников
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n+', ' ', text)

        if source == "wikipedia":
            text = re.sub(r'=+\s*.+\s*=+', '', text)
            text = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]', r'\1', text)

        elif source == "scielo":
            text = re.sub(r'ABSTRACT[:\s]+', '', text, flags=re.IGNORECASE)
            text = re.sub(r'INTRODUCTION[:\s]+', '', text, flags=re.IGNORECASE)
            text = re.sub(r'METHODS[:\s]+', '', text, flags=re.IGNORECASE)
            text = re.sub(r'RESULTS[:\s]+', '', text, flags=re.IGNORECASE)

        elif source == "multi_news":
            text = re.sub(r'@highlight\s*\n', '', text)
            text = re.sub(r'\[START\]|\[END\]', '', text)
            text = re.sub(r'\(\d{1,2}/\d{1,2}/\d{2,4}\)', '', text)

        text = re.sub(r'https?://\S+', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\s*([.,!?])\s*', r'\1 ', text)

        return text.strip()

    def is_valid_chunk(self, chunk: str) -> bool:
        """Проверка валидности чанка"""
        if len(chunk.split()) < self.min_words:
            return False

        num_tokens = len(self.tokenizer.encode(chunk))
        if num_tokens < self.min_tokens or num_tokens > self.max_tokens:
            return False

        return True

    def chunks(self, text: str) -> List[str]:
        """Разбивка текста на чанки с учетом предложений"""
        text = self.clean_text(text)
        if not text:
            return []

        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = []
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = len(self.tokenizer.encode(sentence))

            if current_tokens + sentence_tokens <= self.max_tokens:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
            else:
                if current_chunk and self.is_valid_chunk(' '.join(current_chunk)):
                    chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_tokens = sentence_tokens

        if current_chunk and self.is_valid_chunk(' '.join(current_chunk)):
            chunks.append(' '.join(current_chunk))

        return chunks

class ContextGenerator:
    def __init__(self, total_contexts: int):
        self.total_contexts = total_contexts
        self.processor = TextProcessor()

        self.datasets = [
            DatasetConfig("wikipedia", "wikipedia", "20220301.en",
                         text_field="text", proportion=0.3),
            DatasetConfig("c4", "allenai/c4", "en",  # обновили путь для c4
                         text_field="text", proportion=0.3),
            DatasetConfig("openwebtext", "openwebtext",
                         text_field="text", proportion=0.25),
            DatasetConfig("multi_news", "multi_news",
                         text_field="document", proportion=0.15)  # увеличили долю
        ]

    def create_context_chunk(self, text: str, source: str) -> List[Dict]:
        """Создание чанков с учетом источника"""
        cleaned_text = self.processor.clean_text(text, source)
        chunks = self.processor.chunks(cleaned_text)
        return [{
            'content': chunk,
            'title': chunk.split('.')[0][:100],
            'source': source,
            'stats': {
                'tokens': len(self.processor.tokenizer.encode(chunk)),
                'length': len(chunk),
                'fetch_time': datetime.now().isoformat()
            }
        } for chunk in chunks]

    def load_dataset_sample(self, config: DatasetConfig) -> List[Tuple[str, str]]:
        """Загрузка сэмпла из датасета"""
        try:
            n_samples = int(self.total_contexts * config.proportion * 1.5)

            if config.subset:
                ds = load_dataset(config.dataset, config.subset, split='train', streaming=True, trust_remote_code=True)
            else:
                ds = load_dataset(config.dataset, split='train', streaming=True, trust_remote_code=True)

            texts = [(item[config.text_field], config.name) for item in ds.take(n_samples)
                    if isinstance(item[config.text_field], str)]

            self.processor.logger.info(f"Loaded {len(texts)} samples from {config.name}")
            return texts

        except Exception as e:
            self.processor.logger.error(f"Error loading {config.name}: {str(e)}")
            return []

    def process_contexts(self, output_dir: str = 'clean_contexts', batch_size: int = 100) -> None:
        """Обработка и сохранение контекстов"""
        all_chunks = []

        # Загружаем и обрабатываем каждый датасет
        for config in self.datasets:
            texts = self.load_dataset_sample(config)

            with ThreadPoolExecutor() as executor:
                chunk_lists = list(tqdm(
                    executor.map(lambda x: self.create_context_chunk(x[0], x[1]), texts),
                    total=len(texts),
                    desc=f"Processing {config.name}"
                ))

                # Объединяем все чанки
                for chunks in chunk_lists:
                    all_chunks.extend(chunks)

        # Перемешиваем и обрезаем до нужного количества
        random.shuffle(all_chunks)
        all_chunks = all_chunks[:self.total_contexts]

        # Сохраняем батчами
        os.makedirs(output_dir, exist_ok=True)

        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i + batch_size]
            timestamp = datetime.now().strftime('%Y%m%d_%H%M')

            filename = os.path.join(
                output_dir,
                f'contexts_batch_{i//batch_size}_{timestamp}.json'
            )

            with open(filename, 'w', encoding='utf-8') as f:
                json.dump({
                    'metadata': {
                        'batch_size': len(batch),
                        'avg_tokens': sum(c['stats']['tokens'] for c in batch) / len(batch),
                        'timestamp': timestamp
                    },
                    'contexts': batch
                }, f, indent=2, ensure_ascii=False)

            self.processor.logger.info(f"Saved batch {i//batch_size} to {filename}")

def main():
    parser = argparse.ArgumentParser(description='Generate context batches for RAG evaluation')
    parser.add_argument('--total', type=int, default=200000,
                      help='Total number of contexts to generate (default: 200000)')
    parser.add_argument('--batch_size', type=int, default=100,
                      help='Number of contexts per batch file (default: 100)')
    parser.add_argument('--output_dir', type=str, default='clean_contexts',
                      help='Output directory for context files (default: clean_contexts)')

    args = parser.parse_args()

    generator = ContextGenerator(args.total)
    generator.process_contexts(args.output_dir, args.batch_size)

    print(f"\nGeneration completed:")
    print(f"Total contexts: {args.total}")
    print(f"Batch size: {args.batch_size}")
    print(f"Number of files: {args.total // args.batch_size + bool(args.total % args.batch_size)}")

if __name__ == "__main__":
    main()
