import json
import random
from typing import List, Dict
from datetime import datetime
import os
import argparse
from tqdm import tqdm
import hashlib
import math

class BatchGenerator:
    def __init__(self):
        self.model = "gpt-4o-mini"
        self.system_prompt = """You are an expert at creating evaluation data for testing RAG systems."""

        self.template = """Generate a question and answer pair for relevancy evaluation.

Context: {context}

The answer should be {relevancy_type} to the question. If irrelevant, make sure the answer appears plausible but doesn't actually address the question.

Return in JSON format:
{{
    "rag_context": "the exact context provided above",
    "query": "your specific question",
    "answer": "your generated answer"
}}"""

    def create_request(self, context: Dict, index: int) -> Dict:
        """Создает один запрос для batch API"""
        is_relevant = random.random() < 0.5
        relevancy_type = "relevant" if is_relevant else "irrelevant"
        reference = 1.0 if is_relevant else 0.0

        return {
            "custom_id": f"relevancy_{index}_{reference}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": self.template.format(
                        context=context['content'],
                        relevancy_type=relevancy_type
                    )}
                ],
                "max_tokens": 1000
            }
        }

    def process_batch(self, contexts: List[Dict]) -> List[Dict]:
        """Создает запросы для списка контекстов"""
        requests = []
        for i, context in enumerate(contexts):
            requests.append(self.create_request(context, i))
        return requests

class BatchProcessor:
    def __init__(self, input_dir: str, output_dir: str, total_batches: int, batch_size: int):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.total_batches = total_batches
        self.batch_size = batch_size
        self.generator = BatchGenerator()
        os.makedirs(output_dir, exist_ok=True)

    def _generate_batch_id(self, content: str) -> str:
        """Генерирует уникальный ID для batch файла"""
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def load_contexts(self) -> List[Dict]:
        """Загружает контексты из всех файлов"""
        contexts = []
        for filename in sorted(os.listdir(self.input_dir)):
            if filename.startswith('contexts_batch_') and filename.endswith('.json'):
                file_path = os.path.join(self.input_dir, filename)
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    contexts.extend(data['contexts'])
                    if len(contexts) >= self.total_batches:
                        break
        return contexts[:self.total_batches]

    def save_batch_file(self, requests: List[Dict], batch_number: int) -> str:
        """Сохраняет batch и возвращает имя файла"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        content = '\n'.join(json.dumps(req) for req in requests)
        batch_id = self._generate_batch_id(content)

        filename = f"batch_{batch_id}_{batch_number}_{timestamp}.jsonl"
        output_path = os.path.join(self.output_dir, filename)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)

        return filename

    def process(self) -> Dict:
        """Обрабатывает контексты и создает батчи"""
        # Загружаем и перемешиваем контексты
        contexts = self.load_contexts()
        random.shuffle(contexts)

        if not contexts:
            raise ValueError(f"No contexts found in {self.input_dir}")

        if len(contexts) < self.total_batches:
            print(f"Warning: Only {len(contexts)} contexts available")
            self.total_batches = len(contexts)

        stats = {
            'total_batches': self.total_batches,
            'batches_per_file': self.batch_size,
            'total_files': math.ceil(self.total_batches / self.batch_size),
            'files': []
        }

        # Разбиваем на файлы
        for i in range(0, self.total_batches, self.batch_size):
            batch_contexts = contexts[i:min(i + self.batch_size, self.total_batches)]
            if not batch_contexts:
                break

            requests = self.generator.process_batch(batch_contexts)
            batch_file = self.save_batch_file(requests, i // self.batch_size)

            relevant = sum(1 for req in requests if "_1.0" in req["custom_id"])
            file_stats = {
                'filename': batch_file,
                'requests': len(requests),
                'relevant': relevant,
                'irrelevant': len(requests) - relevant
            }
            stats['files'].append(file_stats)

        return stats

def main():
    parser = argparse.ArgumentParser(description='Generate batches for OpenAI API')
    parser.add_argument('--total', type=int, required=True,
                      help='Total number of batches to generate')
    parser.add_argument('--batch_size', type=int, default=1000,
                      help='Number of batches per file (default: 1000)')
    parser.add_argument('--input_dir', type=str, default='clean_contexts',
                      help='Input directory with context files')
    parser.add_argument('--output_dir', type=str, default='api_batches',
                      help='Output directory for batch files')

    args = parser.parse_args()

    processor = BatchProcessor(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        total_batches=args.total,
        batch_size=args.batch_size
    )

    print("\nStarting batch generation...")
    print(f"Total batches to generate: {args.total}")
    print(f"Batches per file: {args.batch_size}")

    try:
        stats = processor.process()

        print("\nGeneration completed!")
        print(f"\nGenerated {stats['total_batches']} batches in {len(stats['files'])} files")
        print("\nFiles generated:")

        total_relevant = 0
        total_requests = 0

        for file_stat in stats['files']:
            print(f"\n{file_stat['filename']}:")
            print(f"  Requests: {file_stat['requests']}")
            print(f"  Relevant: {file_stat['relevant']}")
            print(f"  Irrelevant: {file_stat['irrelevant']}")
            print(f"  Relevance ratio: {file_stat['relevant']/file_stat['requests']:.2%}")

            total_relevant += file_stat['relevant']
            total_requests += file_stat['requests']

        print(f"\nOverall statistics:")
        print(f"Total requests: {total_requests}")
        print(f"Total relevant: {total_relevant}")
        print(f"Overall relevance ratio: {total_relevant/total_requests:.2%}")

    except Exception as e:
        print(f"\nError during generation: {str(e)}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
