import json
import pandas as pd
import argparse
from typing import List, Dict
from datetime import datetime
import os
from tqdm import tqdm
from pathlib import Path

class BatchResponseProcessor:
    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.required_fields = [
            'task', 'api', 'model_id', 'rag_context',
            'query', 'llm_response', 'reference'
        ]

    def process_single_response(self, response: Dict) -> Dict:
        """Обрабатывает один ответ API"""
        try:
            if response.get('response', {}).get('status_code') != 200:
                return None

            body = response['response']['body']
            gpt_response = json.loads(body['choices'][0]['message']['content'].strip('`json\n'))

            return {
                'task': 'relevancy',
                'api': 'LLMAPIs.OPENAI',
                'model_id': body['model'],
                'rag_context': gpt_response['rag_context'],
                'query': gpt_response['query'],
                'llm_response': gpt_response['answer'],
                'reference': float(response['custom_id'].split('_')[-1])
            }
        except Exception as e:
            print(f"Error processing response: {str(e)}")
            return None

    def validate_data(self, data: Dict) -> bool:
        """Проверяет наличие всех необходимых полей"""
        if not data:
            return False
        return all(field in data for field in self.required_fields)

    def process_file(self, input_file: str) -> List[Dict]:
        """Обрабатывает один файл с ответами"""
        processed_data = []
        error_count = 0

        try:
            with open(input_file, 'r') as f:
                for line_num, line in enumerate(f):
                    try:
                        response = json.loads(line)
                        processed = self.process_single_response(response)
                        if processed and self.validate_data(processed):
                            processed_data.append(processed)
                        else:
                            error_count += 1
                    except json.JSONDecodeError:
                        print(f"Error decoding JSON at line {line_num + 1}")
                        error_count += 1
                        continue
        except Exception as e:
            print(f"Error reading file {input_file}: {str(e)}")

        return processed_data, error_count

    def process_directory(self) -> bool:
        """Обрабатывает все файлы в директории"""
        all_processed_data = []
        total_error_count = 0
        files_processed = 0

        # Находим все JSONL файлы
        input_files = list(Path(self.input_dir).glob('*.jsonl'))

        if not input_files:
            print(f"No JSONL files found in {self.input_dir}")
            return False

        print(f"\nFound {len(input_files)} files to process")

        # Обрабатываем каждый файл
        for input_file in tqdm(input_files, desc="Processing files"):
            processed_data, error_count = self.process_file(input_file)
            all_processed_data.extend(processed_data)
            total_error_count += error_count
            files_processed += 1

        if not all_processed_data:
            print("No valid responses found in any file")
            return False

        # Сохраняем результаты
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        output_file = os.path.join(self.output_dir, f'processed_data_{timestamp}.csv')

        try:
            df = pd.DataFrame(all_processed_data)
            df.to_csv(output_file, index=False)

            print(f"\nProcessing completed:")
            print(f"Files processed: {files_processed}")
            print(f"Total responses: {len(all_processed_data)}")
            print(f"Total errors: {total_error_count}")

            relevant = (df['reference'] == 1.0).sum()
            print(f"\nDataset statistics:")
            print(f"Relevant responses: {relevant}")
            print(f"Irrelevant responses: {len(df) - relevant}")
            print(f"Relevance ratio: {relevant/len(df):.2%}")

            print(f"\nOutput saved to: {output_file}")
            return True

        except Exception as e:
            print(f"Error saving results: {str(e)}")
            return False

def main():
    parser = argparse.ArgumentParser(description='Process OpenAI API responses')
    parser.add_argument('--input_dir', type=str, required=True,
                      help='Input directory with JSONL files')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Output directory for CSV files')

    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory {args.input_dir} not found")
        return 1

    processor = BatchResponseProcessor(args.input_dir, args.output_dir)
    success = processor.process_directory()

    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
