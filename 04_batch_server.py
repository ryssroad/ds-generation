import os
import time
import json
import logging
from datetime import datetime
from typing import List, Dict, Optional
from openai import OpenAI
import watchdog.observers
import watchdog.events
from pathlib import Path

class BatchMonitor:
    """Монитор для отслеживания и управления батчами OpenAI"""
    
    def __init__(self, api_key: str, 
                 input_dir: str = "api_batches",
                 output_dir: str = "batch_results",
                 log_dir: str = "logs"):
        self.client = OpenAI(api_key=api_key)
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.log_dir = Path(log_dir)
        
        # Создаем необходимые директории
        self.input_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)
        
        # Настраиваем логирование
        self.setup_logging()
        
        # Активные батчи и их статусы
        self.active_batches: Dict[str, Dict] = {}
        
    def setup_logging(self):
        """Настройка логирования"""
        log_file = self.log_dir / f"batch_monitor_{datetime.now().strftime('%Y%m%d_%H%M')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def upload_batch_file(self, file_path: Path) -> Optional[str]:
        """Загружает файл батча в OpenAI"""
        try:
            self.logger.info(f"Uploading batch file: {file_path}")
            
            with open(file_path, 'rb') as f:
                response = self.client.files.create(
                    file=f,
                    purpose="batch"
                )
            return response.id
            
        except Exception as e:
            self.logger.error(f"Error uploading file {file_path}: {e}")
            return None
            
    def create_batch(self, file_id: str) -> Optional[str]:
        """Создает новый batch в OpenAI"""
        try:
            self.logger.info(f"Creating batch for file: {file_id}")
            
            batch = self.client.batches.create(
                input_file_id=file_id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
            )
            
            return batch.id
            
        except Exception as e:
            self.logger.error(f"Error creating batch for file {file_id}: {e}")
            return None
            
    def check_batch_status(self, batch_id: str) -> Optional[Dict]:
        """Проверяет статус batch"""
        try:
            batch = self.client.batches.retrieve(batch_id)
            
            status = {
                'status': batch.status,
                'total': batch.request_counts.total,
                'completed': batch.request_counts.completed,
                'failed': batch.request_counts.failed,
                'output_file_id': batch.output_file_id,
                'error_file_id': batch.error_file_id
            }
            
            self.logger.info(f"Batch {batch_id} status: {status}")
            return status
            
        except Exception as e:
            self.logger.error(f"Error checking batch {batch_id}: {e}")
            return None
            
    def download_batch_results(self, batch_id: str, file_id: str, is_error: bool = False):
        """Скачивает результаты или ошибки batch"""
        try:
            self.logger.info(f"Downloading {'error' if is_error else 'results'} for batch {batch_id}")
            
            response = self.client.files.content(file_id)
            
            # Формируем имя файла
            file_type = 'errors' if is_error else 'results'
            timestamp = datetime.now().strftime('%Y%m%d_%H%M')
            output_file = self.output_dir / f"batch_{batch_id}_{file_type}_{timestamp}.jsonl"
            
            with open(output_file, 'w') as f:
                f.write(response.text)
                
            self.logger.info(f"Saved to {output_file}")
            
        except Exception as e:
            self.logger.error(f"Error downloading {file_type} for batch {batch_id}: {e}")
    
    def process_completed_batch(self, batch_id: str, status: Dict):
        """Обрабатывает завершенный batch"""
        # Скачиваем результаты
        if status['output_file_id']:
            self.download_batch_results(batch_id, status['output_file_id'])
            
        # Скачиваем ошибки если есть
        if status['error_file_id']:
            self.download_batch_results(batch_id, status['error_file_id'], is_error=True)
            
        # Удаляем из активных
        if batch_id in self.active_batches:
            del self.active_batches[batch_id]
    
    def scan_input_directory(self):
        """Сканирует директорию на наличие новых файлов для обработки"""
        self.logger.info(f"Scanning {self.input_dir} for new files")
        
        for file_path in self.input_dir.glob("*.jsonl"):
            if file_path.stem not in self.processed_files:
                self.logger.info(f"Found new file: {file_path}")
                
                # Загружаем файл
                file_id = self.upload_batch_file(file_path)
                if not file_id:
                    continue
                    
                # Создаем batch
                batch_id = self.create_batch(file_id)
                if not batch_id:
                    continue
                    
                # Добавляем в активные
                self.active_batches[batch_id] = {
                    'file_path': file_path,
                    'file_id': file_id,
                    'start_time': datetime.now()
                }
                
                self.processed_files.add(file_path.stem)
    
    def monitor_active_batches(self):
        """Мониторит активные батчи"""
        for batch_id in list(self.active_batches.keys()):
            status = self.check_batch_status(batch_id)
            if not status:
                continue
                
            if status['status'] in ['completed', 'failed', 'expired', 'cancelled']:
                self.process_completed_batch(batch_id, status)
    
    def run(self, check_interval: int = 60):
        """Запускает мониторинг"""
        self.logger.info("Starting batch monitor")
        self.processed_files = set()
        
        while True:
            try:
                # Проверяем активные батчи
                self.monitor_active_batches()
                
                # Если есть место в очереди, ищем новые файлы
                if len(self.active_batches) < 5:  # Максимум 5 активных батчей
                    self.scan_input_directory()
                
                time.sleep(check_interval)
                
            except KeyboardInterrupt:
                self.logger.info("Stopping batch monitor")
                break
            except Exception as e:
                self.logger.error(f"Error in main loop: {e}")
                time.sleep(check_interval)

if __name__ == "__main__":
    # Загружаем API ключ из переменных окружения
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")
        
    # Создаем и запускаем монитор
    monitor = BatchMonitor(api_key)
    monitor.run()