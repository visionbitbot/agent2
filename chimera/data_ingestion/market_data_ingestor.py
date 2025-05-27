import asyncio
import csv
from pathlib import Path
import pandas as pd 

from chimera.data_ingestion.base_ingestor import BaseIngestor
from chimera.core.message_queue.in_memory_mq import InMemoryMQClient

class CSVMarketDataIngestor(BaseIngestor):
    def __init__(self, csv_file_path: str, symbols: list[str], mq_client: InMemoryMQClient, 
                 output_topic: str = "raw_market_data", loop_delay_sec: float = 0.01, batch_size: int = 1):
        super().__init__("CSVMarketData", mq_client, output_topic)
        
        if not isinstance(csv_file_path, str) or not csv_file_path:
            raise ValueError("CSV file path must be a non-empty string.")
        self.csv_file_path = Path(csv_file_path)
        
        if not isinstance(symbols, list) or not all(isinstance(s, str) for s in symbols) or not symbols:
            raise ValueError("Symbols must be a non-empty list of strings.")
        self.symbols = symbols
        
        if not isinstance(loop_delay_sec, (int, float)) or loop_delay_sec < 0:
            raise ValueError("Loop delay must be a non-negative number.")
        self.loop_delay_sec = loop_delay_sec

        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("Batch size must be a positive integer.")
        self.batch_size = batch_size
        
        self._csv_reader = None
        self._file_handle = None

    async def _connect(self):
        if not self.csv_file_path.is_file():
            raise FileNotFoundError(f"CSV file not found: {self.csv_file_path}")
        try:
            self._file_handle = open(self.csv_file_path, 'r', newline='')
            self._csv_reader = csv.DictReader(self._file_handle)
        except Exception as e:
            if self._file_handle:
                self._file_handle.close()
                self._file_handle = None
            raise IOError(f"Failed to open or read CSV file {self.csv_file_path}: {e}")

    async def _listen_and_process(self):
        if not self._csv_reader:
            return 

        try:
            batch = []
            for row in self._csv_reader:
                if not self._running:
                    break 
                
                try:
                    csv_symbol = row.get("Symbol")
                    if csv_symbol not in self.symbols:
                        continue

                    timestamp_val = row.get("Timestamp")
                    try:
                        pd_timestamp = pd.to_datetime(timestamp_val)
                        timestamp_ms = int(pd_timestamp.timestamp() * 1000)
                    except ValueError:
                        continue

                    normalized_data = {
                        "timestamp_ms": timestamp_ms,
                        "symbol": csv_symbol,
                        "price": float(row.get("Price", 0.0)),
                        "quantity": float(row.get("Quantity", 0.0)),
                        "source": self.data_source_name,
                        "type": "trade", 
                        "id": row.get("ID", f"csv_{timestamp_ms}_{csv_symbol}") 
                    }
                    if "ATR" in row and row["ATR"] and row["ATR"] != "": 
                        try:
                            normalized_data["atr"] = float(row["ATR"])
                        except ValueError:
                            pass 
                    
                    batch.append(normalized_data)

                    if len(batch) >= self.batch_size:
                        for data_item in batch:
                            await self._publish_data(data_item)
                        batch = []
                        if self.loop_delay_sec > 0:
                            await asyncio.sleep(self.loop_delay_sec)
                except ValueError: 
                    continue
                except Exception: 
                    continue 

            for data_item in batch: 
                await self._publish_data(data_item)
        except Exception:
            pass 
        finally:
            if self._file_handle:
                self._file_handle.close()
                self._file_handle = None
            self._running = False 

    async def stop(self):
        await super().stop() 
        if self._file_handle: 
            self._file_handle.close()
            self._file_handle = None
