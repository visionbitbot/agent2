from abc import ABC, abstractmethod
import asyncio
from chimera.core.message_queue.in_memory_mq import InMemoryMQClient 

class BaseIngestor(ABC):
    def __init__(self, data_source_name: str, mq_client: InMemoryMQClient, output_topic: str):
        if not isinstance(data_source_name, str) or not data_source_name:
            raise ValueError("Data source name must be a non-empty string.")
        if not isinstance(mq_client, InMemoryMQClient): 
            raise TypeError("mq_client must be an instance of InMemoryMQClient or a compatible MQ client.")
        if not isinstance(output_topic, str) or not output_topic:
            raise ValueError("Output topic must be a non-empty string.")
            
        self.data_source_name = data_source_name
        self.mq_client = mq_client
        self.output_topic = output_topic
        self._running = False
        self._ingestion_task: asyncio.Task | None = None

    @abstractmethod
    async def _connect(self):
        pass

    @abstractmethod
    async def _listen_and_process(self):
        pass

    async def start(self):
        if self._running:
            return
            
        await self._connect() 
        self._running = True
        self._ingestion_task = asyncio.create_task(self._listen_and_process())

    async def stop(self):
        if not self._running:
            return

        self._running = False
        if self._ingestion_task and not self._ingestion_task.done():
            self._ingestion_task.cancel()
            try:
                await self._ingestion_task
            except asyncio.CancelledError:
                pass
        self._ingestion_task = None

    async def _publish_data(self, data: dict):
        if not isinstance(data, dict):
            return 
        await self.mq_client.publish(self.output_topic, data)
