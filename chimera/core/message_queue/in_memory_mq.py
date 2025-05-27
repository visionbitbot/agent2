import asyncio
from collections import defaultdict
from typing import Callable, DefaultDict, List, Any, Dict

class InMemoryMQClient:
    def __init__(self):
        self.queues: DefaultDict[str, asyncio.Queue] = defaultdict(asyncio.Queue)
        self.subscribers: DefaultDict[str, List[Callable]] = defaultdict(list)
        self._listener_tasks: Dict[str, asyncio.Task] = {}
        # print("InMemoryMQClient initialized.") # Optional: for debugging

    async def publish(self, topic: str, message: Any):
        # print(f"MQ: Publishing to {topic}: {message}") # Optional: for debugging
        if not isinstance(topic, str):
            raise TypeError(f"Topic must be a string, got {type(topic)}")
        await self.queues[topic].put(message)

    async def subscribe(self, topic: str, callback: Callable):
        # print(f"MQ: Subscribing to {topic} with {callback.__name__}") # Optional: for debugging
        if not isinstance(topic, str):
            raise TypeError(f"Topic must be a string, got {type(topic)}")
        if not callable(callback):
            raise TypeError(f"Callback must be callable, got {type(callback)}")

        self.subscribers[topic].append(callback)
        if topic not in self._listener_tasks or self._listener_tasks[topic].done():
            self._listener_tasks[topic] = asyncio.create_task(self._listen(topic))
            # print(f"MQ: Created listener task for topic {topic}") # Optional: for debugging

    async def _listen(self, topic: str):
        # print(f"MQ: Listener started for {topic}") # Optional: for debugging
        while True:
            try:
                message = await self.queues[topic].get()
                # print(f"MQ: Received on {topic}: {message}") # Optional: for debugging
                
                current_subscribers = list(self.subscribers[topic]) # Iterate over a copy

                if not current_subscribers:
                    # print(f"MQ: No subscribers for {topic}, message lost: {message}") # Optional
                    self.queues[topic].task_done()
                    continue

                for callback_fn in current_subscribers: # Renamed to avoid conflict
                    try:
                        if asyncio.iscoroutinefunction(callback_fn):
                            await callback_fn(message)
                        else:
                            callback_fn(message)
                    except Exception as e:
                        print(f"MQ: Error in callback for topic {topic}: {e} while processing {message}")
                self.queues[topic].task_done()
            except asyncio.CancelledError:
                # print(f"MQ: Listener for {topic} cancelled.") # Optional
                break
            except Exception as e:
                print(f"MQ: Unexpected error in listener for {topic}: {e}")
                await asyncio.sleep(1)

    async def unsubscribe(self, topic: str, callback: Callable):
        if not isinstance(topic, str):
            raise TypeError(f"Topic must be a string, got {type(topic)}")
        if not callable(callback):
            raise TypeError(f"Callback must be callable, got {type(callback)}")
            
        if topic in self.subscribers and callback in self.subscribers[topic]:
            self.subscribers[topic].remove(callback)
            # print(f"MQ: Unsubscribed {callback.__name__} from {topic}") # Optional
        
        if topic in self.subscribers and not self.subscribers[topic] and topic in self._listener_tasks:
            task_to_cancel = self._listener_tasks[topic]
            if not task_to_cancel.done():
                task_to_cancel.cancel()
            # print(f"MQ: Cancelled listener task for topic {topic} as no subscribers left.") # Optional
            try:
                await task_to_cancel 
            except asyncio.CancelledError:
                pass 
            finally:
                if topic in self._listener_tasks and self._listener_tasks[topic] is task_to_cancel: 
                     del self._listener_tasks[topic]


    async def stop_all_listeners(self):
        for topic in list(self._listener_tasks.keys()):
            task = self._listener_tasks[topic]
            if task and not task.done():
                task.cancel()
                try:
                    await task 
                except asyncio.CancelledError:
                    pass 
        self._listener_tasks.clear()
        # print("InMemoryMQClient: All listener tasks cancelled.") # Optional

    async def get_queue_size(self, topic: str) -> int:
        if not isinstance(topic, str):
            raise TypeError(f"Topic must be a string, got {type(topic)}")
        return self.queues[topic].qsize()

    async def wait_for_queues_empty(self, topics: List[str], timeout: float = 5.0):
        if not isinstance(topics, list) or not all(isinstance(t, str) for t in topics):
            raise TypeError("Topics must be a list of strings.")

        overall_start_time = asyncio.get_event_loop().time()
        for topic in topics:
            topic_start_time = asyncio.get_event_loop().time()
            while not self.queues[topic].empty():
                if asyncio.get_event_loop().time() - topic_start_time > timeout:
                    # print(f"MQ: Timeout waiting for queue {topic} to empty. Size: {self.queues[topic].qsize()}")
                    return False 
                if asyncio.get_event_loop().time() - overall_start_time > timeout * len(topics): 
                    # print(f"MQ: Overall timeout waiting for queues to empty.")
                    return False
                await asyncio.sleep(0.01) 
        return True
