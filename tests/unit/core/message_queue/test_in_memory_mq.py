import pytest
import asyncio
from chimera.core.message_queue.in_memory_mq import InMemoryMQClient

@pytest.mark.asyncio
async def test_publish_subscribe_single_message():
    mq = InMemoryMQClient()
    received_message = None
    event = asyncio.Event()

    async def callback(msg):
        nonlocal received_message
        received_message = msg
        event.set()

    await mq.subscribe("test_topic", callback)
    await mq.publish("test_topic", {"data": "hello"})
    
    await asyncio.wait_for(event.wait(), timeout=1.0) 
    
    assert received_message == {"data": "hello"}
    await mq.stop_all_listeners()

@pytest.mark.asyncio
async def test_publish_before_subscribe():
    mq = InMemoryMQClient()
    await mq.publish("test_topic_presub", {"data": "world"}) 

    received_message = None
    event = asyncio.Event()
    async def callback(msg):
        nonlocal received_message
        received_message = msg
        event.set()

    await mq.subscribe("test_topic_presub", callback) 
    
    await asyncio.wait_for(event.wait(), timeout=1.0)
    assert received_message == {"data": "world"}
    await mq.stop_all_listeners()

@pytest.mark.asyncio
async def test_multiple_subscribers():
    mq = InMemoryMQClient()
    messages_received_cb1 = []
    messages_received_cb2 = []
    event1 = asyncio.Event()
    event2 = asyncio.Event()

    async def callback1(msg):
        messages_received_cb1.append(msg)
        if msg == {"data": "last"}: event1.set()

    async def callback2(msg):
        messages_received_cb2.append(msg)
        if msg == {"data": "last"}: event2.set()

    await mq.subscribe("multi_topic", callback1)
    await mq.subscribe("multi_topic", callback2)

    await mq.publish("multi_topic", {"data": "first"})
    await mq.publish("multi_topic", {"data": "last"})

    await asyncio.wait_for(asyncio.gather(event1.wait(), event2.wait()), timeout=2.0)

    assert messages_received_cb1 == [{"data": "first"}, {"data": "last"}]
    assert messages_received_cb2 == [{"data": "first"}, {"data": "last"}]
    await mq.stop_all_listeners()

@pytest.mark.asyncio
async def test_unsubscribe():
    mq = InMemoryMQClient()
    received_messages = []
    event = asyncio.Event()
    # Use a counter for the event to handle multiple expected messages before unsubscribe
    expected_messages_before_unsub = 1 

    async def callback(msg):
        received_messages.append(msg)
        if len(received_messages) >= expected_messages_before_unsub:
             event.set()


    await mq.subscribe("unsub_topic", callback)
    await mq.publish("unsub_topic", {"data": "one"})
    await asyncio.wait_for(event.wait(), timeout=1.0)
    assert len(received_messages) == 1

    await mq.unsubscribe("unsub_topic", callback)
    await asyncio.sleep(0.05) # Allow time for listener task cancellation if it's the last one
    
    await mq.publish("unsub_topic", {"data": "two"}) 
    
    await asyncio.sleep(0.1) 
    assert len(received_messages) == 1, "Message received after unsubscribe"
    
    if not mq.subscribers.get("unsub_topic"): 
        assert await mq.get_queue_size("unsub_topic") == 1, "Message 'two' was not in queue or was processed by unsubscribed callback"
    
    await mq.stop_all_listeners()


@pytest.mark.asyncio
async def test_stop_all_listeners_cancels_tasks():
    mq = InMemoryMQClient()
    event = asyncio.Event()
    async def callback(msg):
        event.set() 

    await mq.subscribe("stop_test_topic", callback)
    assert "stop_test_topic" in mq._listener_tasks
    listener_task_before_stop = mq._listener_tasks["stop_test_topic"]
    
    await mq.stop_all_listeners()
    
    assert listener_task_before_stop.cancelled() or listener_task_before_stop.done()
    
    await mq.publish("stop_test_topic", {"data": "after_stop"})
    await asyncio.sleep(0.01) 
    assert not event.is_set()
    assert await mq.get_queue_size("stop_test_topic") == 1

@pytest.mark.asyncio
async def test_publish_type_errors():
    mq = InMemoryMQClient()
    with pytest.raises(TypeError, match="Topic must be a string"):
        await mq.publish(123, {"data": "test"})
    await mq.stop_all_listeners()

@pytest.mark.asyncio
async def test_subscribe_type_errors():
    mq = InMemoryMQClient()
    async def dummy_callback(msg): pass
    with pytest.raises(TypeError, match="Topic must be a string"):
        await mq.subscribe(123, dummy_callback)
    with pytest.raises(TypeError, match="Callback must be callable"):
        await mq.subscribe("test_topic", "not_a_callable")
    await mq.stop_all_listeners()

@pytest.mark.asyncio
async def test_unsubscribe_type_errors():
    mq = InMemoryMQClient()
    async def dummy_callback(msg): pass
    await mq.subscribe("another_test_topic", dummy_callback) 
    with pytest.raises(TypeError, match="Topic must be a string"):
        await mq.unsubscribe(123, dummy_callback)
    with pytest.raises(TypeError, match="Callback must be callable"):
        await mq.unsubscribe("another_test_topic", "not_a_callable")
    await mq.stop_all_listeners() 

@pytest.mark.asyncio
async def test_wait_for_queues_empty():
    mq = InMemoryMQClient()
    await mq.publish("q_empty_test_1", {"data": "message1"})
    await mq.publish("q_empty_test_2", {"data": "message2"})

    assert await mq.get_queue_size("q_empty_test_1") == 1
    assert await mq.get_queue_size("q_empty_test_2") == 1
    
    # Define a consumer for the test
    async def consume_from_queue(topic_name):
        await mq.queues[topic_name].get()
        mq.queues[topic_name].task_done()

    # Create tasks to consume the messages, which will empty the queues
    asyncio.create_task(consume_from_queue("q_empty_test_1"))
    asyncio.create_task(consume_from_queue("q_empty_test_2"))

    assert await mq.wait_for_queues_empty(["q_empty_test_1", "q_empty_test_2"], timeout=1.0)

    # Test timeout
    await mq.publish("q_empty_test_3", {"data": "message3"}) # This queue won't be consumed by a task
    assert not await mq.wait_for_queues_empty(["q_empty_test_3"], timeout=0.05) 

    await mq.stop_all_listeners()
