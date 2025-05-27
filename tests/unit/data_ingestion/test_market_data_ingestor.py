import pytest
import asyncio
import csv
from pathlib import Path
from unittest.mock import AsyncMock
import pandas as pd 

from chimera.data_ingestion.base_ingestor import BaseIngestor
from chimera.data_ingestion.market_data_ingestor import CSVMarketDataIngestor
from chimera.core.message_queue.in_memory_mq import InMemoryMQClient


@pytest.fixture
def mock_mq_client_fixture(): 
    client = InMemoryMQClient()
    client.publish = AsyncMock() 
    return client

@pytest.fixture
def sample_csv_file_fixture(tmp_path: Path) -> Path: 
    csv_file = tmp_path / "test_data.csv"
    fieldnames = ["Timestamp", "Symbol", "Price", "Quantity", "ID", "ATR"]
    data = [
        {"Timestamp": "2023-01-01T10:00:00.000Z", "Symbol": "BTCUSD", "Price": "20000.50", "Quantity": "0.1", "ID": "t1", "ATR": "150.5"},
        {"Timestamp": "2023-01-01T10:00:01.000Z", "Symbol": "ETHUSD", "Price": "1500.75", "Quantity": "1.2", "ID": "t2", "ATR": "50.2"},
        {"Timestamp": "2023-01-01T10:00:02.000Z", "Symbol": "BTCUSD", "Price": "20001.00", "Quantity": "0.05", "ID": "t3", "ATR": "150.6"},
        {"Timestamp": "invalid_ts", "Symbol": "BTCUSD", "Price": "20002.00", "Quantity": "0.05", "ID": "t4", "ATR": "150.7"}, 
        {"Timestamp": "2023-01-01T10:00:03.000Z", "Symbol": "SOLUSD", "Price": "20.00", "Quantity": "10", "ID": "t5", "ATR": "1.0"}, 
        {"Timestamp": "2023-01-01T10:00:04.000Z", "Symbol": "BTCUSD", "Price": "not_a_float", "Quantity": "0.05", "ID": "t6", "ATR": "150.8"}, 
        {"Timestamp": "2023-01-01T10:00:05.000Z", "Symbol": "ETHUSD", "Price": "1502.00", "Quantity": "0.3", "ID": "t7", "ATR": ""}, 
    ]
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    return csv_file

@pytest.fixture
def concrete_base_ingestor_fixture(mock_mq_client_fixture): 
    class ConcreteIngestor(BaseIngestor):
        async def _connect(self): pass
        async def _listen_and_process(self): pass
    return ConcreteIngestor("TestDS", mock_mq_client_fixture, "test_topic")

def test_base_ingestor_constructor_valid(concrete_base_ingestor_fixture): 
    assert concrete_base_ingestor_fixture.data_source_name == "TestDS"

def test_base_ingestor_constructor_invalid_args(mock_mq_client_fixture):
    class ConcreteIngestor(BaseIngestor): 
        async def _connect(self): pass
        async def _listen_and_process(self): pass
    with pytest.raises(ValueError): ConcreteIngestor("", mock_mq_client_fixture, "topic")
    with pytest.raises(TypeError): ConcreteIngestor("TestDS", None, "topic") # type: ignore
    with pytest.raises(ValueError): ConcreteIngestor("TestDS", mock_mq_client_fixture, "")


def test_csv_ingestor_constructor_valid(mock_mq_client_fixture, sample_csv_file_fixture):
    ingestor = CSVMarketDataIngestor(str(sample_csv_file_fixture), ["BTCUSD"], mock_mq_client_fixture)
    assert ingestor.csv_file_path == sample_csv_file_fixture

def test_csv_ingestor_constructor_invalid_args(mock_mq_client_fixture, sample_csv_file_fixture):
    with pytest.raises(ValueError): CSVMarketDataIngestor("", ["BTCUSD"], mock_mq_client_fixture) # type: ignore
    # FileNotFoundError is raised in _connect(), not in the constructor.
    # This scenario is correctly tested in test_csv_ingestor_connect_file_not_found.
    with pytest.raises(ValueError): CSVMarketDataIngestor(str(sample_csv_file_fixture), [], mock_mq_client_fixture) # type: ignore
    with pytest.raises(ValueError): CSVMarketDataIngestor(str(sample_csv_file_fixture), ["BTCUSD"], mock_mq_client_fixture, loop_delay_sec=-1)
    with pytest.raises(ValueError): CSVMarketDataIngestor(str(sample_csv_file_fixture), ["BTCUSD"], mock_mq_client_fixture, batch_size=0)

@pytest.mark.asyncio
async def test_csv_ingestor_connect_file_not_found(mock_mq_client_fixture):
    ingestor = CSVMarketDataIngestor("non_existent.csv", ["BTCUSD"], mock_mq_client_fixture)
    with pytest.raises(FileNotFoundError):
        await ingestor._connect()

@pytest.mark.asyncio
async def test_csv_ingestor_process_data_and_publish(mock_mq_client_fixture, sample_csv_file_fixture):
    symbols_to_ingest = ["BTCUSD", "ETHUSD"]
    ingestor = CSVMarketDataIngestor(
        csv_file_path=str(sample_csv_file_fixture), 
        symbols=symbols_to_ingest, 
        mq_client=mock_mq_client_fixture,
        loop_delay_sec=0 
    )
    await ingestor.start() 
    if ingestor._ingestion_task:
         await asyncio.wait_for(ingestor._ingestion_task, timeout=2.0)

    assert mock_mq_client_fixture.publish.call_count == 4 
    published_datas = [call.args[1] for call in mock_mq_client_fixture.publish.call_args_list]
    
    data_t1 = next(d for d in published_datas if d["id"] == "t1")
    assert data_t1["symbol"] == "BTCUSD"
    assert data_t1["price"] == 20000.50
    assert data_t1["atr"] == 150.5

    data_t2 = next(d for d in published_datas if d["id"] == "t2")
    assert data_t2["symbol"] == "ETHUSD"
    assert data_t2["price"] == 1500.75
    assert data_t2["atr"] == 50.2

    data_t3 = next(d for d in published_datas if d["id"] == "t3")
    assert data_t3["symbol"] == "BTCUSD"
    assert data_t3["price"] == 20001.00
    assert data_t3["atr"] == 150.6
    
    data_t7 = next(d for d in published_datas if d["id"] == "t7")
    assert data_t7["symbol"] == "ETHUSD"
    assert "atr" not in data_t7 
    await ingestor.stop() 

@pytest.mark.asyncio
async def test_csv_ingestor_stop_during_processing(mock_mq_client_fixture, tmp_path): 
    long_csv_file = tmp_path / "long_test_data.csv"
    fieldnames = ["Timestamp", "Symbol", "Price", "Quantity"]
    data = [{"Timestamp": f"2023-01-01T10:00:{i:02d}.000Z", "Symbol": "BTCUSD", "Price": str(20000+i), "Quantity": "0.1"} for i in range(20)]
    with open(long_csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

    ingestor = CSVMarketDataIngestor(
        csv_file_path=str(long_csv_file), 
        symbols=["BTCUSD"], 
        mq_client=mock_mq_client_fixture,
        loop_delay_sec=0.01, 
        batch_size=1 
    )
    await ingestor.start()
    await asyncio.sleep(0.05) 
    publish_count_before_stop = mock_mq_client_fixture.publish.call_count
    assert publish_count_before_stop > 0 
    await ingestor.stop()
    await asyncio.sleep(0.05) 
    publish_count_after_stop = mock_mq_client_fixture.publish.call_count
    assert publish_count_after_stop < len(data) 
    # This assertion needs to be more precise, considering potential race conditions with async code
    # It should be publish_count_before_stop or publish_count_before_stop + 1 (if one more batch item got through)
    assert publish_count_after_stop <= publish_count_before_stop + (ingestor.batch_size if publish_count_before_stop > 0 else 0)

    assert ingestor._running is False
    assert ingestor._ingestion_task is None or ingestor._ingestion_task.done()

@pytest.mark.asyncio
async def test_csv_ingestor_empty_csv(mock_mq_client_fixture, tmp_path: Path):
    empty_csv_file = tmp_path / "empty_data.csv"
    fieldnames = ["Timestamp", "Symbol", "Price", "Quantity", "ID", "ATR"]
    with open(empty_csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader() # Only header, no data
    
    ingestor = CSVMarketDataIngestor(
        csv_file_path=str(empty_csv_file), 
        symbols=["BTCUSD"], 
        mq_client=mock_mq_client_fixture
    )
    await ingestor.start()
    if ingestor._ingestion_task:
        await asyncio.wait_for(ingestor._ingestion_task, timeout=1.0)
    
    mock_mq_client_fixture.publish.assert_not_called()
    await ingestor.stop()

@pytest.mark.asyncio
async def test_csv_ingestor_no_matching_symbols(mock_mq_client_fixture, sample_csv_file_fixture):
    ingestor = CSVMarketDataIngestor(
        csv_file_path=str(sample_csv_file_fixture), 
        symbols=["NONE"], # Symbol not in the sample CSV
        mq_client=mock_mq_client_fixture
    )
    await ingestor.start()
    if ingestor._ingestion_task:
        await asyncio.wait_for(ingestor._ingestion_task, timeout=1.0)
    
    mock_mq_client_fixture.publish.assert_not_called()
    await ingestor.stop()

@pytest.mark.asyncio
async def test_csv_ingestor_batch_processing(mock_mq_client_fixture, sample_csv_file_fixture):
    symbols_to_ingest = ["BTCUSD", "ETHUSD"]
    batch_s = 2
    ingestor = CSVMarketDataIngestor(
        csv_file_path=str(sample_csv_file_fixture), 
        symbols=symbols_to_ingest, 
        mq_client=mock_mq_client_fixture,
        loop_delay_sec=0.01, # Small delay to ensure batches are processed
        batch_size=batch_s
    )
    await ingestor.start()
    if ingestor._ingestion_task:
        # Wait long enough for all potential batches to be processed
        await asyncio.wait_for(ingestor._ingestion_task, timeout=2.0) 

    # Total valid rows for BTCUSD, ETHUSD is 4.
    # Batch 1: t1 (BTCUSD), t2 (ETHUSD) -> publish
    # Batch 2: t3 (BTCUSD), t7 (ETHUSD) -> publish
    assert mock_mq_client_fixture.publish.call_count == 4 
    
    # Check that publish was called after each batch of 2 (or remaining at the end)
    # This is a bit harder to assert directly with AsyncMock's call_count over time
    # But the total count should be correct, and loop_delay_sec implies they weren't all at once.
    await ingestor.stop()
