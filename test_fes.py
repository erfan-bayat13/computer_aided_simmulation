import unittest
from datetime import datetime
from typing import List
from uber_stratch import *


class TestFutureEventSet(unittest.TestCase):
    def setUp(self):
        """Setup test environment before each test"""
        self.city = city_map((10, 10), num_hotspots=1)
        self.network = uber_network(
            city_map=self.city,
            client_arrival_rate=0.1,
            driver_arrival_rate=0.05,
            simulation_time=1000,
            pre_simulation_driver=5
        )
        
    def test_event_scheduling_order(self):
        """Test if events are processed in correct time order"""
        # Schedule events in random order
        current_time = self.network.current_time
        
        # Create test events with different times
        client1 = uber_client.generate_client(self.city, current_time + 10.0)
        client2 = uber_client.generate_client(self.city, current_time + 5.0)
        driver1 = uber_driver.generate_driver(self.city)
        
        # Schedule events
        self.network.FES.schedule_client_arrival(current_time + 10.0, client1)
        self.network.FES.schedule_client_arrival(current_time + 5.0, client2)
        self.network.FES.schedule_driver_arrival(current_time + 7.0, driver1)
        
        # Verify events come out in correct order
        expected_times = [current_time + 5.0, current_time + 7.0, current_time + 10.0]
        actual_times = []
        
        while not self.network.FES.is_empty():
            event, _, _ = self.network.FES.get_next_event()
            if event:
                actual_times.append(event.time)
        
        self.assertEqual(expected_times, actual_times)

    def test_event_cancellation(self):
        """Test event cancellation functionality"""
        current_time = self.network.current_time
        
        # Schedule a client arrival
        client = uber_client.generate_client(self.city, current_time + 5.0)
        client_id = self.network.FES.register_client(client)
        
        # Schedule events
        self.network.FES.schedule_client_arrival(current_time + 5.0, client)
        self.network.FES.schedule_client_cancellation(current_time + 10.0, client)
        
        # Cancel all client's events
        self.network.FES.cancel_entity_events(client_id)
        
        # Verify no events are processed
        events_processed = 0
        while not self.network.FES.is_empty():
            event, _, _ = self.network.FES.get_next_event()
            if event and not event.info.cancelled:
                events_processed += 1
        
        self.assertEqual(0, events_processed)

    def test_registry_management(self):
        """Test client and driver registry functionality"""
        # Register entities
        client = uber_client.generate_client(self.city, self.network.current_time)
        driver = uber_driver.generate_driver(self.city)
        
        client_id = self.network.FES.register_client(client)
        driver_id = self.network.FES.register_driver(driver)
        
        # Verify retrieval
        self.assertEqual(client, self.network.FES.get_client(client_id))
        self.assertEqual(driver, self.network.FES.get_driver(driver_id))
        
        # Test cleanup
        self.network.FES.cleanup_registry(client_id)
        self.assertIsNone(self.network.FES.get_client(client_id))
        
        self.network.FES.cleanup_registry(driver_id)
        self.assertIsNone(self.network.FES.get_driver(driver_id))

    def test_time_validation(self):
        """Test that events cannot be scheduled in the past"""
        # Set current time
        self.network.current_time = 10.0
        self.network.FES.current_time = 10.0
        
        # Try to schedule event in the past
        client = uber_client.generate_client(self.city, 5.0)
        self.network.FES.schedule_client_arrival(5.0, client)
        
        # Verify event was adjusted to current time
        event, _, _ = self.network.FES.get_next_event()
        self.assertGreaterEqual(event.time, 10.0)

    def test_complete_event_chain(self):
        """Test that events trigger appropriate follow-up events"""
        current_time = self.network.current_time
        
        # Create and schedule initial client arrival
        client = uber_client.generate_client(self.city, current_time)
        driver = uber_driver.generate_driver(self.city)
        
        # Schedule initial events
        self.network.FES.schedule_client_arrival(current_time, client)
        self.network.FES.schedule_driver_arrival(current_time + 1, driver)
        
        # Process the first few events
        events_processed = []
        for _ in range(3):  # Process first 3 events
            event, _, _ = self.network.FES.get_next_event()
            if event:
                events_processed.append(event.event_type)
                # Let network handle the event
                if event.event_type == EventType.CLIENT_ARRIVAL:
                    self.network.handle_client_arrival(event, client)
                elif event.event_type == EventType.DRIVER_ARRIVAL:
                    self.network.handle_driver_arrival(event, driver)
        
        # Verify that client arrival and driver arrival were processed
        self.assertIn(EventType.CLIENT_ARRIVAL, events_processed)
        self.assertIn(EventType.DRIVER_ARRIVAL, events_processed)
        
        # Verify that FES isn't empty (should have follow-up events scheduled)
        self.assertFalse(self.network.FES.is_empty())

def run_fes_tests():
    """Run all FES tests and print results"""
    suite = unittest.TestLoader().loadTestsFromTestCase(TestFutureEventSet)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()

if __name__ == '__main__':
    success = run_fes_tests()
    print(f"\nAll tests {'passed' if success else 'failed'}")