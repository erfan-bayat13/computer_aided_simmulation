import networkx as nx
import numpy as np
from queue import PriorityQueue
from dataclasses import dataclass, field
from typing import Tuple, Dict, List
import random
from collections import defaultdict
import matplotlib.pyplot as plt

@dataclass(order=True)
class EdgeWakeUpEvent:
    time: float
    edge: Tuple[int, int] = field(compare=False)

class MarkovianMajorityModel:
    def __init__(self, G: nx.Graph, edge_rates: Dict[Tuple[int, int], float] = None):
        self.G = G
        if edge_rates is None:
            # Scale rates by number of edges to get more natural time scale
            base_rate = 10.0 / G.number_of_edges()
            self.edge_rates = {e: base_rate for e in G.edges()}
        else:
            self.edge_rates = edge_rates
        self.event_queue = PriorityQueue()
        self.opinions = {}
        self.time = 0.0
        self.history = defaultdict(list)
        
    def initialize_random_opinions(self, p_positive: float = 0.5):
        self.opinions = {v: 1 if random.random() < p_positive else -1 
                        for v in self.G.nodes()}
        
        # Print initial state
        print("\nInitial opinions:")
        print(f"Number of +1: {sum(1 for v in self.opinions.values() if v == 1)}")
        print(f"Number of -1: {sum(1 for v in self.opinions.values() if v == -1)}")
        
        # Record initial state
        for v in self.G.nodes():
            self.history[v].append((0.0, self.opinions[v]))
            
        # Schedule initial events
        for edge in self.G.edges():
            first_time = random.expovariate(self.edge_rates[edge])
            self.event_queue.put(EdgeWakeUpEvent(first_time, edge))
    
    def get_local_majority(self, vertex: int) -> int:
        neighbor_sum = sum(self.opinions[w] for w in self.G.neighbors(vertex))
        if neighbor_sum > 0:
            return 1
        elif neighbor_sum < 0:
            return -1
        return self.opinions[vertex]
    
    def step(self) -> Tuple[float, List[Tuple[int, int]]]:
        if self.event_queue.empty():
            return self.time, []
            
        event = self.event_queue.get()
        self.time = event.time
        u, v = event.edge
        changed_vertices = []
        
        # Print event details
        #print(f"\nTime {self.time:.3f}: Edge {u}-{v} wakes up")
        #print(f"Before update: u({u})={self.opinions[u]}, v({v})={self.opinions[v]}")
        
        # Update opinions for both endpoints
        for vertex in (u, v):
            new_opinion = self.get_local_majority(vertex)
            if new_opinion != self.opinions[vertex]:
                old_opinion = self.opinions[vertex]
                self.opinions[vertex] = new_opinion
                self.history[vertex].append((self.time, new_opinion))
                changed_vertices.append(vertex)
                #print(f"Vertex {vertex} changed opinion: {old_opinion} -> {new_opinion}")
        
        #if not changed_vertices:
            #print("No opinions changed")
        
        # Schedule next wake-up for this edge
        next_time = self.time + random.expovariate(self.edge_rates[event.edge])
        self.event_queue.put(EdgeWakeUpEvent(next_time, event.edge))
        
        return self.time, changed_vertices
    
    def get_opinion_counts(self) -> Tuple[int, int]:
        pos_count = sum(1 for v in self.opinions.values() if v == 1)
        neg_count = len(self.opinions) - pos_count
        return pos_count, neg_count
    
    def simulate(self, max_time: float = None, max_events: int = None) -> Dict:
        num_events = 0
        last_change_time = 0.0
        
        print("\nStarting simulation...")
        
        # Track opinion counts over time
        times = [0.0]
        pos_counts = [sum(1 for v in self.opinions.values() if v == 1)]
        
        while True:
            if max_time and self.time >= max_time:
                print(f"\nReached max time: {max_time}")
                break
            if max_events and num_events >= max_events:
                print(f"\nReached max events: {max_events}")
                break
            if self.event_queue.empty():
                print("\nNo more events in queue")
                break
                
            time, changed = self.step()
            num_events += 1
            
            # Track opinion counts
            pos_count, neg_count = self.get_opinion_counts()
            times.append(time)
            pos_counts.append(pos_count)
            
            
            
            if changed:
                last_change_time = time
                
            # Check for consensus
            if pos_count == len(self.opinions) or neg_count == len(self.opinions):
                print("\nReached consensus!")
                break
        
        # Plot opinion evolution
        plt.figure(figsize=(10, 6))
        plt.plot(times, pos_counts, 'b-', label='+1 opinions')
        plt.plot(times, [len(self.opinions) - pc for pc in pos_counts], 'r-', label='-1 opinions')
        plt.xlabel('Time')
        plt.ylabel('Number of vertices')
        plt.title('Opinion Evolution')
        plt.legend()
        plt.grid(True)
        plt.show()
                
        return {
            'final_opinions': self.opinions.copy(),
            'history': dict(self.history),
            'num_events': num_events,
            'final_time': self.time,
            'last_change_time': last_change_time
        }
    
    def get_consensus_status(self) -> Tuple[bool, int]:
        values = set(self.opinions.values())
        if len(values) == 1:
            return True, values.pop()
        return False, None

# Example usage with more detailed output
if __name__ == "__main__":
    # Create a small graph for testing
    n = 10000  # 10^4 vertices
    p = 3 * np.log(n) / n  # Edge probability
    G = nx.fast_gnp_random_graph(n, p)
    
    print(f"\nCreated random graph:")
    print(f"Vertices: {n}")
    print(f"Edges: {G.number_of_edges()}")
    
    # Initialize model
    model = MarkovianMajorityModel(G)
    model.initialize_random_opinions(p_positive=0.5)
    
    # Run simulation
    results = model.simulate(max_time=100.0)
    
    # Print final results
    print("\nSimulation complete!")
    print(f"Number of events: {results['num_events']}")
    print(f"Final time: {results['final_time']:.2f}")
    print(f"Last opinion change at: {results['last_change_time']:.2f}")
    
    has_consensus, consensus_value = model.get_consensus_status()
    print(f"\nReached consensus: {has_consensus}")
    if has_consensus:
        print(f"Consensus value: {consensus_value}")
    
    final_pos, final_neg = model.get_opinion_counts()
    print(f"\nFinal opinion counts:")
    print(f"+1: {final_pos}")
    print(f"-1: {final_neg}")