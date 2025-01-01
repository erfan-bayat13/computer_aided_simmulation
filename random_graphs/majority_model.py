import networkx as nx
import numpy as np
from queue import PriorityQueue
from dataclasses import dataclass, field
from typing import Tuple, Dict, List
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

@dataclass(order=True)
class UpdateEvent:
    time: float
    vertex: int = field(compare=False)

class MarkovianMajorityModel:
    def __init__(self, G: nx.Graph, vertex_rates: Dict[int, float] = None):
        self.G = G
        self.avg_degree = 2 * G.number_of_edges() / G.number_of_nodes()
        if vertex_rates is None:
            base_rate = 1.0 
            self.vertex_rates = {v: base_rate for v in G.nodes()}
        else:
            self.vertex_rates = vertex_rates
        self.event_queue = PriorityQueue()
        self.opinions = {}
        self.time = 0.0
        self.history = defaultdict(list)

    def initialize_random_opinions(self, p_positive: float = 0.5):
        self.opinions = {v: 1 if random.random() < p_positive else -1 
                        for v in self.G.nodes()}
        for v in self.G.nodes():
            self.history[v].append((0.0, self.opinions[v]))
        self._schedule_all_updates()

    def _schedule_all_updates(self):
        """Schedule updates for all vertices"""
        for vertex in self.G.nodes():
            # Generate a time delta
            time_delta = random.expovariate(self.vertex_rates[vertex])
            # Add the delta to the current time
            event_time = self.time + time_delta
            self.event_queue.put(UpdateEvent(event_time, vertex))

    def get_local_majority(self, vertex: int) -> int:
        neighbors = list(self.G.neighbors(vertex))
        if not neighbors:
            return self.opinions[vertex]
        weighted_sum = sum(self.opinions[w] * self.G.degree(w) for w in neighbors)
        if weighted_sum > 0:
            return 1
        elif weighted_sum < 0:
            return -1
        else:
            return self.opinions[vertex]

    def step(self) -> Tuple[float, List[int]]:
        if self.event_queue.empty():
            return self.time, []
        event = self.event_queue.get()
        # Update simulation time from the event
        self.time = event.time  
        vertex = event.vertex
        changed_vertices = []
        new_opinion = self.get_local_majority(vertex)
        if new_opinion != self.opinions[vertex]:
            self.opinions[vertex] = new_opinion
            self.history[vertex].append((self.time, new_opinion))
            changed_vertices.append(vertex)
        # Schedule next update for this vertex using time delta
        time_delta = random.expovariate(self.vertex_rates[vertex])
        next_time = self.time + time_delta
        self.event_queue.put(UpdateEvent(next_time, vertex))
        return self.time, changed_vertices

    def get_opinion_counts(self) -> Tuple[int, int]:
        pos_count = sum(1 for v in self.opinions.values() if v == 1)
        neg_count = len(self.opinions) - pos_count
        return pos_count, neg_count

    def simulate(self, max_time: float = None, max_events: int = None,plot=True) -> Dict:
        num_events = 0
        last_change_time = 0.0
        print("\nStarting simulation...")
        print(f"Initial state - Positive: {sum(1 for v in self.opinions.values() if v == 1)}, " 
              f"Negative: {sum(1 for v in self.opinions.values() if v == -1)}")
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
            if changed:
                last_change_time = time
                pos_count, neg_count = self.get_opinion_counts()
                times.append(time)
                pos_counts.append(pos_count)
                if pos_count == len(self.opinions):
                    print(f"\nConsensus reached at time {time:.2f}: All positive")
                    break
                elif neg_count == len(self.opinions):
                    print(f"\nConsensus reached at time {time:.2f}: All negative")
                    break
                if num_events % 1000 == 0:
                    print(f"Time {time:.2f}: +1: {pos_count}, -1: {neg_count}")
        if plot:
            self._plot_opinion_evolution(times, pos_counts)
        return {
            'final_opinions': self.opinions.copy(),
            'history': dict(self.history),
            'num_events': num_events,
            'final_time': self.time,
            'last_change_time': last_change_time
        }

    def _plot_opinion_evolution(self, times, pos_counts):
        plt.figure(figsize=(10, 6))
        plt.plot(times, pos_counts, 'b-', label='+1 opinions')
        plt.plot(times, [len(self.opinions) - pc for pc in pos_counts], 
                'r-', label='-1 opinions')
        plt.xlabel('Time')
        plt.ylabel('Number of vertices')
        plt.title('Opinion Evolution')
        plt.legend()
        plt.grid(True)
        plt.show()

    def get_consensus_status(self) -> Tuple[bool, int]:
        values = set(self.opinions.values())
        return (True, values.pop()) if len(values) == 1 else (False, None)
    

def run_majority_analysis(n=10000, trials=10, max_time=15.0):
    initial_probs = np.array([0.5, 0.6, 0.7, 0.8])
    consensus_probs = []
    consensus_times = []
    time_stds = []
    evolution_data = {p: [] for p in initial_probs}
    
    for p_init in initial_probs:
        successes = 0
        times = []
        
        for _ in range(trials):
            G = nx.fast_gnp_random_graph(n, 3 * np.log(n) / n)
            model = MarkovianMajorityModel(G)
            model.initialize_random_opinions(p_positive=p_init)
            
            time_points = []
            ratios = []
            current_time = 0
            current_ratio = p_init
            
            while current_time < max_time:
                time, changed = model.step()
                if changed:
                    pos_count = sum(1 for v in model.opinions.values() if v == 1)
                    current_ratio = pos_count / n
                    time_points.append(time)
                    ratios.append(current_ratio)
                current_time = time
                
                if current_ratio in (0, 1):
                    break
            
            if current_ratio == 1:
                successes += 1
                times.append(current_time)
            
            evolution_data[p_init].append((np.array(time_points), np.array(ratios)))
        
        consensus_probs.append(successes / trials)
        consensus_times.append(np.mean(times) if times else max_time)
        time_stds.append(np.std(times) if times else 0)

    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1: Consensus Probability
    ax1.plot(initial_probs, consensus_probs, 'o-', color='dodgerblue', markersize=8, linewidth=2)
    ax1.set_xlabel('Initial Probability of +1 Opinion', fontsize=12)
    ax1.set_ylabel('Probability of +1 Consensus', fontsize=12)
    ax1.set_title('Consensus Probability', fontsize=14)
    ax1.grid(visible=True, linestyle='--', alpha=0.7)
    
    # Plot 2: Time to Consensus
    ax2.errorbar(initial_probs, consensus_times, yerr=time_stds, fmt='o-', color='darkorange', 
                 ecolor='gray', elinewidth=1.5, capsize=4, markersize=8, linewidth=2)
    ax2.set_xlabel('Initial Probability of +1 Opinion', fontsize=12)
    ax2.set_ylabel('Steps to Consensus', fontsize=12)
    ax2.set_title('Time to Consensus', fontsize=14)
    ax2.grid(visible=True, linestyle='--', alpha=0.7)
    
    # Plot 3: Opinion Evolution
    colors = sns.color_palette("viridis", len(initial_probs))
    for p_init, color in zip(initial_probs, colors):
        time_points, ratios = evolution_data[p_init][0]
        ax3.plot(time_points, ratios, color=color, label=f'pi={p_init:.1f}', linewidth=2)
    
    ax3.set_xlabel('Step', fontsize=12)
    ax3.set_ylabel('Ratio of +1 Opinions', fontsize=12)
    ax3.set_title('Opinion Evolution', fontsize=14)
    ax3.grid(visible=True, linestyle='--', alpha=0.7)
    ax3.legend(loc='upper left', fontsize=10, frameon=True, title="Initial Probability")
    
    plt.tight_layout()
    return fig

# fig = run_majority_analysis()
# plt.show()


# Example usage
if __name__ == "__main__":
    n = 10000
    p = 3 * np.log(n) / n
    G = nx.fast_gnp_random_graph(n, p)
    model = MarkovianMajorityModel(G)
    model.initialize_random_opinions(p_positive=0.5)
    results = model.simulate(max_time=100.0)
    has_consensus, consensus_value = model.get_consensus_status()
    final_pos, final_neg = model.get_opinion_counts()