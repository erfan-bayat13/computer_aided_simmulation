from complete_queue import Client, MultiServerQueueSimulator
from gaussian_rv import GaussianGenerator
import random

## for lab 1
## note that for calling preemptive priority queue, you need to set is_preemptive=True and queue_type="Priority"
## for calling non-preemptive priority queue, you need to set is_preemptive=False and queue_type="Priority"
## for calling non-priority queue, you need to set is_preemptive=False and queue_type="FIFO" or queue_type="LIFO"
'''
random.seed(42) # Setting the seed for reproducibility
# Parameters for simulation
num_servers = 3
queue_capacity = 5
arrival_rate = 0.8  # Average of 1 customer every ~1.25 time units
service_rate = 0.3 # Average of 1 service every ~1.67 time units
simulation_time = 1000  # Total time to run the simulation

# Running the simulation
simulator = MultiServerQueueSimulator(num_servers, queue_capacity, arrival_rate, service_rate, simulation_time, queue_type="Priority", is_preemptive=False)
simulator.run()
'''

## for lab 2 

### Example Usage box_muller
'''
gen = GaussianGenerator(method='box_muller', seed=42)
samples = [sample for _ in range(1000) for sample in gen.box_muller()]
'''
### Example Usage central limit theorem
'''
gen = GaussianGenerator(method='clt', seed=42)
samples = gen.generate(num_samples=1000)
'''
### Example Usage newton inverse transform
'''
gen = GaussianGenerator(method='newton', seed=42)
samples = gen.generate(num_samples=1000)
'''
### example of plotting samples and testing distribution and independence
'''
gen.plot_samples(samples)
gen.test_distribution(samples)
gen.chi_square_test(samples, bins=10)
gen.autocorrelation_test(samples, lags=40)
gen.runs_test(samples)
'''
