
import time
import numpy as np
from Network_Traversal.core.engine import SimulationEngine

def benchmark():
    print("Initializing Engine...")
    start_init = time.time()
    engine = SimulationEngine()
    engine.build_network()
    print(f"Initialization took: {time.time() - start_init:.4f}s")
    
    print("\nStarting Benchmark (10 runs)...")
    times = []
    
    for i in range(10):
        start = time.time()
        res = engine.run_simulation()
        duration = time.time() - start
        times.append(duration)
        print(f"Run {i+1}: {duration:.4f}s (Success: {res.success})")
        
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    print(f"\nAverage Simulation Cost: {avg_time:.4f}s +/- {std_time:.4f}s")
    print(f"Simulations per minute: {60/avg_time:.2f}")

if __name__ == "__main__":
    benchmark()
