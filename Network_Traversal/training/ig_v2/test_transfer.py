
import numpy as np
from Network_Traversal.training.ig_v2.surrogate import SurrogateModel

def test_length_scale_transfer():
    print("Testing Length Scale Transfer...")
    
    # 1. Setup Initial State
    n_features = 2
    bounds = (0.1, 200.0)
    # Simulate a learned state where features have different importance
    # 5.0 -> High Importance, 20.0 -> Low Importance
    initial_ls = np.array([5.0, 20.0])
    
    surrogate = SurrogateModel(n_features, bounds, initial_length_scale=initial_ls)
    
    current_ls = surrogate.get_length_scales()
    print(f"Initial Length Scales: {current_ls}")
    assert np.allclose(current_ls, initial_ls), "Constructor did not use initial_length_scale"
    
    # 2. Simulate SPLIT of Index 0 (Value 5.0)
    idx = 0
    val = current_ls[idx]
    
    # Logic from agent.py
    ls_prime = np.delete(current_ls, idx)
    ls_new = np.concatenate([ls_prime, [val, val]])
    
    print(f"Simulated New Length Scales (Expected): {ls_new}")
    
    # 3. Create New Surrogate with Transferred LS
    new_surrogate = SurrogateModel(n_features=3, bounds=bounds, initial_length_scale=ls_new)
    new_actual_ls = new_surrogate.get_length_scales()
    
    print(f"Actual New Length Scales: {new_actual_ls}")
    
    assert np.allclose(new_actual_ls, ls_new), "New surrogate did not accept transferred length scales"
    
    # 4. Check Default Behavior (Reset)
    reset_surrogate = SurrogateModel(n_features=3, bounds=bounds)
    reset_ls = reset_surrogate.get_length_scales()
    print(f"Reset (Default) Length Scales: {reset_ls}")
    assert np.allclose(reset_ls, 10.0), "Default surrogate should be 10.0"
    
    print("\nSUCCESS: Length Scale Transfer Logic Verified!")

if __name__ == "__main__":
    test_length_scale_transfer()
