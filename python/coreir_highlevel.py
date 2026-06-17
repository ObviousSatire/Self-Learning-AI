"""
CoreIR High-Level Python API
ML-framework-style interface for self-learning programs.
"""
import numpy as np
from coreir_bridge import CoreIR

class SelfLearningModel:
    def __init__(self):
        self.runtime = CoreIR()
        self.parameters = {}
        self.performance_history = []
    
    def register_parameter(self, name, data):
        self.parameters[name] = np.asarray(data, dtype=np.float64)
    
    def forward(self, x):
        result = np.zeros_like(x, dtype=np.float64)
        for param in self.parameters.values():
            result += param * x
        return result
    
    def compute_loss(self, predicted, target):
        return np.mean((predicted - target) ** 2)
    
    def search_improvements(self, num_candidates=5, temperature=0.1):
        space = self.runtime.create_search(temperature)
        best_score = -float('inf')
        best_params = None
        for i in range(num_candidates):
            candidate = {}
            for name, param in self.parameters.items():
                noise = np.random.normal(0, 0.1, param.shape)
                candidate[name] = param + noise
            score = 1.0 / (1.0 + abs(np.sum([np.sum(v) for v in candidate.values()])))
            self.runtime.search_add(id(candidate), score)
            if score > best_score:
                best_score = score
                best_params = candidate
        return best_params
    
    def apply_improvement(self, new_params):
        if new_params:
            for name in self.parameters:
                if name in new_params:
                    self.parameters[name] = new_params[name]
    
    def learn(self, x, y, steps=10):
        print(f"Starting self-learning for {steps} steps...")
        for step in range(steps):
            predicted = self.forward(x)
            loss = self.compute_loss(predicted, y)
            if step % 3 == 0:
                improved = self.search_improvements()
                self.apply_improvement(improved)
            self.runtime.learn_step()
            self.performance_history.append(float(loss))
            if step % 5 == 0:
                print(f"  Step {step}: loss = {loss:.4f}")
        final_perf = self.runtime.get_performance()
        print(f"Learning complete. Runtime performance: {final_perf:.3f}")
        return self.performance_history
    
    def shutdown(self):
        self.runtime.shutdown()

if __name__ == "__main__":
    print("=== CoreIR High-Level API Demo ===")
    model = SelfLearningModel()
    model.register_parameter("w", np.array([2.0]))
    model.register_parameter("b", np.array([1.0]))
    np.random.seed(42)
    x = np.linspace(0, 10, 100)
    y = 3.0 * x + 2.0 + np.random.normal(0, 0.5, 100)
    history = model.learn(x, y, steps=20)
    print(f"Final loss: {history[-1]:.4f}")
    model.shutdown()
