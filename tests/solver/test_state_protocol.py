import numpy as np

from aerislab.core.protocols import StateProvider, AuxDynamics, InertialProvider

class DummyClass:

    def num_states(self):
        return 2
    
    def pack_state(self, out):
        out[:] = [1.0, 2.0]
        return out

    def unpack_state(self, y):
        self.y_helper = np.asarray(y).copy()

        return self.y_helper
    
    def compute_derivatives(self, t):
        return np.array([0.1, 0.2])
    

def test_state_provider_protocol():
    dummy = DummyClass()
    assert isinstance(dummy, StateProvider)
    assert isinstance(dummy, AuxDynamics)
    assert not isinstance(dummy, InertialProvider)





