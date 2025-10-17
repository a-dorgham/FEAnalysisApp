import unittest
import sys
import numpy as np
from src.core.elements.beam import LinearBeamElement

class TestLinearBeamElement(unittest.TestCase):

    def setUp(self):
        """Set up for test methods."""
        self.tolerance = 1e-2
        np.set_printoptions(precision=2, suppress=True)
        
    def test_simple_horizontal_beam(self):
        """Test case for a simple horizontal beam."""
        imported_data = {
            'structure_info': {'dofs_per_node': 3, 'dimension': 2, 'element_type': 'Beam'},
            'nodes': {
                1: {'X': 0.0, 'Y': 0.0},
                2: {'X': 1.0, 'Y': 0.0},
            },
            'elements': {
                1: {'node1': 1, 'node2': 2, 'E': 1.0, 'A': 1.0, 'Iy': 1.0},
            },
        }
        solver = LinearBeamElement(imported_data)
        K_global_expected = np.array([
            [ 1.0,  0.0,  0.0, -1.0,  0.0,  0.0],
            [ 0.0, 12.0,  6.0,  0.0,-12.0,  6.0],
            [ 0.0,  6.0,  4.0,  0.0, -6.0,  2.0],
            [-1.0,  0.0,  0.0,  1.0,  0.0,  0.0],
            [ 0.0,-12.0, -6.0,  0.0, 12.0, -6.0],
            [ 0.0,  6.0,  2.0,  0.0, -6.0,  4.0]
        ])
        K_global_actual = solver.assemble_global_stiffness()
        np.testing.assert_allclose(K_global_actual, K_global_expected, atol=self.tolerance)

    def test_2d_frame_example_5_1(self):
        """Test case for a 2D frame (Ex 5.1, p243)."""
        imported_data = {
            'structure_info': {'dofs_per_node': 3, 'dimension': 2, 'element_type': 'Beam'},
            'nodes': {
                1: {'X': 0.0, 'Y': 0.0},
                2: {'X': 0.0, 'Y': 3.0},
                3: {'X': 3.0, 'Y': 3.0},
                4: {'X': 3.0, 'Y': 0.0},
            },
            'elements': {
                1: {'node1': 1, 'node2': 2, 'E': 200.0E9, 'A': 6.5E-3, 'Iy': 80E-6},
                2: {'node1': 2, 'node2': 3, 'E': 200.0E9, 'A': 6.5E-3, 'Iy': 40E-6},
                3: {'node1': 3, 'node2': 4, 'E': 200.0E9, 'A': 6.5E-3, 'Iy': 80E-6},
            },
        }
        solver = LinearBeamElement(imported_data)
        K_global_actual = solver.assemble_global_stiffness()
        # Expected global stiffness matrix 

        k1 = np.array([
            [7.111e+06, 0.000e+00, -1.067e+07, -7.111e+06, 0.000e+00, -1.067e+07],
            [0.000e+00, 4.333e+08, 0.000e+00, 0.000e+00, -4.333e+08, 0.000e+00],
            [-1.067e+07, 0.000e+00, 2.133e+07, 1.067e+07, 0.000e+00, 1.067e+07],
            [-7.111e+06, 0.000e+00, 1.067e+07, 7.111e+06, 0.000e+00, 1.067e+07],
            [0.000e+00, -4.333e+08, 0.000e+00, 0.000e+00, 4.333e+08, 0.000e+00],
            [-1.067e+07, 0.000e+00, 1.067e+07, 1.067e+07, 0.000e+00, 2.133e+07]
        ])

        k2 = np.array([
            [4.333e+08, 0.000e+00, 0.000e+00, -4.333e+08, 0.000e+00, 0.000e+00],
            [0.000e+00, 3.556e+06, 5.333e+06, 0.000e+00, -3.556e+06, 5.333e+06],
            [0.000e+00, 5.333e+06, 1.067e+07, 0.000e+00, -5.333e+06, 5.333e+06],
            [-4.333e+08, 0.000e+00, 0.000e+00, 4.333e+08, 0.000e+00, 0.000e+00],
            [0.000e+00, -3.556e+06, -5.333e+06, 0.000e+00, 3.556e+06, -5.333e+06],
            [0.000e+00, 5.333e+06, 5.333e+06, 0.000e+00, -5.333e+06, 1.067e+07]
        ])

        k3 = np.array([
            [7.111e+06, 0.000e+00, 1.067e+07, -7.111e+06, 0.000e+00, 1.067e+07],
            [0.000e+00, 4.333e+08, 0.000e+00, 0.000e+00, -4.333e+08, 0.000e+00],
            [1.067e+07, 0.000e+00, 2.133e+07, -1.067e+07, 0.000e+00, 1.067e+07],
            [-7.111e+06, 0.000e+00, -1.067e+07, 7.111e+06, 0.000e+00, -1.067e+07],
            [0.000e+00, -4.333e+08, 0.000e+00, 0.000e+00, 4.333e+08, 0.000e+00],
            [1.067e+07, 0.000e+00, 1.067e+07, -1.067e+07, 0.000e+00, 2.133e+07]
        ])

        K_global_expected = np.zeros((12, 12)) # 4 nodes * 3 DOF/node
        dof_map1 = [0, 1, 2, 3, 4, 5]
        dof_map2 = [3, 4, 5, 6, 7, 8]
        dof_map3 = [6, 7, 8, 9, 10, 11]

        for i in range(6):
            for j in range(6):
                K_global_expected[dof_map1[i], dof_map1[j]] += k1[i, j]
                K_global_expected[dof_map2[i], dof_map2[j]] += k2[i, j]
                K_global_expected[dof_map3[i], dof_map3[j]] += k3[i, j]
        
        print('\n\n\n=========Actual:\n',K_global_actual/66.67e7,'\n\n\n=========Expected:\n', K_global_expected/66.67e7)
        np.testing.assert_allclose(K_global_actual, K_global_expected,  rtol=1e-2)

if __name__ == '__main__':
    sys.stdout = sys.__stdout__ 
    unittest.main()