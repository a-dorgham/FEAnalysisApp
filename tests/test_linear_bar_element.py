import sys
import unittest
import numpy as np
from linear_bar_element import LinearBarElement  

class TestLinearBarElement(unittest.TestCase):

    def setUp(self):
        """Set up for test methods."""
        self.tolerance = 1e-2
        np.set_printoptions(precision=2, suppress=True)

    def test_simple_2D_truss_example_3_3(self):
        """Test case for a simple 2D truss (Ex. 3.3 p88)."""
        imported_data = {
            'structure_info': {'dofs_per_node': 2, 'dimension': 2, 'element_type': 'Truss'},
            'nodes': {
                1: {'X': 0.0, 'Y': 0.0},
                2: {'X': 1.2 * np.cos(np.deg2rad(30)), 'Y': 1.2 * np.sin(np.deg2rad(30))},
            },
            'elements': {
                1: {'node1': 1, 'node2': 2, 'E': 200E9, 'A': 6E-4},
            },
        }
        solver = LinearBarElement(imported_data)
        K_global_actual = solver.assemble_global_stiffness()
        K_global_expected = 1E8 * np.array([
            [ 0.75,  0.43, -0.75, -0.43],
            [ 0.43,  0.25, -0.43, -0.25],
            [-0.75, -0.43,  0.75,  0.43],
            [-0.43, -0.25,  0.43,  0.25]
        ])
        self.assertTrue(np.allclose(K_global_actual / 1E8, K_global_expected / 1E8, atol=self.tolerance))
        #print(K_global_actual / 1E8, K_global_expected / 1E8)


    def test_simple_2D_truss_example_3_5(self):
        """Test case for a simple 2D truss (Ex. 3.5 p93)."""
        imported_data = {
            'structure_info': {'dofs_per_node': 2, 'dimension': 2, 'element_type': 'Truss'},
            'nodes': {
                1: {'X': 0.0, 'Y': 0.0},
                2: {'X': 0.0, 'Y': 3.0},
                3: {'X': 3.0, 'Y': 3.0},
                4: {'X': 3.0, 'Y': 0.0},
            },
            'elements': {
                1: {'node1': 1, 'node2': 2, 'E': 200E9, 'A': 6E-4},
                2: {'node1': 1, 'node2': 3, 'E': 200E9, 'A': 6E-4},
                3: {'node1': 1, 'node2': 4, 'E': 200E9, 'A': 6E-4},
            },
        }
        solver = LinearBarElement(imported_data)
        K_global_actual = solver.assemble_global_stiffness()
        K_global_expected = 4E7 * np.array([
            [ 1.35,  0.35,  0.,    0.,   -0.35, -0.35, -1.,    0.  ],
            [ 0.35,  1.35,  0.,   -1.,   -0.35, -0.35,  0.,    0.  ],
            [ 0.,    0.  ,  0.,    0.,    0.  ,  0.  ,  0.,    0.  ],
            [ 0.,   -1.  ,  0.,    1.,    0.  ,  0.  ,  0.,    0.  ],
            [-0.35, -0.35,  0.,    0.,    0.35,  0.35,  0.,    0.  ],
            [-0.35, -0.35,  0.,    0.,    0.35,  0.35,  0.,    0.  ],
            [-1.,    0.  ,  0.,    0.,    0.  ,  0.  ,  1.,    0.  ],
            [ 0.,    0.  ,  0.,    0.,    0.  ,  0.  ,  0.,    0.  ]
        ])
        self.assertTrue(np.allclose(K_global_actual / 4E7, K_global_expected / 4E7, atol=self.tolerance))
        #print(K_global_actual / 4E7,'\n\n', K_global_expected / 4E7)


    def test_simple_3D_truss(self):
        """Test case for a simple 3D truss."""
        nodes_coords = [
            (1800.0e-3, 0.0, 0.0),
            (0.0, 900.0e-3, 0.0),
            (0.0, 900.0e-3, 1800.0e-3),
            (0.0, 0.0, -1200.0e-3),
        ]
        elements_data = [
            (1, 2, 8E9, 200.0e-6),
            (1, 3, 8E9, 500.0e-6),
            (1, 4, 8E9, 125.0e-6),
        ]
        imported_data = {
            'structure_info': {'dofs_per_node': 3, 'dimension': 3, 'element_type': 'Truss'},
            'nodes': {i + 1: {'X': coord[0], 'Y': coord[1], 'Z': coord[2]} for i, coord in enumerate(nodes_coords)},
            'elements': {i + 1: {'node1': data[0], 'node2': data[1], 'E': data[2], 'A': data[3]} for i, data in enumerate(elements_data)},
        }
        solver = LinearBarElement(imported_data)
        K_global_actual = solver.assemble_global_stiffness()
        K_global_expected = np.array([
            [1614492.67, -647236.66, -445089.98, -636037.11,  318018.56,       0.        , -658436.21,  329218.11,  658436.21, -320019.34,       0.        , -213346.23],
            [-647236.66,  323618.33,  329218.11,  318018.56, -159009.28,       0.        , 329218.11, -164609.05, -329218.11,       0.        ,       0.        ,       0.        ],
            [-445089.98,  329218.11,  800667.03,       0.        ,       0.        ,       0.        ,658436.21, -329218.11, -658436.21, -213346.23,       0.        , -142230.82],
            [-636037.11,  318018.56,       0.  ,  636037.11, -318018.56,       0.        , 0.        ,       0.        ,       0.        ,       0.        ,       0.        ,       0.        ],
            [ 318018.56, -159009.28,       0.    , -318018.56,  159009.28,       0.        ,  0.        ,       0.        ,       0.        ,       0.        ,       0.        ,       0.        ],
            [   0.  ,       0.    ,       0.    ,       0.        ,       0.        ,       0.        ,   0.        ,       0.        ,       0.        ,       0.        ,       0.        ,       0.        ],
            [-658436.21,  329218.11,  658436.21,       0.        ,       0.        ,       0.        ,658436.21, -329218.11, -658436.21,       0.        ,       0.        ,       0.        ],
            [ 329218.11, -164609.05, -329218.11,       0.        ,       0.        ,       0.        ,-329218.11,  164609.05,  329218.11,       0.        ,       0.        ,       0.        ],
            [ 658436.21, -329218.11, -658436.21,       0.        ,       0.        ,       0.        ,-658436.21,  329218.11,  658436.21,       0.        ,       0.        ,       0.        ],
            [-320019.34,    0.     , -213346.23,       0.        ,       0.        ,       0.        ,  0.        ,       0.        ,       0.        ,  320019.34,       0.        ,  213346.23],
            [       0.        ,       0.        ,       0.        ,       0.        ,       0.        ,       0.        , 0.        ,       0.        ,       0.        ,       0.        ,       0.        ,       0.        ],
            [-213346.23,       0.        , -142230.82,       0.        ,       0.        ,       0.        ,    0.        ,       0.        ,       0.        ,  213346.23,       0.        ,  142230.82]
        ])
        self.assertTrue(np.allclose(K_global_actual, K_global_expected, atol=self.tolerance)) 



if __name__ == '__main__':
    sys.stdout = sys.__stdout__ 
    unittest.main()