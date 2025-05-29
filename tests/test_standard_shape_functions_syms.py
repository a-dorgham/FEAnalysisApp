import sympy as sp
import numpy as np
from typing import Tuple

def get_standard_hermite_shape_functions(xi: sp.Symbol, L: sp.Symbol) -> Tuple[sp.Matrix, sp.Matrix]:
    """
    Returns the standard cubic Hermite shape functions for beam bending
    and their first derivatives with respect to the natural coordinate xi.

    Args:
        xi (sp.Symbol): The natural coordinate ranging from -1 to 1.
        L (sp.Symbol): The length of the beam element.

    Returns:
        Tuple[sp.Matrix, sp.Matrix]: A tuple containing two symbolic matrices:
            - N_bending (sp.Matrix): The cubic Hermite shape functions [N1, N2*L/2, N3, N4*L/2]^T
                                      associated with [w1, theta_z1, w2, theta_z2]^T for bending
                                      about the y-axis (and similarly for bending about z-axis).
                                      The rotational shape functions are scaled by L/2.
            - dN_bending_dxi (sp.Matrix): The first derivatives of the shape functions
                                           with respect to xi.
    """
    N1 = 1 - 3*xi**2 + 2*xi**3
    N2 = xi - 2*xi**2 + xi**3
    N3 = 3*xi**2 - 2*xi**3
    N4 = -xi**2 + xi**3

    N_bending = sp.Matrix([N1, N2*L/2, N3, N4*L/2])

    # First derivatives with respect to xi
    dN1_dxi = -6*xi + 6*xi**2
    dN2_dxi = 1 - 4*xi + 3*xi**2
    dN3_dxi = 6*xi - 6*xi**2
    dN4_dxi = -2*xi + 3*xi**2
        
    dN_bending_dxi = sp.Matrix([dN1_dxi,
                                dN2_dxi * L/2,
                                dN3_dxi,
                                dN4_dxi * L/2])

    # Second derivatives with respect to xi
    d2N1_dxi2 = -6 + 12*xi
    d2N2_dxi2 = -4 + 6*xi
    d2N3_dxi2 = 6 - 12*xi
    d2N4_dxi2 = -2 + 6*xi

    d2N_bending_dxi2 = sp.Matrix([d2N1_dxi2,
                                d2N2_dxi2 * L/2,
                                d2N3_dxi2,
                                d2N4_dxi2 * L/2])

    return N_bending, dN_bending_dxi, d2N_bending_dxi2

def get_standard_B_matrix(xi: sp.Symbol, L: sp.Symbol) -> sp.Matrix:
    """
    Returns the standard B matrix for a 3D frame element based on
    linear interpolation for axial and torsion, and cubic Hermite
    interpolation for bending.

    Args:
        xi (sp.Symbol): The natural coordinate ranging from -1 to 1.
        L (sp.Symbol): The length of the beam element.

    Returns:
        sp.Matrix: The 6x12 B matrix. The rows correspond to [epsilon_x, gamma_x, kappa_y, kappa_z, gamma_y, gamma_z]^T
                   and the columns correspond to the 12 nodal DOFs
                   [u1, v1, w1, theta_x1, theta_y1, theta_z1, u2, v2, w2, theta_x2, theta_y2, theta_z2]^T.
    """
    J_inv = 2/L  # Jacobian inverse

    # Shape functions for axial displacement u
    N_axial = sp.Matrix([(1 - xi)/2, (1 + xi)/2])
    dN_axial_dxi = sp.diff(N_axial, xi)
    dB_axial_dx = dN_axial_dxi * J_inv

    # Shape functions for torsional rotation theta_x
    N_torsion = sp.Matrix([(1 - xi)/2, (1 + xi)/2])
    dN_torsion_dxi = sp.diff(N_torsion, xi)
    dB_torsion_dx = dN_torsion_dxi * J_inv

    # Shape functions for transverse displacements v (bending about z) and w (bending about y)
    N_bending, dN_bending_dxi, d2N_bending_dxi2 = get_standard_hermite_shape_functions(xi, L)
    d2N_bending_dx2 = d2N_bending_dxi2 * (J_inv**2)

    B = sp.zeros(6, 12)

    # Axial strain (du/dx)
    B[0, 0] = dB_axial_dx[0]
    B[0, 6] = dB_axial_dx[1]

    # Torsion (d(theta_x)/dx)
    B[1, 3] = dB_torsion_dx[0]
    B[1, 9] = dB_torsion_dx[1]

    # Bending about y-axis (curvature -d^2w/dx^2, DOFs: w1, theta_y1, w2, theta_y2)
    B[2, 2] = -d2N_bending_dx2[0]
    B[2, 4] = -d2N_bending_dx2[1]
    B[2, 8] = -d2N_bending_dx2[2]
    B[2, 10] = -d2N_bending_dx2[3]

    # Bending about z-axis (curvature d^2v/dx^2, DOFs: v1, theta_z1, v2, theta_z2)
    B[3, 1] = d2N_bending_dx2[0]
    B[3, 5] = d2N_bending_dx2[1]
    B[3, 7] = d2N_bending_dx2[2]
    B[3, 11] = d2N_bending_dx2[3]

    # Shear strains (simplified, neglecting contributions from bending derivatives for now)
    # These are more complex and depend on the specific beam theory (e.g., Timoshenko).
    # For Euler-Bernoulli, these are often not directly derived from the cubic shape functions in a simple B matrix.
    # A more rigorous Timoshenko beam element would include shear deformation shape functions.
    # For a basic Euler-Bernoulli frame element, these rows might be zero or based on other assumptions.
    # B[4, 1] = J_inv * sp.diff(N_bending[0], xi)  # approx dv/dx related to v1
    # B[4, 5] = 1                                # - theta_z1
    # B[4, 7] = J_inv * sp.diff(N_bending[2], xi)  # approx dv/dx related to v2
    # B[4, 11] = 1                               # - theta_z2

    # B[5, 2] = J_inv * sp.diff(N_bending[0], xi)  # approx dw/dx related to w1
    # B[5, 4] = -1                               # + theta_y1
    # B[5, 8] = J_inv * sp.diff(N_bending[2], xi)  # approx dw/dx related to w2
    # B[5, 10] = -1                              # + theta_y2

    return B

if __name__ == '__main__':
    xi_sym = sp.Symbol('xi')
    L_sym = sp.Symbol('L')

    # Get standard shape functions and their first derivatives for bending
    N_b, dN_b_dxi,_ = get_standard_hermite_shape_functions(xi_sym, L_sym)
    print("Standard Cubic Hermite Shape Functions (for w and theta_z):\n", N_b)
    print("\nFirst Derivatives of Shape Functions (dN/dxi):\n", dN_b_dxi)

    # Get the standard B matrix
    B_standard = get_standard_B_matrix(xi_sym, L_sym)
    print("\nStandard B Matrix for 3D Frame Element:\n", B_standard)

    # Example of how to use the B matrix to form the stiffness matrix (conceptual)
    E, G, A, J, Iy, Iz = sp.symbols('E G A J Iy Iz')
    C_standard = sp.diag(E*A, G*J, E*Iy, E*Iz, G*A, G*A) # Using small factor for shear area for conceptual integration

    K_e_standard = sp.zeros(12, 12)
    integrand_standard = B_standard.T * C_standard * B_standard * (L_sym/2)

    print("\nConceptual Integrand for Stiffness Matrix (Standard B):\n", sp.pprint(integrand_standard))

    i = -1
    j = 1
    
    K_e_full = sp.integrate(integrand_standard, (xi_sym, i,j))
    print("\nConceptual Integrand for Stiffness Matrix (Standard B):\n", sp.pprint(K_e_full))

    # Extract the term corresponding to (v1, v1) which is K_e[1, 1]
    K_e_1_1 = sp.integrate(integrand_standard[1, 1], (xi_sym, i,j))
    print("\nStiffness matrix term K_e[1, 1] (v1, v1):\n", sp.simplify(K_e_1_1))

    # Extract the term corresponding to (v1, theta_z1) which is K_e[1, 5]
    K_e_1_5 = sp.integrate(integrand_standard[1, 5], (xi_sym, i,j))
    print("\nStiffness matrix term K_e[1, 5] (v1, theta_z1):\n", sp.simplify(K_e_1_5))

    # Extract the term corresponding to (v1, v2) which is K_e[1, 7]
    K_e_1_7 = sp.integrate(integrand_standard[1, 7], (xi_sym, i,j))
    print("\nStiffness matrix term K_e[1, 7] (v1, v2):\n", sp.simplify(K_e_1_7))

    # Extract the term corresponding to (v1, theta_z2) which is K_e[1, 11]
    K_e_1_11 = sp.integrate(integrand_standard[1, 11], (xi_sym, i,j))
    print("\nStiffness matrix term K_e[1, 11] (v1, theta_z2):\n", sp.simplify(K_e_1_11))