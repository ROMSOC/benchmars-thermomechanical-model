from __future__ import print_function
from dolfin import *
import matplotlib.pyplot as plt
import numpy as np

# Read the mesh file from specified path.
mesh = Mesh("../../benchmarks/data_files/mesh_data/hearth.xml")
domains = MeshFunction('size_t',mesh,mesh.topology().dim()) # Read domain marker
subdomains = MeshFunction("size_t", mesh, "../../benchmarks/data_files/mesh_data/hearth_physical_region.xml") # Read subdomain markers
boundaries = MeshFunction("size_t", mesh, "../../benchmarks/data_files/mesh_data/hearth_facet_region.xml") # Read boundary markers
dx = Measure('dx', domain = mesh, subdomain_data = domains) # Volume measure
ds = Measure('ds', domain = mesh, subdomain_data = boundaries) # Boundary measure
n = as_vector(FacetNormal(mesh)) # Edge unit normal vector

d_bottom = ds(2) + ds(3) + ds(4) + ds(5) + ds(6) # Markers of bottom boundary \gamma_{-}
d_out = ds(7) + ds(8) + ds(9) + ds(10) + ds(11) # Markers of outer boundary \gamma_{out}
d_sf = ds(13) + ds(14) + ds(15) + ds(16) + ds(17) + ds(18) + ds(19) + ds(20) # Markers of inner boundary \gamma_{sf}

# Function space for temperature
VT = FunctionSpace(mesh,"CG",1) # Function space for temperature
x = list()
x.append(Expression("x[0]", element=VT.ufl_element())) # Read mesh coordinates where x[0] = r and x[1] = y
psi, T_ = TestFunction(VT), TrialFunction(VT)
T = Function(VT, name = "temperature")

# Thermal material properties and boundary data
k = 10 # Thermal conductivity
h_fluid = 200 # Convection coefficient h_{c,f}
h_right = 2000 # Convection coefficient h_{c,out}
h_bottom = 2000 # Convection coefficient h_{c,-}
T_right = 313 # Environmental temperature T_{out}
T_fluid = 1773 # Enviromental temperature T_{sf}
T_bottom = 313 # Temperature on Bottom boundary

# solving weak form of energy equation
a_T = k * inner(grad(psi),grad(T_)) * x[0] * dx + h_fluid * psi * T_ * x[0] * d_sf + h_right * psi * T_ * x[0] * d_out + h_bottom * psi * T_ * x[0] * d_bottom # bilinear form
l_T = h_fluid * psi * T_fluid * x[0] * d_sf + h_right * psi * T_right * x[0] * d_out + h_bottom * psi * T_bottom * x[0] * d_bottom # linear form
solve(a_T == l_T, T) # solve the equation

# # Function space for displacement
VM = VectorFunctionSpace(mesh,"CG",1) # Function space for displacement
x = Expression(("x[0]","x[1]"), element=VM.ufl_element()) # Read mesh coordinates where x[0] = r and x[1] = y
phi, u_ = TestFunction(VM), TrialFunction(VM)
u = Function(VM, name = "Displacement")

# Mechanical material parameters and imposition of Dirichlet boundary value
T_0 = 298 # Reference temperature
E = Constant(5e9) # Young's modulus
nu = Constant(0.2) # Poission's ratio
mu = E/2/(1+nu) # Lam\' parameter
lmbda = E*nu/(1+nu)/(1-2*nu) # Lam\' parameter
alpha = Constant(1e-6) # Thermal expansion coefficient
W = 0 # weight at top boundary
rho = 7460 # Molten metal density
g = 10. # Gravitation accelaration
p = rho * g * (7.265-x[1]) # Fluid pressure

# Axisymmetric strain tensor definition. Alternative could be to express stress as vector using Voigt notation.
def eps(u):
	return \
		sym(as_tensor([[u[0].dx(0), u[0].dx(1), 0. ],\
		[u[1].dx(0), u[1].dx(1), 0.],\
		[0., 0., u[0]/x[0]]]))

# Axisymmetric stress tensor definition. Alternative could be to express stress as vector using Voigt notation.
def sigma(u):
	return lmbda * tr(eps(u)) * Identity(3) + 2.0 * mu * eps(u)

# Dirichlet boundary data (Displacement).
bcs_M = [DirichletBC( VM.sub(0), Constant(0.), 'x[0] < DOLFIN_EPS and on_boundary'), DirichletBC( VM.sub(1), Constant(0.), 'near(x[1],0) and on_boundary')] 

#solving weak form of momentum equation
a_M = inner(sigma(u_),eps(phi)) * x[0] * dx # bilinear form
l_M = (2 * mu + 3 * lmbda) * alpha * inner((T - T_0) * Identity(3), eps(phi)) * x[0] * dx - dot( phi, W * n) * x[0] * ds(12) - dot( phi, p * n) * x[0] * d_sf # linear form
solve(a_M == l_M, u, bcs_M) # solve variation form

# Plotting and visualization
File("../../benchmarks/result_files/actual_problem/Temperature_computed.pvd") << T
File("../../benchmarks/result_files/actual_problem/Displacement.pvd") << u
