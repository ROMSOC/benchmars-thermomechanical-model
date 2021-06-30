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

# Define function space
VT = FunctionSpace(mesh,"CG",3) # Function space for temperature
psi, T_ = TestFunction(VT), TrialFunction(VT) # Evaluate trial and test function
T = Function(VT, name = "temperature increase")

# Known analytical solution, Thermal material properties and Boundary data
T_analytical = Expression('x[0]*x[0]*x[1]',degree = 3) 
VT_analytical = FunctionSpace(mesh,"CG",3) #Space for analytical solution
T_analytical = project(T_analytical,VT_analytical)
k = 10. # Thermal conductivity
h_fluid = 200. # Convection coefficient on \gamma_{sf}
h_right = 2000. # Convection coefficient on \gamma_{out}
h_bottom = 2000. # Convection coefficient on \gamma_{-}
x = list()
x.append(Expression("x[0]", element=VT.ufl_element())) #r coordinate
x.append(Expression("x[1]", element=VT.ufl_element())) #y coordinate

# solving weak form of energy equation
a_T = k * inner(grad(psi),grad(T_)) * x[0] * dx + \
	h_fluid * psi * T_ * x[0] * d_sf + h_right * psi * T_ * x[0] * d_out + \
	h_bottom * psi * T_ * x[0] * d_bottom # Bilinear side
l_T = h_fluid * psi * (x[0] * x[0] * x[1] + k/h_fluid * ( 2 * x[0] * x[1] * n[0] + x[0] * x[0] * n[1] ) ) * x[0] * d_sf + \
	h_right * psi * (x[0]*x[0]*x[1]+2*x[0]*x[1]*k/h_right) * x[0] * d_out + \
	h_bottom * psi * (x[0] * x[0] * x[1] - x[0] * x[0] * k / h_bottom) * x[0] * d_bottom + \
	-4 * k * x[1] * psi * x[0] * dx + psi * k * x[0] * x[0] * x[0] * ds(12) # Linear side
solve(a_T == l_T, T) # Solve the variational form

# Define \mathbb{U} norm
def compute_U_norm(phi,mesh):
	x = SpatialCoordinate(mesh)
	a = inner(phi,phi)*x[0]*dx + inner(grad(phi),grad(phi))*x[0]*dx + (phi[0]**2/x[0])*dx
	A = assemble(a)
	return sqrt(A)

# Axisymmetric strain tensor definition. Alternative could be to express strain as vector using Voigt notation.
def eps(u):
	return \
		sym(as_tensor([[u[0].dx(0), u[0].dx(1), 0. ],\
		[u[1].dx(0), u[1].dx(1), 0.],\
		[0., 0., u[0]/x[0]]]))

# Axisymmetric thermo-mechanical stress tensor definition. Alternative could be to express as vector using Voigt notation.
def sigma(u,T):
	return lmbda * tr(eps(u)) * Identity(3) + 2.0 * mu * eps(u) - (2 * mu + 3 * lmbda) * alpha * (T - T_0) * Identity(3)

# Axisymmetric mechanical stress tensor definition. Alternative could be to express as vector using Voigt notation.
def sigma2(u):
	return lmbda * tr(eps(u)) * Identity(3) + 2.0 * mu * eps(u)

error_u_vector = [] #List of absolute error
p = range(1,4) #Range of polynomials

# Known analytical solution, Mechanical material properties
u_analytical = Expression(("A * x[0] * x[1] * x[1]","A * x[0] * x[0] * x[1]"),A=1e-4,degree=3) 
# analytical solution initiated as expression and projected onto relevant funtion space
VM_analytical = VectorFunctionSpace(mesh,"CG",3)
u_analytical = project(u_analytical,VM_analytical)
T_0 = 298 # Reference temperature for zero thermal stress
E = Constant(5e9) # Young's modulus
nu = Constant(0.2) # Poisson's ratio
mu = E/2/(1+nu) # Lame\'e parameter
lmbda = E*nu/(1+nu)/(1-2*nu) # Lame\'e parameter
alpha = Constant(1e-6) # Thermal expansion coefficient

for i in p:
	# Define function space for displacement
	VM = VectorFunctionSpace(mesh,"CG",i) # Function space for displacement
	x = Expression(("x[0]","x[1]"), element=VM.ufl_element())
	phi, u_ = TestFunction(VM), TrialFunction(VM)
	u = Function(VM, name = "Displacement") # u[0] = u_r and u[1] = u_y
	VS = FunctionSpace(mesh,"CG",max(i-1,1)) # Function space for shear component of stress

	# Dirichlet boundary data
	bcs_M = [DirichletBC( VM.sub(0), Constant(0.), 'x[0] < DOLFIN_EPS and on_boundary'), DirichletBC( VM.sub(1), Constant(0.), 'near(x[1],0) and on_boundary')] 

	#Boundary and source terms
	f0_r = - (2*E*nu*1e-4*x[0]/(1-2*nu)/(1+nu)+2*E*1e-4*x[0]/(1+nu)-2*E*x[0]*x[1]*alpha/(1-2*nu))
	f0_y = - (4*E*1e-4*x[1]/(1+nu)+4*E*1e-4*x[1]*nu/(1-2*nu)/(1+nu)-E*x[0]*x[0]*alpha/(1-2*nu))
	
	g_plus_r = 2*E*1e-4*x[0]*x[1]/(1+nu)
	g_plus_y = E / (1-2*nu) / (1+nu) * (2*nu*1e-4*x[1]*x[1]+(1-nu)*1e-4*x[0]*x[0]) - E*alpha/(1-2*nu)*(x[0]*x[0]*x[1] - T_0)
	
	g_minus_r = -g_plus_r
	
	g_sf_r = (E / (1-2*nu) / (1+nu) * (1e-4 * x[1] * x[1] + nu * 1e-4 * x[0] * x[0]) - E*alpha/(1-2*nu)*(x[0]*x[0]*x[1] - T_0)) * n[0] + 2 * E * 1e-4 * x[0] * x[1] / (1 + nu) * n[1]
	g_sf_y = 2 * E * 1e-4 * x[0] * x[1] / (1 + nu) * n[0] + (E / (1-2*nu) / (1+nu) * (2 * nu * 1e-4 * x[1] * x[1] + (1 - nu) * 1e-4 * x[0] * x[0])- E*alpha/(1-2*nu)*(x[0]*x[0]*x[1] - T_0)) * n[1]
	
	g_out_r = E / (1-2*nu) / (1+nu) * (1e-4 * x[1] * x[1] + nu * 1e-4 * x[0] * x[0]) - E*alpha/(1-2*nu)*(x[0]*x[0]*x[1] - T_0)
	g_out_y = 2*E*1e-4*x[0]*x[1]/(1+nu)
	
	# solving weak form of momentum equation
	# This is not bilinear side as terms related to thermal stress are included.
	a_M1 = inner(sigma(u_,T),eps(phi)) * x[0] * dx
	# This is not linear side as terms related to thermal stress are not included.
	l_M1 = (phi[0] * f0_r + phi[1] * f0_y) * x[0] * dx + (phi[0] * g_plus_r + phi[1] * g_plus_y) * x[0] * ds(12) + \
	(phi[0] * g_minus_r) * x[0] * d_bottom + (phi[0] * g_sf_r + phi[1] * g_sf_y) * x[0] * d_sf + \
	(phi[0] * g_out_r + phi[1] * g_out_y) * x[0] * d_out
	F = a_M1 - l_M1
	a_M = lhs(F) # Now a_M is bilinear form
	l_M = rhs(F) # Now l_M is linear form
	solve(a_M == l_M, u, bcs_M) # Solve equation
	
	# Compute \mathbb{U} norm of error
	error_u = compute_U_norm(u_analytical-u,mesh)/compute_U_norm(u_analytical,mesh)
	error_u_vector.append(error_u)
	print("Relative error in U-norm : ",str(error_u))

	# Von Mises stress for computed displacement
	sigma_dev = sigma(u,T) - tr(sigma(u,T)) / 3 * Identity(3) 
	sigma_vm = sqrt(3 * inner( sigma_dev, sigma_dev) / 2) # Von mises stress
	# Von Mises stress for analytical displacement
	sigma_dev_analytical = sigma(u_analytical,T) - tr(sigma(u_analytical,T)) / 3 * Identity(3)
	sigma_vm_analytical = sqrt(3 * inner( sigma_dev_analytical, sigma_dev_analytical) / 2) # Von mises stress
	# Spherical stress for computed displacement
	sigma_spherical = tr(sigma(u,T)) / 3
	# Spherical stress for analytical displacement
	sigma_spherical_analytical = tr(sigma(u_analytical,T)) / 3
	# Spherical mechanical stress for computed displacement
	sigma_spherical_non_thermal = tr(sigma2(u)) / 3

# Post-processing and visualization
File("../../benchmarks/result_files/coupled_model/Temperature_computed.pvd") << T
File("../../benchmarks/result_files/coupled_model/Temperature_analytical.pvd") << T_analytical
File("../../benchmarks/result_files/coupled_model/Teperature_error.pvd") << project(T-T_analytical,VT)

File("../../benchmarks/result_files/coupled_model/displacement_computed.pvd") << u
File("../../benchmarks/result_files/coupled_model/displacement_analytical.pvd") << u_analytical
File("../../benchmarks/result_files/coupled_model/displacement_absolute_error.pvd") << project(u_analytical - u,VM)

File("../../benchmarks/result_files/coupled_model/von_mises_computed_coupling.pvd") << project(sigma_vm,VS)
File("../../benchmarks/result_files/coupled_model/von_mises_analytical_coupling.pvd") << project(sigma_vm_analytical,VS)
error_stress_von_mises = Function(VS)
error_stress_von_mises.vector()[:] = abs(project(sigma_vm,VS).vector().get_local() - project(sigma_vm_analytical,VS).vector().get_local())
File("../../benchmarks/result_files/coupled_model/von_mises_stress_error_coupling.pvd") << project(error_stress_von_mises,VS)

File("../../benchmarks/result_files/coupled_model/difference_in_spherical_stress.pvd") << project(sigma_spherical - sigma_spherical_non_thermal,VS)
File("../../benchmarks/result_files/coupled_model/thermal_part_of_stress.pvd") << project(-(2 * mu + 3 * lmbda) * alpha * (T - T_0),VS)
error_stress_spherical = Function(VS)
error_stress_spherical.vector()[:] = abs(project(sigma_spherical,VS).vector().get_local() - project(sigma_spherical_non_thermal - (2 * mu + 3 * lmbda) * alpha * (T - T_0),VS).vector().get_local())
File("../../benchmarks/result_files/coupled_model/absolute_error_spherical_stress.pvd") << error_stress_spherical

#Convergence tests
plt.figure(figsize=[10,8])
a = plt.semilogy([1,2,3],error_u_vector,marker='o',linewidth=4)
plt.xticks([1,2,3],fontsize=18)
plt.yticks(fontsize=18)
plt.xlabel('Polynomial degree',fontsize=24)
plt.ylabel('Relative error',fontsize=24)
plt.axis('tight')
plt.savefig('../../benchmarks/result_files/coupled_model/Convergence_coupling_displacement_benchmark_comparison.png')
plt.show()
