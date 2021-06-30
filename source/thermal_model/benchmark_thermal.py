from __future__ import print_function
from dolfin import *
import matplotlib.pyplot as plt

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

#Computation of H^1_r(\omega) norm
def compute_h1r_norm(psi,mesh):
	r = SpatialCoordinate(mesh)[0]
	dx = Measure('dx', domain = mesh)
	a = inner(psi,psi)*r*dx + inner(grad(psi),grad(psi))*r*dx
	A = assemble(a)
	return sqrt(A)

error_T_vector = [] #List to store error in temperature w.r.t. polynomial degree
p = range(1,4) # List of polynomial degrees

# Known analytical solution, Thermal material properties and Boundary data
T_analytical = Expression('x[0]*x[0]*x[1]',degree = 3) 
VT_analytical = FunctionSpace(mesh,"CG",3) #Space for analytical solution
T_analytical = project(T_analytical,VT_analytical)
k = 10. # Thermal conductivity
h_fluid = 200. # Convection coefficient on \gamma_{sf}
h_right = 2000. # Convection coefficient on \gamma_{out}
h_bottom = 2000. # Convection coefficient on \gamma_{-}

for i in p:
	# Define function space
	VT = FunctionSpace(mesh,"CG",i) # Function space for temperature
	psi, T_ = TestFunction(VT), TrialFunction(VT) # Evaluate trial and test function
	T = Function(VT, name = "temperature increase")
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

	#Computing temperature relative error in relevant norm
	error_T = compute_h1r_norm(T_analytical-T,mesh)/compute_h1r_norm(T,mesh)
	error_T_vector.append(error_T)
	print("Temperature relative error in H^1_r norm : " + str(error_T))

# Plotting and visualization
File("../../benchmarks/result_files/thermal_model/temperature_computed.pvd") << T
File("../../benchmarks/result_files/thermal_model/temperature_analytical.pvd") << T_analytical
error_temperature = Function(VT) #Function for Spatial distribution of temperature absolute error
error_temperature.vector()[:] = abs(T_analytical.vector().get_local() - T.vector().get_local())
File("../../benchmarks/result_files/thermal_model/temperature_absolute_error.pvd") << error_temperature

# Plotting and printing convergence tests
plt.figure(figsize=[10,8])
a = plt.semilogy([1,2,3],error_T_vector,marker='o',linewidth=4)
plt.xticks([1,2,3],fontsize=18)
plt.yticks(fontsize=18)
plt.xlabel('Polynomial degree',fontsize=24)
plt.ylabel('Relative error',fontsize=24)
plt.axis('tight')
plt.savefig("../../benchmarks/result_files/thermal_model/convergence_test")
plt.show()
print("Relative error in H^1_r norm: "+ str(error_T_vector))
