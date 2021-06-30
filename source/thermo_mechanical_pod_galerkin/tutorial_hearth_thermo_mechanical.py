from __future__ import print_function
from dolfin import *
from rbnics import *
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
from rbnics.utils.io import Timer,TextLine

@PullBackFormsToReferenceDomain() #Decorator for operator transformation between parameterized domain to reference domain
@AffineShapeParametrization("../../benchmarks/data_files/mesh_data/hearth_vertices_mapping.vmp") #Decorator for shape parametrization with mapping defined in specified file
class HearthThermal(EllipticCoerciveProblem):
	
	# Default initialization of members
	def __init__(self, V, **kwargs):
		# Call the standard initialization
		EllipticCoerciveProblem.__init__(self, V, **kwargs)
		# ... and also store FEniCS data structures for assembly
		assert "subdomains" in kwargs
		assert "boundaries" in kwargs
		assert "mesh" in kwargs
		assert "h_cf" in kwargs
		assert "h_out" in kwargs
		assert "h_bottom" in kwargs
		self.subdomains, self.boundaries = kwargs["subdomains"], kwargs["boundaries"]
		self.u = TrialFunction(V) 
		self.v = TestFunction(V)
		self.dx = Measure("dx")(subdomain_data=subdomains)
		self.ds = Measure("ds")(subdomain_data=boundaries)
		self.subdomains = subdomains
		self.boundaries = boundaries
		self.reference_mesh = kwargs["mesh"]
		self.h_cf = kwargs["h_cf"]
		self.h_out = kwargs["h_out"]
		self.h_bottom = kwargs["h_bottom"]
		self.x0 = Expression("x[0]", element=V.ufl_element())
	# Return theta multiplicative terms of the affine expansion of the problem.
	def compute_theta(self, term):
		mu = self.mu
		if term == "a":
			theta_a0 = mu[10]
			theta_a1 = 1.0
			return (theta_a0, theta_a1)
		elif term == "f":
			theta_f0 = 1.0
			return (theta_f0, )
		else:
			raise ValueError("Invalid term for compute_theta().")
				
	# Return forms resulting from the discretization of the affine expansion of the problem operators.
	def assemble_operator(self, term):
		u = self.u
		v = self.v
		reference_mesh = self.reference_mesh
		dx = self.dx
		ds = self.ds
		h_cf = self.h_cf
		h_out = self.h_out
		h_bottom = self.h_bottom
		r = self.x0
		d_bottom = ds(2) + ds(3) + ds(4) + ds(5) + ds(6)
		d_out = ds(7) + ds(8) + ds(9) + ds(10) + ds(11)
		d_sf = ds(13) + ds(14) + ds(15) + ds(16) + ds(17) + ds(18) + ds(19) + ds(20)
		if term == "a":
			a0 = inner(grad(u), grad(v))*r*dx
			a1 = h_bottom*u*v*r*d_bottom + h_out*u*v*r*d_out + h_cf*u*v*r*d_sf
			return (a0, a1)
		elif term == "f":
			f0 = h_bottom*313*v*r*d_bottom + h_out*313*v*r*d_out + h_cf*1773*v*r*d_sf
			return (f0, )
		elif term == "inner_product":
			x0 = u*v*r*dx + inner(grad(u), grad(v))*r*dx
			return (x0,)
		else:
			raise ValueError("Invalid term for assemble_operator().")

@PullBackFormsToReferenceDomain() #Decorator for operator transformation between parameterized domain to reference domain
@AffineShapeParametrization("../../benchmarks/data_files/mesh_data/hearth_vertices_mapping.vmp") #Decorator for shape parametrization with mapping defined in specified file
class HearthMechanical(EllipticCoerciveProblem):

	# Default initialization of members
	def __init__(self, V, **kwargs):
		# Call the standard initialization
		EllipticCoerciveProblem.__init__(self, V, **kwargs)
		# ... and also store FEniCS data structures for assembly
		assert "subdomains" in kwargs
		assert "boundaries" in kwargs
		assert "mesh" in kwargs
		self.normal = as_vector(FacetNormal(kwargs["mesh"]))
		self.subdomains, self.boundaries = kwargs["subdomains"], kwargs["boundaries"]
		self.u = TrialFunction(V)
		self.v = TestFunction(V)
		self.dx = Measure("dx")(subdomain_data=subdomains)
		self.ds = Measure("ds")(subdomain_data=boundaries)
		self.subdomains = subdomains
		self.boundaries = boundaries
		self.x0 = Expression("x[0]", element=V.sub(0).ufl_element())
		self.x1 = Expression("x[1]", element=V.sub(1).ufl_element())
	# Return theta multiplicative terms of the affine expansion of the problem.
	def compute_theta(self, term):
		mu = self.mu
		if term == "a":
			theta_a0 = mu[11]
			theta_a1 = 2*mu[12]
			return (theta_a0, theta_a1, )
		elif term == "f":
			theta_f0 = 1.0
			return (theta_f0, )
		else:
			raise ValueError("Invalid term for compute_theta().")

	# Return strain tensor
	def strain(self,u):
		r = self.x0
		return sym(as_tensor([[u[0].dx(0), u[0].dx(1), 0. ], [u[1].dx(0), u[1].dx(1), 0.], [0., 0., u[0]/r]]))

	# Return forms resulting from the discretization of the affine expansion of the problem operators.
	def assemble_operator(self, term):
		u = self.u
		v = self.v
		dx = self.dx
		ds = self.ds
		r = self.x0
		x1 = self.x1
		n = self.normal
		d_bottom = ds(2) + ds(3) + ds(4) + ds(5) + ds(6)
		d_out = ds(7) + ds(8) + ds(9) + ds(10) + ds(11)
		d_sf = ds(13) + ds(14) + ds(15) + ds(16) + ds(17) + ds(18) + ds(19) + ds(20)
		if term == "a":
			a0 = (u[0].dx(0)+u[1].dx(1)+u[0]/r)*(v[0].dx(0)+v[1].dx(1)+v[0]/r)*r*dx
			a1 = (u[0].dx(0)*v[0].dx(0) + u[1].dx(1)*v[1].dx(1) + (u[0]*v[0])/(r)**2 + 0.5*(u[0].dx(1)+u[1].dx(0))*(v[0].dx(1)+v[1].dx(0))) * r * dx
			return (a0, a1,)
		elif term == "f":
			f0 = - dot( v, 7460*9.81*(7.265-x1)*n) * r * d_sf
			return (f0,)
		elif term == "inner_product":
			x0 = inner(u,v) * r * dx + inner(self.strain(u),self.strain(v)) * r * dx
			return (x0,)
		elif term == "dirichlet_bc":
			bc0 = [DirichletBC(self.V.sub(0), Constant(0.), self.boundaries, 1),
				DirichletBC(self.V.sub(1), Constant(0.), self.boundaries, 2),
				DirichletBC(self.V.sub(1), Constant(0.), self.boundaries, 3),
				DirichletBC(self.V.sub(1), Constant(0.), self.boundaries, 4),
				DirichletBC(self.V.sub(1), Constant(0.), self.boundaries, 5),
				DirichletBC(self.V.sub(1), Constant(0.), self.boundaries, 6),]
			return (bc0,)
		else:
			raise ValueError("Invalid term for assemble_operator().")

@ExactParametrizedFunctions() #Decorator for computing temperature field required for linear side
@PullBackFormsToReferenceDomain() #Decorator for operator transformation between parameterized domain to reference domain
@AffineShapeParametrization("../../benchmarks/data_files/mesh_data/hearth_vertices_mapping.vmp") #Decorator for shape parametrization with mapping defined in specified file
class HearthThermoMechanical(EllipticCoerciveProblem):

	# Default initialization of members
	def __init__(self, V, **kwargs):
		# Call the standard initialization
		EllipticCoerciveProblem.__init__(self, V, **kwargs)
		# ... and also store FEniCS data structures for assembly
		assert "subdomains" in kwargs
		assert "boundaries" in kwargs
		assert "mesh" in kwargs
		assert "hearth_problem_thermal" in kwargs
		assert "ref_temperature" in kwargs
		self.subdomains, self.boundaries = kwargs["subdomains"], kwargs["boundaries"]
		self.u = TrialFunction(V)
		self.v = TestFunction(V)
		self.dx = Measure("dx")(subdomain_data=subdomains)
		self.ds = Measure("ds")(subdomain_data=boundaries)
		self.subdomains = subdomains
		self.boundaries = boundaries
		self.hearth_problem_thermal = kwargs["hearth_problem_thermal"]
		self.T_0 = kwargs["ref_temperature"]
		self.x0 = Expression("x[0]", element=V.sub(0).ufl_element())

	# Return theta multiplicative terms of the affine expansion of the problem.
	def compute_theta(self, term):
		mu = self.mu
		if term == "a":
			theta_a0 = mu[11]
			theta_a1 = 2*mu[12]
			return (theta_a0, theta_a1,)
		elif term == "f":
			theta_f0 = (2 * mu[11] + 3 * mu[12]) * mu[13]
			return (theta_f0,)
		else:
			raise ValueError("Invalid term for compute_theta().")

	# Return strain tensor
	def strain(self,u):
		r = self.x0
		return sym(as_tensor([[u[0].dx(0), u[0].dx(1), 0. ], [u[1].dx(0), u[1].dx(1), 0.], [0., 0., u[0]/r]]))

	# Return forms resulting from the discretization of the affine expansion of the problem operators.
	def assemble_operator(self, term):
		u = self.u
		v = self.v
		dx = self.dx
		ds = self.ds
		T_0 = self.T_0
		T = self.hearth_problem_thermal._solution
		r = self.x0
		d_bottom = ds(2) + ds(3) + ds(4) + ds(5) + ds(6)
		d_out = ds(7) + ds(8) + ds(9) + ds(10) + ds(11)
		d_sf = ds(13) + ds(14) + ds(15) + ds(16) + ds(17) + ds(18) + ds(19) + ds(20)
		if term == "a":
			a0 = (u[0].dx(0)+u[1].dx(1)+u[0]/r)*(v[0].dx(0)+v[1].dx(1)+v[0]/r)*r*dx
			a1 = (u[0].dx(0)*v[0].dx(0) + u[1].dx(1)*v[1].dx(1) + (u[0]*v[0])/(r)**2 + 0.5*(u[0].dx(1)+u[1].dx(0))*(v[0].dx(1)+v[1].dx(0))) * r * dx
			return (a0, a1,)
		elif term == "f":
			f0 = (T-T_0) * (v[0].dx(0) + v[1].dx(1) + v[0]/r) * r * dx
			return (f0,)
		elif term == "inner_product":
			x0 = inner(u,v) * r * dx + inner(self.strain(u),self.strain(v)) * r * dx
			return (x0,)
		elif term == "dirichlet_bc":
			bc0 = [DirichletBC(self.V.sub(0), Constant(0.), self.boundaries, 1),
				DirichletBC(self.V.sub(1), Constant(0.), self.boundaries, 2),
				DirichletBC(self.V.sub(1), Constant(0.), self.boundaries, 3),
				DirichletBC(self.V.sub(1), Constant(0.), self.boundaries, 4),
				DirichletBC(self.V.sub(1), Constant(0.), self.boundaries, 5),
				DirichletBC(self.V.sub(1), Constant(0.), self.boundaries, 6),]
			return (bc0,)
		else:
			raise ValueError("Invalid term for assemble_operator().")

# Read the mesh file from specified path.
mesh = Mesh("../../benchmarks/data_files/mesh_data/hearth.xml")
domains = MeshFunction('size_t',mesh,mesh.topology().dim()) # Read domain marker
subdomains = MeshFunction("size_t", mesh, "../../benchmarks/data_files/mesh_data/hearth_physical_region.xml") # Read subdomain markers
boundaries = MeshFunction("size_t", mesh, "../../benchmarks/data_files/mesh_data/hearth_facet_region.xml") # Read boundary markers
dx = Measure('dx', domain = mesh, subdomain_data = domains) # Volume measure
ds = Measure('ds', domain = mesh, subdomain_data = boundaries) # Boundary measure
n = as_vector(FacetNormal(mesh)) # Edge unit normal vector

# Lam\'e parameters based on Young's modulus 5.e9 and Poisson's ratio 0.2
lame1 = 5.e9/2/(1+0.2)
lame2 = 5.e9*0.2/(1+0.2)/(1-2*0.2)
np.random.seed(42)

##########Thermal problem setup#########################

# 2A. Create Finite Element space (Lagrange P1)
VT = FunctionSpace(mesh, "Lagrange", 1) # For temperature

# 3A. Allocate an object of the Hearth class
hearth_problem_thermal = HearthThermal(VT, subdomains=subdomains, boundaries=boundaries, mesh=mesh, h_cf=200., h_out=2000., h_bottom=2000.)
#specify and set range of each parameter
mu_range = [(2.3,2.4), (0.5,0.7), (0.5,0.7), (0.4,0.6), (3.05,3.35), (13.5,14.5), (8.3,8.7), (8.8,9.2), (9.8,10.2), (10.4,10.8), (9.8,10.2), (2.08e9,2.08e9), (1.39e9,1.39e9), (1e-6,1e-6)]
hearth_problem_thermal.set_mu_range(mu_range)

# 4A. Prepare reduction with a POD-Galerkin method
#NOTE : truth_problem attribute is FEM problem and reduced_problem is RB problem
pod_galerkin_method_thermal = PODGalerkin(hearth_problem_thermal)
pod_galerkin_method_thermal.set_Nmax(100) #Maximum size of reduced basis space
pod_galerkin_method_thermal.set_tolerance(1e-4) #Maximum eigenvalue tolerance

##################Mechanical problem setup############################## 
# 2B. Create Finite Element space (Lagrange P1)
VM = VectorFunctionSpace(mesh,"Lagrange",1) # For mechanical

# 3B. Allocate an object of the HearthThermoMechanical class
hearth_problem_mechanical = HearthMechanical(VM, subdomains=subdomains, boundaries=boundaries, mesh=mesh)
#specify and set range of each parameter
mu_range = [(2.3,2.4), (0.5,0.7), (0.5,0.7), (0.4,0.6), (3.05,3.35), (13.5,14.5), (8.3,8.7), (8.8,9.2), (9.8,10.2), (10.4,10.8), (10.,10.), (1.9e9,2.5e9), (1.2e9,1.8e9), (1e-6,1e-6)]
hearth_problem_mechanical.set_mu_range(mu_range)

# 4B. Prepare reduction with a POD-Galerkin method
#NOTE : truth_problem attribute is FEM problem and reduced_problem is RB problem
pod_galerkin_method_mechanical = PODGalerkin(hearth_problem_mechanical)
pod_galerkin_method_mechanical.set_Nmax(100) #Maximum size of reduced basis space
pod_galerkin_method_mechanical.set_tolerance(1e-4) #Maximum eigenvalue tolerance

# 5B. Perform the offline phase
pod_galerkin_method_mechanical.initialize_training_set(1000) #Initialize training set with specified number of training parameters
reduced_hearth_problem_mechanical = pod_galerkin_method_mechanical.offline() #Perform offline phase

# 7B. Perform an error analysis
pod_galerkin_method_mechanical.initialize_testing_set(50) #Initialize error analysis set with specified number of parameters
pod_galerkin_method_mechanical.error_analysis() #Perform error analysis

# 8B1. Perform a speedup analysis - Compute time for truth solutions
pod_galerkin_method_mechanical.initialize_testing_set(50) #Initialize speedup analysis set with specified number of parameters
testing_set_speedup_analysis = pod_galerkin_method_mechanical.testing_set

pod_galerkin_method_mechanical._patch_truth_solve(True) # To disable cache reading

truth_timer = Timer("parallel") #Timer for computation of FEM solution
time_mechanical_truth = np.empty(len(testing_set_speedup_analysis)) #Storage of time taken for solving FEM equation. It is a vector of size of number of speedup analysis parameters

# Iteration over speedup analysis parameters for measuring time taken for FEM solution
for (mu_index, mu_test) in enumerate(testing_set_speedup_analysis):
	print(TextLine(str(mu_index), fill="#"))
	pod_galerkin_method_mechanical.truth_problem.set_mu(mu_test) #Set the parameter
	truth_timer.start()
	pod_galerkin_method_mechanical.truth_problem.solve() #Solve the FEM problem
	truth_time_mechanical = truth_timer.stop()
	print("Truth time mechanical : ",truth_time_mechanical)
	time_mechanical_truth[mu_index] = truth_time_mechanical #Save time taken for truth solve

pod_galerkin_method_mechanical._undo_patch_truth_solve(True) #To enable cache reading

np.save("time_mechanical_truth",time_mechanical_truth) #Save numpy array of time taken for FEM solution

# 8B2. Perform a speedup analysis - Compute time for reduced solutions
pod_galerkin_method_mechanical._patch_truth_solve(True) #To disable cache reading

reduced_timer = Timer("serial") #Timer for computation of reduced solution
max_basis_function = reduced_hearth_problem_mechanical.N # Size of reduced basis space
time_mechanical_reduced = np.empty([max_basis_function,len(testing_set_speedup_analysis)]) #Storage of time taken for solving RB equation. It is a matrix of size size of reduced basis space \times number of speedup analysis parameters

# Iteration over speedup analysis parameters for measuring time taken for RB solution
for basis_size in range(1,max_basis_function+1):
	for (mu_index, mu_test) in enumerate(testing_set_speedup_analysis):
		print(TextLine(str(mu_index), fill="#"))
		pod_galerkin_method_mechanical.reduced_problem.set_mu(mu_test) #Set the parameter
		reduced_timer.start()
		pod_galerkin_method_mechanical.reduced_problem.solve(basis_size) #Solve the RB problem
		rb_time_mechanical = reduced_timer.stop()
		print("Reduced time mechanical : ",rb_time_mechanical)
		time_mechanical_reduced[basis_size-1,mu_index] = rb_time_mechanical #Save time taken for reduced basis solution

pod_galerkin_method_mechanical._undo_patch_truth_solve(True) #To enable cache reading

np.save("time_mechanical_reduced",time_mechanical_reduced) #Save numpy array of time taken for RB solution

##################ThermoMechanical problem setup############################## 
# 2C. Create Finite Element space (Lagrange P1)
#Same as space for mechanical solution
T_0 = 298. # Reference temperature

# 3C. Allocate an object of the HearthThermoMechanical class
hearth_problem_thermo_mechanical = HearthThermoMechanical(VM, subdomains=subdomains, boundaries=boundaries, mesh=mesh, hearth_problem_thermal=hearth_problem_thermal, ref_temperature=T_0)
#specify and set range of each parameter
mu_range = [(2.3,2.4), (0.5,0.7), (0.5,0.7), (0.4,0.6), (3.05,3.35), (13.5,14.5), (8.3,8.7), (8.8,9.2), (9.8,10.2), (10.4,10.8), (9.8,10.2), (1.9e9,2.5e9), (1.2e9,1.8e9), (0.8e-6,1.2e-6)]
hearth_problem_thermo_mechanical.set_mu_range(mu_range)

# 4C. Prepare reduction with a POD-Galerkin method
#NOTE : truth_problem attribute is FEM problem and reduced_problem is RB problem
pod_galerkin_method_thermo_mechanical = PODGalerkin(hearth_problem_thermo_mechanical)
pod_galerkin_method_thermo_mechanical.set_Nmax(100) #Maximum size of reduced basis space
pod_galerkin_method_thermo_mechanical.set_tolerance(1e-4) #Maximum eigenvalue tolerance

# 5C. Perform the offline phase
pod_galerkin_method_thermo_mechanical.initialize_training_set(1000) #Initialize training set with specified number of training parameters
reduced_hearth_problem_thermo_mechanical = pod_galerkin_method_thermo_mechanical.offline() #Perform offline phase

# 6C. Perform a truth solve : Reference domain
online_mu = ( 2.365, 0.6, 0.6, 0.5, 3.2, 14.10, 8.50, 9.2, 9.9, 10.6, 10., lame1, lame2, 1e-6)
pod_galerkin_method_thermo_mechanical.truth_problem.set_mu(online_mu)
u_ref = pod_galerkin_method_thermo_mechanical.truth_problem.solve()
pod_galerkin_method_thermo_mechanical.truth_problem.export_solution(filename="reference_domain_fem")

# 6C. Perform a truth solve : Parametrized domain
online_mu = ( 2.365, 0.6, 0.6, 0.45, 3.2, 14.10, 8.30, 9.2, 9.9, 10.6, 10., lame1, lame2, 1e-6)
pod_galerkin_method_thermo_mechanical.truth_problem.set_mu(online_mu)
u_par = pod_galerkin_method_thermo_mechanical.truth_problem.solve()
pod_galerkin_method_thermo_mechanical.truth_problem.export_solution(filename="parametric_domain_fem")

# 7C1. Perform an error analysis - Compute truth solutions
pod_galerkin_method_thermo_mechanical.initialize_testing_set(50) #Initialize error analysis set with specified number of parameters
testing_set_error_analysis = pod_galerkin_method_thermo_mechanical.testing_set 

truth_solution_thermo_mechanical = list()

# Iteration over error analysis parameters for measuring time taken for FEM solution
for (mu_index, mu_test) in enumerate(testing_set_error_analysis):
	print(TextLine(str(mu_index), fill="#"))
	pod_galerkin_method_thermo_mechanical.truth_problem.set_mu(mu_test) #Set parameter
	truth_solution_thermo_mechanical.append(pod_galerkin_method_thermo_mechanical.truth_problem.solve()) #Solve and store FEM solution

# 8C1. Perform a speedup analysis - Compute time for truth solutions
pod_galerkin_method_thermo_mechanical.initialize_testing_set(50) #Initialize truth solution with specified number of parameters
testing_set_speedup_analysis = pod_galerkin_method_thermo_mechanical.testing_set

pod_galerkin_method_thermo_mechanical._patch_truth_solve(True) #To disable cache reading

truth_timer = Timer("parallel") #Timer for computation of FEM solution
time_thermo_mechanical_truth = np.empty(len(testing_set_speedup_analysis)) #Storage of time taken for solving FEM equation. It is a vector of size of number of speedup analysis parameters

# Iteration over speedup analysis parameters for measuring time taken for RB solution
for (mu_index, mu_test) in enumerate(testing_set_speedup_analysis):
	print(TextLine(str(mu_index), fill="#"))
	pod_galerkin_method_thermo_mechanical.truth_problem.set_mu(mu_test) #Set the parameter
	truth_timer.start()
	pod_galerkin_method_thermo_mechanical.truth_problem.solve() #Solve the RB problem
	truth_time_thermo_mechanical = truth_timer.stop()
	print("Truth time thermomechanical : ",truth_time_thermo_mechanical)
	time_thermo_mechanical_truth[mu_index] = truth_time_thermo_mechanical #Save time taken for reduced basis solution

pod_galerkin_method_thermo_mechanical._undo_patch_truth_solve(True) #To disable cache reading

np.save("time_thermo_mechanical_truth",time_thermo_mechanical_truth) #Save numpy array of time taken for RB solution

##################Thermal problem setup (continued)############################## 

# 5A. Perform the offline phase
pod_galerkin_method_thermal.initialize_training_set(1000) #Initialize training set with specified number of training parameters
reduced_hearth_problem_thermal = pod_galerkin_method_thermal.offline() #Perform offline phase

#6C. Perform an online solve : Reference domain
online_mu_reference = ( 2.365, 0.6, 0.6, 0.5, 3.2, 14.10, 8.50, 9.2, 9.9, 10.6, 10., lame1, lame2, 1e-6)
online_mu = online_mu_reference
pod_galerkin_method_thermo_mechanical.reduced_problem.set_mu(online_mu)
u_rb = pod_galerkin_method_thermo_mechanical.reduced_problem.solve()
pod_galerkin_method_thermo_mechanical.reduced_problem.export_solution(filename="reference_domain_thermomechanical_rb")
u_rb = pod_galerkin_method_thermo_mechanical.reduced_problem.basis_functions * u_rb
pod_galerkin_method_thermo_mechanical.truth_problem.mesh_motion.move_mesh()
File("HearthThermoMechanical/reference_domain_thermomechanical_spatial_error.pvd") << project(u_ref-u_rb,VM)
pod_galerkin_method_thermo_mechanical.truth_problem.mesh_motion.reset_reference()

# 6C. Perform an online solve : Parametrized domain
online_mu_parametrized = ( 2.365, 0.6, 0.6, 0.45, 3.2, 14.10, 8.30, 9.2, 9.9, 10.6, 10., lame1, lame2, 1e-6)
online_mu = online_mu_parametrized
pod_galerkin_method_thermo_mechanical.reduced_problem.set_mu(online_mu)
u_rb = pod_galerkin_method_thermo_mechanical.reduced_problem.solve()
pod_galerkin_method_thermo_mechanical.reduced_problem.export_solution(filename="parametric_domain_thermomechanical_rb")
u_rb = pod_galerkin_method_thermo_mechanical.reduced_problem.basis_functions * u_rb
pod_galerkin_method_thermo_mechanical.truth_problem.mesh_motion.move_mesh()
File("HearthThermoMechanical/parametric_domain_thermomechanical_spatial_error.pvd") << project(u_par-u_rb,VM)
pod_galerkin_method_thermo_mechanical.truth_problem.mesh_motion.reset_reference()

# 7C2. Perform an error analysis - Compute reduced basis solution
dx = Measure("dx")(subdomain_data=subdomains) #Volume measure
r = Expression("x[0]", element=VM.sub(0).ufl_element()) # 

max_basis_function = reduced_hearth_problem_thermo_mechanical.N # Size of reduced basis space
error_thermo_mechanical = np.empty([max_basis_function,len(testing_set_error_analysis)]) # Numpy array of size of reduced basis space \times number of error analysis parameters for storing error
# Iteration over error analysis parameters for measuring time taken for RB solution
for basis_size in range(1,max_basis_function+1):
	for (mu_index, mu_test) in enumerate(testing_set_error_analysis):
		print(TextLine(str(mu_index), fill="#"))
		pod_galerkin_method_thermo_mechanical.reduced_problem.set_mu(mu_test) #Set parameter
		rb_dofs = pod_galerkin_method_thermo_mechanical.reduced_problem.solve(basis_size) #Compute reduced basis degrees of freddom
		rb_solution = reduced_hearth_problem_thermo_mechanical.basis_functions[:basis_size] * rb_dofs #RB solution projected back to FEM space
		# Absolute and relative error measurement
		absolute_error = assemble(inner(truth_solution_thermo_mechanical[mu_index] - rb_solution,truth_solution_thermo_mechanical[mu_index] - rb_solution) * r * dx + inner(hearth_problem_thermo_mechanical.strain(truth_solution_thermo_mechanical[mu_index] - rb_solution), hearth_problem_thermo_mechanical.strain(truth_solution_thermo_mechanical[mu_index] - rb_solution)) * r * dx)
		error_thermo_mechanical[basis_size-1,mu_index] = np.sqrt(absolute_error / assemble(inner(truth_solution_thermo_mechanical[mu_index],truth_solution_thermo_mechanical[mu_index]) * r * dx + inner(hearth_problem_thermo_mechanical.strain(truth_solution_thermo_mechanical[mu_index]), hearth_problem_thermo_mechanical.strain(truth_solution_thermo_mechanical[mu_index])) * r * dx))

np.save("HearthThermoMechanical/error_analysis/error_thermo_mechanical",error_thermo_mechanical)

# 8C2. Perform a speedup analysis - Compute time for reduced solutions
pod_galerkin_method_thermo_mechanical._patch_truth_solve(True) #To disable cache reading

reduced_timer = Timer("serial") #Timer for computation of RB solution
time_thermo_mechanical_reduced = np.empty([max_basis_function,len(testing_set_speedup_analysis)]) #Storage of time taken for solving RB equation. It is a matrix of size size of reduced basis space \times number of speedup analysis parameters

# Iteration over speedup analysis parameters for measuring time taken for RB solution
for basis_size in range(1,max_basis_function+1):
	for (mu_index, mu_test) in enumerate(testing_set_speedup_analysis):
		print(TextLine(str(mu_index), fill="#"))
		pod_galerkin_method_thermo_mechanical.reduced_problem.set_mu(mu_test) #Set parameter
		reduced_timer.start()
		pod_galerkin_method_thermo_mechanical.reduced_problem.solve(basis_size) #Solve the RB problem
		rb_time_thermo_mechanical = reduced_timer.stop()
		print("Reduced time thermomechanical : ",rb_time_thermo_mechanical)
		time_thermo_mechanical_reduced[basis_size-1,mu_index] = rb_time_thermo_mechanical #Save time taken for reduced basis solution

pod_galerkin_method_thermo_mechanical._undo_patch_truth_solve(True) # To disable cache reading

np.save("time_thermo_mechanical_reduced",time_thermo_mechanical_reduced) #Save numpy array of time taken for RB solution

# 7A. Perform an error analysis
pod_galerkin_method_thermal.initialize_testing_set(50) #Initialize error analysis with specified number of parameters
pod_galerkin_method_thermal.error_analysis() #Perform error analysis

# 8A1. Perform a speedup analysis - Compute time for truth solutions
pod_galerkin_method_thermal.initialize_testing_set(50) #Initialize truth time computation with specified number of parameters
testing_set_speedup_analysis = pod_galerkin_method_thermal.testing_set

pod_galerkin_method_thermal._patch_truth_solve(True) #To enable cahce reading

truth_timer = Timer("parallel") #Timer for computation of FEM solution
time_thermal_truth = np.empty(len(testing_set_speedup_analysis)) #Storage of time taken for solving FEM equation. It is a vector of size of number of speedup analysis parameters

# Iteration over speedup analysis parameters for measuring time taken for FEM solution
for (mu_index, mu_test) in enumerate(testing_set_speedup_analysis):
	print(TextLine(str(mu_index), fill="#"))
	pod_galerkin_method_thermal.truth_problem.set_mu(mu_test) #Set the parameter
	truth_timer.start()
	pod_galerkin_method_thermal.truth_problem.solve() #Solve the FEM problem
	truth_time_thermal = truth_timer.stop()
	print("Truth time thermal : ",truth_time_thermal)
	time_thermal_truth[mu_index] = truth_time_thermal #Save time taken for truth solve

np.save("time_thermal_truth",time_thermal_truth) #Save time taken for computation of FEM solution

pod_galerkin_method_thermal._undo_patch_truth_solve(True) #To enable cache reading

# 8A2. Perform a speedup analysis - Compute time for reduced solutions
pod_galerkin_method_thermal._patch_truth_solve(True) #To disable cache reading
reduced_timer = Timer("serial") #Timer for computation of RB solution
max_basis_function = reduced_hearth_problem_thermal.N #Size of reduced basis space
time_thermal_reduced = np.empty([max_basis_function,len(testing_set_speedup_analysis)]) #Storage of time taken for solving RB equation. It is a matrix of size size of reduced basis space \times number of speedup analysis parameters

# Iteration over speedup analysis parameters for measuring time for RB solution
for basis_size in range(1,max_basis_function+1):
	for (mu_index, mu_test) in enumerate(testing_set_speedup_analysis):
		print(TextLine(str(mu_index), fill="#"))
		pod_galerkin_method_thermal.reduced_problem.set_mu(mu_test) #Set parameter
		reduced_timer.start()
		pod_galerkin_method_thermal.reduced_problem.solve(basis_size) #Solve the RB problem
		rb_time_thermal = reduced_timer.stop()
		print("Reduced time thermal : ",rb_time_thermal)
		time_thermal_reduced[basis_size-1,mu_index] = rb_time_thermal #Save time taken for RB solve

pod_galerkin_method_thermal._undo_patch_truth_solve(True) #To disable cache reading

np.save("time_thermal_reduced",time_thermal_reduced) #Save time taken for computation for RB solution

# 6A. Perform an online solve
online_mu = online_mu_reference
pod_galerkin_method_thermal.reduced_problem.set_mu(online_mu) #Set parameter
T_rb = pod_galerkin_method_thermal.reduced_problem.solve() #Reduced problem solve
pod_galerkin_method_thermal.reduced_problem.export_solution(filename="reference_domain_thermal_rb") #Save solution for visualization with paraview
T_rb = pod_galerkin_method_thermal.reduced_problem.basis_functions * T_rb #RB solution projected back to FEM space
pod_galerkin_method_thermal.truth_problem.set_mu(online_mu) #Set parameter
T = pod_galerkin_method_thermal.truth_problem.solve() #FEM problem solve
pod_galerkin_method_thermal.truth_problem.export_solution(filename="reference_domain_fem") #Save solution for visualization with paraview
pod_galerkin_method_thermal.truth_problem.mesh_motion.move_mesh() #Deform mesh as per geometric parameters
File("HearthThermal/reference_domain_thermal_spatial_error.pvd") << project(T-T_rb,VT) #Spatial error
pod_galerkin_method_thermal.truth_problem.mesh_motion.reset_reference() #Restore mesh to reference configuration

# 6B. Perform an online solve
pod_galerkin_method_mechanical.reduced_problem.set_mu(online_mu) #Set parameter
u_rb = pod_galerkin_method_mechanical.reduced_problem.solve() #Reduced problem solve
pod_galerkin_method_mechanical.reduced_problem.export_solution(filename="reference_domain_mechanical_rb") #Save solution for visualization with paraview
u_rb = pod_galerkin_method_mechanical.reduced_problem.basis_functions * u_rb #RB solution projected back to FEM space
pod_galerkin_method_mechanical.truth_problem.set_mu(online_mu) #Set parameter
u = pod_galerkin_method_mechanical.truth_problem.solve() #FEM problem solve
pod_galerkin_method_mechanical.truth_problem.export_solution(filename="reference_domain_fem") #Save solution for visualization with paraview
pod_galerkin_method_mechanical.truth_problem.mesh_motion.move_mesh() #Deform mesh as per geometric parameters
File("HearthMechanical/reference_domain_mechanical_spatial_error.pvd") << project(u-u_rb,VM) #Spatial error
pod_galerkin_method_mechanical.truth_problem.mesh_motion.reset_reference() #Restore mesh to reference configuration

# 6A. Perform an online solve
online_mu = online_mu_parametrized
pod_galerkin_method_thermal.reduced_problem.set_mu(online_mu) #Set parameter
T_rb = pod_galerkin_method_thermal.reduced_problem.solve() #Reduced problem solve
pod_galerkin_method_thermal.reduced_problem.export_solution(filename="parametric_domain_thermal_rb") #Save solution for visualization with paraview
T_rb = pod_galerkin_method_thermal.reduced_problem.basis_functions * T_rb #RB solution projected back to FEM space
pod_galerkin_method_thermal.truth_problem.set_mu(online_mu) #Set parameter
T = pod_galerkin_method_thermal.truth_problem.solve() #FEM problem solve
pod_galerkin_method_thermal.truth_problem.export_solution(filename="parametric_domain_fem") #Save solution for visualization with paraview
pod_galerkin_method_thermal.truth_problem.mesh_motion.move_mesh() #Deform mesh as per geometric parameters
File("HearthThermal/parametric_domain_thermal_spatial_error.pvd") << project(T-T_rb,VT) #Spatial error
pod_galerkin_method_thermal.truth_problem.mesh_motion.reset_reference() #Restore mesh to reference configuration

# 6B. Perform an online solve
pod_galerkin_method_mechanical.reduced_problem.set_mu(online_mu) #Set parameter
u_rb = pod_galerkin_method_mechanical.reduced_problem.solve() #Reduced problem solve
pod_galerkin_method_mechanical.reduced_problem.export_solution(filename="parametric_domain_mechanical_rb") #Save solution for visualization with paraview
u_rb = pod_galerkin_method_mechanical.reduced_problem.basis_functions * u_rb #RB solution projected back to FEM space
pod_galerkin_method_mechanical.truth_problem.set_mu(online_mu) #Set parameter
u = pod_galerkin_method_mechanical.truth_problem.solve() #FEM problem solve
pod_galerkin_method_mechanical.truth_problem.export_solution(filename="parametric_domain_fem") #Save solution for visualization with paraview
pod_galerkin_method_mechanical.truth_problem.mesh_motion.move_mesh() #Deform mesh as per geometric parameters
File("HearthMechanical/parametric_domain_mechanical_spatial_error.pvd") << project(u-u_rb,VM) #Spatial error
pod_galerkin_method_mechanical.truth_problem.mesh_motion.reset_reference() #Restore mesh to reference configuration

#Move output files to specified folder
os.system("mv *.npy ../../benchmarks/result_files/thermo_mechanical_pod_galerkin")
os.system("mv Hearth* ../../benchmarks/result_files/thermo_mechanical_pod_galerkin")