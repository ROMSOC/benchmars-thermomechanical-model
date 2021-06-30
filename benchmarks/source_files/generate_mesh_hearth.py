# Copyright (C) 2015-2019 by the RBniCS authors
#
# This file is part of RBniCS.
#
# RBniCS is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# RBniCS is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with RBniCS. If not, see <http://www.gnu.org/licenses/>.
#

from dolfin import * #FEniCS library
from mshr import * #mshr - mesh generation component of FEniCS
from rbnics import * #RBniCS library
from rbnics.backends.dolfin.wrapping import counterclockwise
from rbnics.shape_parametrization.utils.symbolic import VerticesMappingIO
import os
import matplotlib.pyplot as plt #matplotlib library

# Define domain
domain = Polygon([Point(0.,0.),Point(7.05,0.),
				Point(7.05,7.265),Point(5.3,7.265),
				Point(5.3,4.065),Point(4.95,4.065),
				Point(4.95,3.565),Point(4.6,3.565),
				Point(4.6,2.965),Point(4.25,2.965),
				Point(4.25,2.365),Point(0.,2.365)])

# Define vertices mappings of affine shape parametrization. These will be used
# to partition the mesh in subdomains.
vertices_mappings = [
	{
		("0", "0"): ("0", "0"),
		("4.25", "0"): ("mu[6]/2", "0"),
		("0", "2.365"): ("0", "mu[0]")
	}, # subdomain 1
	{
		("0", "2.365"): ("0", "mu[0]"),
		("4.25", "0"): ("mu[6]/2", "0"),
		("4.25", "2.365"): ("mu[6]/2", "mu[0]")
	}, # subdomain 2
	{
		("4.25", "0"): ("mu[6]/2", "0"),
		("4.6", "0"): ("mu[7]/2", "0"),
		("4.25", "2.365"): ("mu[6]/2", "mu[0]")
	}, # subdomain 3
	{
		("4.25", "2.365"): ("mu[6]/2", "mu[0]"),
		("4.6", "0"): ("mu[7]/2", "0"),
		("4.6", "2.365"): ("mu[7]/2", "mu[0]")
	}, # subdomain 4
	{
		("4.6", "0"): ("mu[7]/2", "0"),
		("4.95", "0"): ("mu[8]/2", "0"),
		("4.6", "2.365"): ("mu[7]/2", "mu[0]")
	}, # subdomain 5
	{
		("4.6", "2.365"): ("mu[7]/2", "mu[0]"),
		("4.95", "0"): ("mu[8]/2", "0"),
		("4.95", "2.365"): ("mu[8]/2", "mu[0]")
	}, # subdomain 6
	{
		("4.95", "0"): ("mu[8]/2", "0"),
		("5.3", "0"): ("mu[9]/2", "0"),
		("4.95", "2.365"): ("mu[8]/2", "mu[0]")
	}, # subdomain 7
	{
		("4.95", "2.365"): ("mu[8]/2", "mu[0]"),
		("5.3", "0"): ("mu[9]/2", "0"),
		("5.3", "2.365"): ("mu[9]/2", "mu[0]")
	}, # subdomain 8
	{
		("5.3", "0"): ("mu[9]/2", "0"),
		("7.05", "0"): ("mu[5]/2", "0"),
		("5.3", "2.365"): ("mu[9]/2", "mu[0]")
	}, # subdomain 9
	{
		("5.3", "2.365"): ("mu[9]/2", "mu[0]"),
		("7.05", "0"): ("mu[5]/2", "0"),
		("7.05", "2.365"): ("mu[5]/2", "mu[0]")
	}, # subdomain 10
	{
		("4.25", "2.365"): ("mu[6]/2", "mu[0]"),
		("4.6", "2.365"): ("mu[7]/2", "mu[0]"),
		("4.25", "2.965"): ("mu[6]/2", "mu[0]+mu[1]")
	}, # subdomain 11
	{
		("4.25", "2.965"): ("mu[6]/2", "mu[0]+mu[1]"),
		("4.6", "2.365"): ("mu[7]/2", "mu[0]"),
		("4.6", "2.965"): ("mu[7]/2", "mu[0]+mu[1]")
	}, # subdomain 12
	{
		("4.6", "2.365"): ("mu[7]/2", "mu[0]"),
		("4.95", "2.365"): ("mu[8]/2", "mu[0]"),
		("4.6", "2.965"): ("mu[7]/2", "mu[0]+mu[1]")
	}, # subdomain 13
	{
		("4.6", "2.965"): ("mu[7]/2", "mu[0]+mu[1]"),
		("4.95", "2.365"): ("mu[8]/2", "mu[0]"),
		("4.95", "2.965"): ("mu[8]/2", "mu[0]+mu[1]")
	}, # subdomain 14
	{
		("4.95", "2.365"): ("mu[8]/2", "mu[0]"),
		("5.3", "2.365"): ("mu[9]/2", "mu[0]"),
		("4.95", "2.965"): ("mu[8]/2", "mu[0]+mu[1]")
	}, # subdomain 15
	{
		("4.95", "2.965"): ("mu[8]/2", "mu[0]+mu[1]"),
		("5.3", "2.365"): ("mu[9]/2", "mu[0]"),
		("5.3", "2.965"): ("mu[9]/2", "mu[0]+mu[1]")
	}, # subdomain 16
	{
		("5.3", "2.365"): ("mu[9]/2", "mu[0]"),
		("7.05", "2.365"): ("mu[5]/2", "mu[0]"),
		("5.3", "2.965"): ("mu[9]/2", "mu[0]+mu[1]")
	}, # subdomain 17
	{
		("5.3", "2.965"): ("mu[9]/2", "mu[0]+mu[1]"),
		("7.05", "2.365"): ("mu[5]/2", "mu[0]"),
		("7.05", "2.965"): ("mu[5]/2", "mu[0]+mu[1]")
	}, # subdomain 18
	{
		("4.6", "2.965"): ("mu[7]/2", "mu[0]+mu[1]"),
		("4.95", "2.965"): ("mu[8]/2", "mu[0]+mu[1]"),
		("4.6", "3.565"): ("mu[7]/2", "mu[0]+mu[1]+mu[2]")
	}, # subdomain 19
	{
		("4.6", "3.565"): ("mu[7]/2", "mu[0]+mu[1]+mu[2]"),
		("4.95", "2.965"): ("mu[8]/2", "mu[0]+mu[1]"),
		("4.95", "3.565"): ("mu[8]/2", "mu[0]+mu[1]+mu[2]")
	}, # subdomain 20
	{
		("4.95", "2.965"): ("mu[8]/2", "mu[0]+mu[1]"),
		("5.3", "2.965"): ("mu[9]/2", "mu[0]+mu[1]"),
		("4.95", "3.565"): ("mu[8]/2", "mu[0]+mu[1]+mu[2]")
	}, # subdomain 21
	{
		("4.95", "3.565"): ("mu[8]/2", "mu[0]+mu[1]+mu[2]"),
		("5.3", "2.965"): ("mu[9]/2", "mu[0]+mu[1]"),
		("5.3", "3.565"): ("mu[9]/2", "mu[0]+mu[1]+mu[2]")
	}, # subdomain 22
	{
		("5.3", "2.965"): ("mu[9]/2", "mu[0]+mu[1]"),
		("7.05", "2.965"): ("mu[5]/2", "mu[0]+mu[1]"),
		("5.3", "3.565"): ("mu[9]/2", "mu[0]+mu[1]+mu[2]")
	}, # subdomain 23
	{
		("5.3", "3.565"): ("mu[9]/2", "mu[0]+mu[1]+mu[2]"),
		("7.05", "2.965"): ("mu[5]/2", "mu[0]+mu[1]"),
		("7.05", "3.565"): ("mu[5]/2", "mu[0]+mu[1]+mu[2]")
	}, # subdomain 24
	{
		("4.95", "3.565"): ("mu[8]/2", "mu[0]+mu[1]+mu[2]"),
		("5.3", "3.565"): ("mu[9]/2", "mu[0]+mu[1]+mu[2]"),
		("4.95", "4.065"): ("mu[8]/2", "mu[0]+mu[1]+mu[2]+mu[3]")
	}, # subdomain 25
	{
		("4.95", "4.065"): ("mu[8]/2", "mu[0]+mu[1]+mu[2]+mu[3]"),
		("5.3", "3.565"): ("mu[9]/2", "mu[0]+mu[1]+mu[2]"),
		("5.3", "4.065"): ("mu[9]/2", "mu[0]+mu[1]+mu[2]+mu[3]")
	}, # subdomain 26
	{
		("5.3", "3.565"): ("mu[9]/2", "mu[0]+mu[1]+mu[2]"),
		("7.05", "3.565"): ("mu[5]/2", "mu[0]+mu[1]+mu[2]"),
		("5.3", "4.065"): ("mu[9]/2", "mu[0]+mu[1]+mu[2]+mu[3]")
	}, # subdomain 27
	{
		("5.3", "4.065"): ("mu[9]/2", "mu[0]+mu[1]+mu[2]+mu[3]"),
		("7.05", "3.565"): ("mu[5]/2", "mu[0]+mu[1]+mu[2]"),
		("7.05", "4.065"): ("mu[5]/2", "mu[0]+mu[1]+mu[2]+mu[3]")
	}, # subdomain 28
	{
		("5.3", "4.065"): ("mu[9]/2", "mu[0]+mu[1]+mu[2]+mu[3]"),
		("7.05", "4.065"): ("mu[5]/2", "mu[0]+mu[1]+mu[2]+mu[3]"),
		("5.3", "7.265"): ("mu[9]/2", "mu[0]+mu[1]+mu[2]+mu[3]+mu[4]")
	}, # subdomain 29
	{
		("5.3", "7.265"): ("mu[9]/2", "mu[0]+mu[1]+mu[2]+mu[3]+mu[4]"),
		("7.05", "4.065"): ("mu[5]/2", "mu[0]+mu[1]+mu[2]+mu[3]"),
		("7.05", "7.265"): ("mu[5]/2", "mu[0]+mu[1]+mu[2]+mu[3]+mu[4]")
	} # subdomain 30
]

# Loop over all mappings and set subdomain markers
for i, vertices_mapping in enumerate(vertices_mappings):
	print(i,vertices_mapping.keys())
	subdomain_i = Polygon([Point(*[float(coord) for coord in vertex]) for vertex in counterclockwise(vertices_mapping.keys())])
	domain.set_subdomain(i + 1, subdomain_i)

# Create mesh
mesh = generate_mesh(domain, 30) #30 specifies the mesh size.

# Create subdomains
subdomains = MeshFunction("size_t", mesh, 2, mesh.domains())

# Define classes for different boundaries
class Gamma_s(SubDomain):
	def inside(self, x, on_boundary):
		return x[0] < DOLFIN_EPS and on_boundary

class Gamma_minus1(SubDomain):
	def inside(self, x, on_boundary):
		return x[1] < DOLFIN_EPS and x[0] < (4.255 + DOLFIN_EPS) and on_boundary

class Gamma_minus2(SubDomain):
	def inside(self, x, on_boundary):
		return x[1] < DOLFIN_EPS and x[0] > (4.2 - DOLFIN_EPS) and x[0] < (4.65 + DOLFIN_EPS) and on_boundary

class Gamma_minus3(SubDomain):
	def inside(self, x, on_boundary):
		return x[1] < DOLFIN_EPS and x[0] > (4.55 - DOLFIN_EPS) and x[0] < (5.05 + DOLFIN_EPS) and on_boundary

class Gamma_minus4(SubDomain):
	def inside(self, x, on_boundary):
		return x[1] < DOLFIN_EPS and x[0] > (4.9 - DOLFIN_EPS) and x[0] < (5.4 + DOLFIN_EPS) and on_boundary

class Gamma_minus5(SubDomain):
	def inside(self, x, on_boundary):
		return x[1] < DOLFIN_EPS and x[0] > (5.25 - DOLFIN_EPS) and x[0] < (7.15 + DOLFIN_EPS) and on_boundary

class Gamma_out1(SubDomain):
	def inside(self, x, on_boundary):
		return x[0] > (7.04 - DOLFIN_EPS) and x[1] > (0. - DOLFIN_EPS) and x[1] < (2.4 + DOLFIN_EPS) and on_boundary

class Gamma_out2(SubDomain):
	def inside(self, x, on_boundary):
		return x[0] > (7.04 - DOLFIN_EPS) and x[1] > (2.3 - DOLFIN_EPS) and x[1] < (3. + DOLFIN_EPS) and on_boundary

class Gamma_out3(SubDomain):
	def inside(self, x, on_boundary):
		return x[0] > (7.04 - DOLFIN_EPS) and x[1] > (2.9 - DOLFIN_EPS) and x[1] < (3.6 + DOLFIN_EPS) and on_boundary

class Gamma_out4(SubDomain):
	def inside(self, x, on_boundary):
		return x[0] > (7.04 - DOLFIN_EPS) and x[1] > (3.5 - DOLFIN_EPS) and x[1] < (4.1 + DOLFIN_EPS) and on_boundary

class Gamma_out5(SubDomain):
	def inside(self, x, on_boundary):
		return x[0] > (7.04 - DOLFIN_EPS) and x[1] > (4. - DOLFIN_EPS) and x[1] < (7.3 + DOLFIN_EPS) and on_boundary

class Gamma_plus(SubDomain):
	def inside(self, x, on_boundary):
		return x[1] > (7.263 - DOLFIN_EPS) and x[0] > (4.94 - DOLFIN_EPS) and x[0] < (7.06 + DOLFIN_EPS) and on_boundary

class Gamma_sf1(SubDomain):
	def inside(self, x, on_boundary):
		return x[0] < 5.4 and x[0] > 5.28 and x[1] > (4.06 - DOLFIN_EPS) and x[1] < (7.27 + DOLFIN_EPS) and on_boundary

class Gamma_sf2(SubDomain):
	def inside(self, x, on_boundary):
		return x[1] < (4.07 + DOLFIN_EPS) and x[1] > (4.06 - DOLFIN_EPS) and x[0] > (4.93 - DOLFIN_EPS) and x[0] < (5.31 + DOLFIN_EPS) and on_boundary

class Gamma_sf3(SubDomain):
	def inside(self, x, on_boundary):
		return x[1] < (4.07 + DOLFIN_EPS) and x[1] > (3.56 - DOLFIN_EPS) and x[0] > (4.94 - DOLFIN_EPS) and x[0] < (4.96 + DOLFIN_EPS) and on_boundary

class Gamma_sf4(SubDomain):
	def inside(self, x, on_boundary):
		return x[0] < (4.97 + DOLFIN_EPS) and x[0] > (4.58 - DOLFIN_EPS) and x[1] > (3.55 - DOLFIN_EPS) and x[1] < (3.58 + DOLFIN_EPS) and on_boundary

class Gamma_sf5(SubDomain):
	def inside(self, x, on_boundary):
		return x[0] < (4.62 - DOLFIN_EPS) and x[0] > (4.58 + DOLFIN_EPS) and x[1] > (2.965 - DOLFIN_EPS) and x[1] < (3.565 + DOLFIN_EPS) and on_boundary

class Gamma_sf6(SubDomain):
	def inside(self, x, on_boundary):
		return x[0] < (4.62 + DOLFIN_EPS) and x[0] > (4.23 - DOLFIN_EPS) and x[1] > (2.96 - DOLFIN_EPS) and x[1] < (2.98 + DOLFIN_EPS) and on_boundary

class Gamma_sf7(SubDomain):
	def inside(self, x, on_boundary):
		return x[0] < (4.26 + DOLFIN_EPS) and x[0] > (4.24 - DOLFIN_EPS) and x[1] > (2.365 - DOLFIN_EPS) and x[1] < (2.965 + DOLFIN_EPS) and on_boundary

class Gamma_sf8(SubDomain):
	def inside(self, x, on_boundary):
		return x[0] < (4.26 + DOLFIN_EPS) and x[0] > (0. - DOLFIN_EPS) and x[1] > (2.36 - DOLFIN_EPS) and x[1] < (2.37 + DOLFIN_EPS) and on_boundary

# Set bounary markers by initializing instant of each class
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundaries.set_all(0)
gamma_s = Gamma_s()
gamma_s.mark(boundaries, 1)
gamma_minus1 = Gamma_minus1()
gamma_minus1.mark(boundaries, 2)
gamma_minus2 = Gamma_minus2()
gamma_minus2.mark(boundaries, 3)
gamma_minus3 = Gamma_minus3()
gamma_minus3.mark(boundaries, 4)
gamma_minus4 = Gamma_minus4()
gamma_minus4.mark(boundaries, 5)
gamma_minus5 = Gamma_minus5()
gamma_minus5.mark(boundaries, 6)
gamma_out1 = Gamma_out1()
gamma_out1.mark(boundaries, 7)
gamma_out2 = Gamma_out2()
gamma_out2.mark(boundaries, 8)
gamma_out3 = Gamma_out3()
gamma_out3.mark(boundaries, 9)
gamma_out4 = Gamma_out4()
gamma_out4.mark(boundaries, 10)
gamma_out5 = Gamma_out5()
gamma_out5.mark(boundaries, 11)
gamma_plus = Gamma_plus()
gamma_plus.mark(boundaries, 12)
gamma_sf1 = Gamma_sf1()
gamma_sf1.mark(boundaries, 13)
gamma_sf2 = Gamma_sf2()
gamma_sf2.mark(boundaries, 14)
gamma_sf3 = Gamma_sf3()
gamma_sf3.mark(boundaries, 15)
gamma_sf4 = Gamma_sf4()
gamma_sf4.mark(boundaries, 16)
gamma_sf5 = Gamma_sf5()
gamma_sf5.mark(boundaries, 17)
gamma_sf6 = Gamma_sf6()
gamma_sf6.mark(boundaries, 18)
gamma_sf7 = Gamma_sf7()
gamma_sf7.mark(boundaries, 19)
gamma_sf8 = Gamma_sf8()
gamma_sf8.mark(boundaries, 20)

# Save mesh data
os.system("mkdir ../input_data/mesh_data")
VerticesMappingIO.save_file(vertices_mappings, ".", "../data_files/mesh_data/hearth_vertices_mapping.vmp")
File("../data_files/mesh_data/hearth.xml") << mesh
File("../data_files/mesh_data/hearth_physical_region.xml") << subdomains
File("../data_files/mesh_data/hearth_facet_region.xml") << boundaries
XDMFFile("../data_files/mesh_data/hearth.xdmf").write(mesh)
XDMFFile("../data_files/mesh_data/hearth_physical_region.xdmf").write(subdomains)
XDMFFile("../data_files/mesh_data/hearth_facet_region.xdmf").write(boundaries)
File("../data_files/mesh_data/hearth.pvd") << mesh
File("../data_files/mesh_data/hearth_physical_region.pvd") << subdomains
File("../data_files/mesh_data/hearth_facet_region.pvd") << boundaries