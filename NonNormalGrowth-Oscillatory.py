from firedrake import *
import numpy as np
from firedrake import exp
import math
import warnings
warnings.filterwarnings("ignore")

nex = 600
degx = 2 
ndim = 2
dL = 2/nex
nStep = 1000
tau = 1


mm = UnitSquareMesh(180, 540)
mm.coordinates.dat.data[:,0] -= 0.5
mm.coordinates.dat.data[:,0] *=0.3
mm.coordinates.dat.data[:,1] *=0.9

# ####################### Solver parameters #######################
ffc_options = {"optimize": True, \
               "eliminate_zeros": True, \
               "precompute_basis_const": True, \
               "precompute_ip_const": True}
               
solver_parametersMine = {"ksp_type": "cg",  \
    "pc_type": "ilu",    \
    "ksp_rtol": 1e-10,  \
    "ksp_atol": 1e-10   \
}

##################### Defining Funtion Spaces ##########################
xx = SpatialCoordinate(mm)

PhiPhi = FunctionSpace(mm, "CG", 1)
phi = Function(PhiPhi, name = 'Phi') 
phihat = Function(PhiPhi, name = 'Phi hat') 
phiShifted = Function(PhiPhi, name = 'PhiShifted') 
phig = Function(PhiPhi, name = 'Phig') 
rho = Function(PhiPhi, name = 'rho') 
rhog = Function(PhiPhi, name = 'rhog') 
rhoShifted = Function(PhiPhi, name = 'rhoShifted')
dphi = TrialFunction(PhiPhi)
qq = TestFunction(PhiPhi) 

V = VectorFunctionSpace(mm, "CG", 1)
VDG0 = VectorFunctionSpace(mm, "DG", 0)
VDG1 = VectorFunctionSpace(mm, "DG", 1)
v  = TestFunction(V)
vDG1  = TestFunction(VDG1)
u  = Function(V, name = 'Displacement')
uDG0  = Function(VDG0, name = 'Displacement')
uDG1  = Function(VDG1, name = 'Displacement')
u1  = Function(V, name = 'Displacement 1')
du = TrialFunction(V)

FF = TensorFunctionSpace(mm,"CG", 1)
FFDG0 = TensorFunctionSpace(mm,"DG", 0)
Fe = Function(FF, name = 'Deformation Gradient0')
Feg = Function(FF, name = 'Deformation Gradient0 g')
FeShifted = Function(FF, name = 'Deformation Gradient0 Shifted')
F0DG = Function(FFDG0, name = 'Deformation Gradient DG0')
graduDG0 = Function(FFDG0, name = 'gradu DG0')
CauchyStress = Function(FF, name='Cauchy stress tensor')
Fhat  = TestFunction(FFDG0)

######################### Define phi ###################################
MeshSize = 1/600
eps = MeshSize * 8
sigma = 600 * 1

phihat.interpolate(conditional(abs(xx[0])<0.03,conditional(xx[1]<0.01,1,-1),-1))
phi.interpolate(phihat)

Iphi0 = ((sigma* (phi-phihat)**2) + eps * dot(grad(phi), grad(phi)) + 1/2/eps * ((phi**2 - 1)**2))*dx

Fs = derivative(Iphi0, phi,qq)
J = derivative(Fs, phi, dphi)

PETSc.Sys.Print("Smoothing initial phi")

solve(Fs == 0, phi, J=J, form_compiler_parameters=ffc_options)

########################################################################
toler=1e-10
X2 = interpolate(mm.coordinates, V)
X2DG0 = interpolate(mm.coordinates, VDG0)
X_array2 = X2.dat.data[:]
maxx = np.max(X_array2[:,0])
maxy = np.max(X_array2[:,1])
minx = np.min(X_array2[:,0])
miny = np.min(X_array2[:,1]) 



rho += 1
mu = Constant(0.0001)
lmbda = Constant(0.0001)
lambdaS= 1
muS=1
l2 = 0.25

I = Identity(2) # 2x2 Identity tensor
Fe.interpolate(as_tensor([ [1,0.15], [0,1] ] ))

ti=0
tau = 0.01
M = 1/2
u1.interpolate(as_vector([0, tau * M]))
alpha = 5

XU = X2.vector()[:] - u1.vector()[:]
Incoming =  np.where((XU[:,1] < miny) & (np.abs(XU[:,0]) < l2))[0]

uDG0.interpolate(u1)
XUDG0 = X2DG0.vector()[:] - uDG0.vector()[:]
IncomingDG0 =  XUDG0[:,1] <  miny

# Defining the energy function over the computational domain. Growing solid body is a NeoHookean material
def W(uf,Ff, phif):
	F = (I + grad(uf))*Ff
	dJ = det(F)
	dudx = grad(uf)
	I1 = tr(F.T * F)
	W1 = muS/2*(I1 - tr(I) - 2 * ln(dJ)) + lambdaS/2 * (dJ-1)**2 #Compressible Neo-Hookean material
	W2 = (mu/4 * inner( dudx + dudx.T , dudx + dudx.T ) + lmbda/2 * (tr( dudx )**2))
	W = (1+tanh(alpha*phif))/2 * W1 + (1-tanh(alpha*phif))/2 * W2
	return W
	

## Making sure that the initial body is in equilibrium	
PETSc.Sys.Print("Solving for u")
Iu = ( rho*W(u, Fe, phi) ) * dx
Fue = derivative(Iu, u,v)
bcu3 = DirichletBC(V,  [0,0], 3)
problem = NonlinearVariationalProblem(Fue, u, bcs=[bcu3])
sol1 = NonlinearVariationalSolver(problem)
sol1.solve()
	
Fe.interpolate((1+tanh(alpha*phi))/2 * (I+grad(u)) * Fe + (1-tanh(alpha*phi))/2 * I)
rho.interpolate((1+tanh(alpha*phi))/2 * rho/det(I + grad(u)) + (1-tanh(alpha*phi))/2 )

shft_array = u.vector().gather()
a = X2.vector().gather() - shft_array
shifted_coor = a.reshape(-1,2)

shifted_coor[shifted_coor[:,1] >= maxy, 1] = maxy
shifted_coor[shifted_coor[:,0] >= maxx, 0] = maxx
shifted_coor[shifted_coor[:,0] <  minx, 0] = minx
shifted_coor[shifted_coor[:,1] <  miny, 1] = miny

phiShifted.vector()[:] = phi.at(shifted_coor, tolerance=toler)
rhoShifted.vector()[:] = rho.at(shifted_coor, tolerance=toler)
FeShifted.vector()[:] = Fe.at(shifted_coor, tolerance=toler)

phi.interpolate(phiShifted)
Fe.interpolate(FeShifted)
rho.interpolate(rhoShifted)


outputFile = File("Osc.pvd");
outputFile.write(u, Fe, Feg, phi, phig, rho, graduDG0, F0DG, CauchyStress, time=ti)

for iStep in range(0,120):
	
	ti = ti+tau
	
	
	PETSc.Sys.Print("Time t = {:f}".format(ti))
	
	#### Generating phi_g from phi_n.
	PETSc.Sys.Print("Generating phi_g")
	shft_array = u1.vector().gather()
	a = X2.vector().gather() - shft_array
	shifted_coor = a.reshape(-1,2)
	
	shifted_coor[shifted_coor[:,1] >= maxy, 1] = maxy
	shifted_coor[shifted_coor[:,0] >= maxx, 0] = maxx
	shifted_coor[shifted_coor[:,0] <  minx, 0] = minx
	shifted_coor[shifted_coor[:,1] <  miny, 1] = miny
	
	phiShifted.vector()[:] = phi.at(shifted_coor, tolerance=toler)
	
	phig.interpolate(phiShifted)
			
	#### Constructing Feg and rhog. 
	Feg.vector()[:] = Fe.at(shifted_coor, tolerance=toler)
	rhog.vector()[:] = rho.at(shifted_coor, tolerance=toler)

	F0DG.interpolate(Feg)
	F0DG.vector()[IncomingDG0] =  np.array([[1,0.3*np.sin(20*ti)],[0,1]])
	
	rhog.vector()[Incoming] = 1
		
	#### Solving for u
	PETSc.Sys.Print("Solving for u")
	
	Iu = ( rhog*W(u, F0DG, phig) ) * dx
	Fue = derivative(Iu, u,v)
	bcu3 = DirichletBC(V,  [0,0], 3)
	problem = NonlinearVariationalProblem(Fue, u, bcs=[bcu3])
	sol1 = NonlinearVariationalSolver(problem)
	sol1.solve()
	
	
	## Updating non-advective part of Fe and rho
	graduDG0.interpolate(grad(u))
	F0DG.interpolate( (I+graduDG0) * F0DG )
	Fe.interpolate((1+tanh(alpha*phig))/2 * F0DG + (1-tanh(alpha*phig))/2 * I)
	rho.interpolate((1+tanh(alpha*phig))/2 * rhog/det(I + grad(u)) + (1-tanh(alpha*phig))/2 )
	
	## Advecting rho, Fe, and phig
	shft_array = u.vector().gather()
	a = X2.vector().gather() - shft_array
	shifted_coor = a.reshape(-1,2)
	
	shifted_coor[shifted_coor[:,1] >= maxy, 1] = maxy
	shifted_coor[shifted_coor[:,0] >= maxx, 0] = maxx
	shifted_coor[shifted_coor[:,0] <  minx, 0] = minx
	shifted_coor[shifted_coor[:,1] <  miny, 1] = miny
	
	phiShifted.vector()[:] = phig.at(shifted_coor, tolerance=toler)
	rhoShifted.vector()[:] = rho.at(shifted_coor, tolerance=toler)
	FeShifted.vector()[:] = Fe.at(shifted_coor, tolerance=toler)
	
	phi.interpolate(phiShifted)
	Fe.interpolate(FeShifted)
	rho.interpolate(rhoShifted)
	
	CauchyStress.interpolate(rho * diff(W(u-u, Fe, phi),Fe)*Fe.T)
	
	### Writing results at the end of timestep
	outputFile.write(u, Fe, Feg, phi, phig, rho, graduDG0, F0DG, CauchyStress, time=ti)




