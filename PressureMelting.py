from firedrake import *
import numpy as np
from firedrake import exp
from mpi4py import MPI
from firedrake.petsc import PETSc
comm = MPI.COMM_WORLD
import warnings
warnings.filterwarnings("ignore")

nex=250
degx=2 
ndim=2
nStep=1000000
tau = 1e-5
dL = 1/nex/2
eps =  0.03
sigma = 1500 
 

mm = UnitSquareMesh(240, 300)
mm.coordinates.dat.data[:,0] *= .4 
mm.coordinates.dat.data[:,0] -= .2

mm.coordinates.dat.data[:,1] *= .5 
mm.coordinates.dat.data[:,1] -= .25

phiMesh = FiniteElement("CG", triangle, degx-1)
bdy0=[1,2,3,4]

# ####################### Functions
xx = SpatialCoordinate(mm)
PhiPhi = FunctionSpace(mm, phiMesh)
phi = Function(PhiPhi, name='phi')
phinew = Function(PhiPhi, name='phinew')
T = Function(PhiPhi, name='T')
modgradphi = Function(PhiPhi, name='modgradphi')
qq = TestFunction(PhiPhi)
h = Function(PhiPhi, name='h')
rho = Function(PhiPhi, name='rho')
rho0 = Function(PhiPhi, name='rho0')
Ws = Function(PhiPhi, name = 'Ws')
StepSmooth = Function(PhiPhi, name = 'StepSmooth')
phidot = Function(PhiPhi, name='Phidot')
P = Function(PhiPhi, name='Pressure') 
ag = Function(PhiPhi, name='a')

V = VectorFunctionSpace(mm, phiMesh)
u = Function(V, name = 'Displacement')
v = TestFunction(V)


FF = TensorFunctionSpace(mm,phiMesh)
F0 = Function(FF, name = 'Deformation Gradient0')
F0i = Function(FF, name = 'Deformation Gradient0i')
CauchyStress = Function(FF,name='Cauchy stress')
CauchyStressWater = Function(FF,name='Cauchy stress Water')
CauchyStressIce = Function(FF,name='Cauchy stress Ice')
dudx = Function(FF,name='dudx')
Ft = TestFunction(FF)

# #######################
X = interpolate(mm.coordinates, V)
X_array = X.vector().gather()
xnew = X_array.reshape(-1,2)
maxx = np.max(xnew[:,0])
maxy = np.max(xnew[:,1])
minx = np.min(xnew[:,0])
miny = np.min(xnew[:,1])
toler = 1e-12

#######################################################################
ti=0
#Ice and water prop
C1s = 1.76 #GPa mu/2
D1s = 3.27 #GPa lambda/2
C1l = 1e-3
D1l = 8.9 #GPa lambda/2

Tm = 273
T += 272
Hl=1/5

StepSmooth.interpolate(1/2*(1+tanh((xx[1]-0.125)/0.05)))
phi.interpolate((2*StepSmooth -1))

taum = 1e-3
chi  = 0.021

kappa2 = 1/taum*Tm/25
kappa = chi**2/Tm*5

rhos = 0.9
rhol = 1
rhoadded = rhol
I = Identity(ndim)
rho += 1

F0.interpolate(1/2 * (1-(tanh(phi/Hl))) * I * sqrt(rhol/rhos) + 1/2 * (1+(tanh(phi/Hl))) * I)


Latent=1
cs=1
cl=1

bcu1 = DirichletBC(V.sub(0), Constant(0), 1)
bcu2 = DirichletBC(V.sub(0), Constant(0), 2)
bcu3 = DirichletBC(V, [0,0], 3)
bcu4 = DirichletBC(V, [0,0], 4)
bcu = [bcu1, bcu2,  bcu3]

bcphi3 = DirichletBC(PhiPhi, -1, 3)
bcphi = [bcphi3]


outputFile = File("PressureMelting/PressureMelting.pvd");
outputFile.write(phi,modgradphi,F0, u, rho,Ws,P,CauchyStress,CauchyStressWater,CauchyStressIce,  time=ti)
aPre=1
Pressure = 1 * exp(-(((xx[0])**2))/1e-4)
	
	
for iStep in range(0,nStep):
	PETSc.Sys.Print("Time t = {:f}".format(ti))
	meq = ( modgradphi - sqrt(dot(grad(phi), grad(phi))) )*qq
	solve(meq*dx==0, modgradphi)
	
	############# Phi evolution #######################	
	PETSc.Sys.Print("Solving for phi")
	F = F0
	dJ = det(F)
	I1 = tr(F.T * F)
	Fl = sqrt(rhos/rhol)*F0
	dJl = det(Fl)
	I1l = tr(Fl.T * Fl)
	Ws.interpolate((C1l*(I1 - tr(I) - 2 * ln(dJ)) + D1l * (dJ-1)**2) - (C1s*(I1l - tr(I) - 2 * ln(dJl)) + D1s * (dJl-1)**2)) #Compressible Neo-Hookean material

	ag.interpolate(conditional(xx[1]<0.125,0.02,0 ))
	phieq = (phinew - phi)*qq + tau*kappa2 * (kappa * rho * dot(grad( phinew), grad(modgradphi * qq)) + ( \
		rho/4/Tm * phinew*(phinew**2-1) \
		+ rho*( Ws + ag ) * 1/2/Hl * (1-(tanh(phinew/Hl))**2) ) *modgradphi * qq)
	
	solve(phieq*dx == 0, phinew, bcs = bcphi)
		
	####### Determining Y0
	Xphi = np.hstack((xnew, phi.vector().gather()[:, np.newaxis]))
	range1 = [0.00-(1e-5), -0.5, 0]
	range2 = [0.00+(1e-5), 0.5, 1]
	maskY = np.logical_and(np.logical_and(np.logical_and(Xphi[:,0] >= range1[0], Xphi[:,0] <= range2[0]), 
	                      np.logical_and(Xphi[:,1] >= range1[1], Xphi[:,1] <= range2[1])),
	                      np.logical_and(Xphi[:,2] >= range1[2], Xphi[:,2] <= range2[2]))
	
	PETSc.Sys.Print('Wire vertical location is: ')
	aY=Xphi[maskY]
	LocY = np.min(aY[:,1]) + 0.01
	PETSc.Sys.Print(LocY)
	
	####### Evaluating pressure
	#Ramping up to P to ensure numerical convergence in the begining of the simulation
	if (-7/((iStep)/3+1)+7) < 6:
		ramping = (-7/((iStep)/3+1)+7)
	else:
		ramping = 6
	P.interpolate( ramping *  (1+Pressure  *  exp(-1 * (xx[1]-LocY)**2/1e-4)) * (1-(tanh(phi/Hl))**2) * 40)
	P.interpolate(conditional(((xx[1]-(LocY))**2 + xx[0]**2)>0.02**2 , 0, P  ))
	aPre = LocY
	
	u.interpolate(u-u)
	############### Momentum equation ###################
	#Liquid
	F = (I + grad(u))*F0
	dJ = det(F)
	I1 = tr(F.T * F)
	W1 = (C1l*(I1 - tr(I) - 2 * ln(dJ)) + D1l * (dJ-1)**2) #Compressible Neo-Hookean material
	#Solid
	Fl = sqrt(rhos/rhol)*(I + grad(u))*F0
	dJl = det(Fl)
	I1l = tr(Fl.T * Fl)
	W2 = (C1s*(I1l - tr(I) - 2 * ln(dJl)) + D1s * (dJl-1)**2)
	W = 1/2 * (1-(tanh(phinew/Hl))) * W2 + 1/2 * (1+(tanh(phinew/Hl))) * W1
	
	Iu = ( rho*W  +  P/2* u[1]) * dx 
	Fu = derivative(Iu, u,v)
	PETSc.Sys.Print("Solving for u")
	solve(Fu == 0, u, bcs=bcu)	
	
		
	############# Shifting and Updating F and rho due to the elasticity ####################
	############# This section is written to be compatible with parallel mpi run ###########
	PETSc.Sys.Print("Shiftings")
	
	shft_array = u.vector().gather()
	a = X.vector().gather() - shft_array
	shifted_coor = a.reshape(-1,2)
	
	if comm.rank == 0:
		shifted_coor[shifted_coor[:,1] >= maxy, 1] = maxy
		shifted_coor[shifted_coor[:,0] >= maxx, 0] = maxx
		shifted_coor[shifted_coor[:,1] <  miny, 1] = miny
		shifted_coor[shifted_coor[:,0] <  minx, 0] = minx
	shifted_coor = comm.bcast(shifted_coor, root=0)
	
	solve(inner(dudx - grad(u), Ft)*dx==0, dudx)
	m = (I + dudx)*F0
	rho0.interpolate( rho / det( (I + dudx) ) )
	F0i.interpolate(  m  )

	local_range = rho0.dat.dataset.layout_vec.getOwnershipRange()
	rhoarray = np.asarray(rho0.at(shifted_coor, tolerance=toler))
	rho.vector().set_local(rhoarray[local_range[0]:local_range[1]])
	rho.vector().apply('') 
	
	F0array = np.asarray(F0i.at(shifted_coor, tolerance=toler))
	F0array = F0array.ravel()
	local_range = F0.dat.dataset.layout_vec.getOwnershipRange()
	F0.vector().set_local(F0array[local_range[0]:local_range[1]])
	F0.vector().apply('') 
	
	phiarray = np.asarray(phinew.at(shifted_coor, tolerance=toler))
	local_range = phinew.dat.dataset.layout_vec.getOwnershipRange()
	phi.vector().set_local(phiarray[local_range[0]:local_range[1]])
	phi.vector().apply('')
	
	
	PETSc.Sys.Print("Evaluating Stress")
	#Liquid
	dJ = det(F0)
	I1 = tr(F0.T * F0)
	W1 = (C1l*(I1 - tr(I) - 2 * ln(dJ)) + D1l * (dJ-1)**2) #Compressible Neo-Hookean material
	#Solid
	Fl = sqrt(rhos/rhol)*F0
	dJl = det(Fl)
	I1l = tr(Fl.T * Fl)
	W2 = (C1s*(I1l - tr(I) - 2 * ln(dJl)) + D1s * (dJl-1)**2)
	#Applying phase field
	W = (1/2 * (1-(tanh(phi/Hl))) * W2 + 1/2 * (1+(tanh(phi/Hl))) * W1)
	CauchyStress.interpolate(rho * diff(W,F0)*F0.T)
	CauchyStressIce.interpolate(rho * diff(W2,F0)*F0.T)
	CauchyStressWater.interpolate(rho * diff(W1,F0)*F0.T)	
	
	#############################################################3
	
	ti += tau
	if iStep%1 == 0:
		outputFile.write(phi,modgradphi,F0, u, rho,Ws,P,CauchyStress,CauchyStressWater,CauchyStressIce,  time=ti)
			
	u.interpolate(u-u)
