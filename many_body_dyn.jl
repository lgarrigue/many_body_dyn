using LinearAlgebra, Plots
px = print

################ STATIC

size_plots = (1800,1000)

function init_tensor(N,Nx,complex=true)
	t = Tuple(fill(Nx,N))
	complex ? zeros(ComplexF64,t) : zeros(t)
end

mutable struct ParamsN # N-body function parameters in 1D, static
	N # bodies number
	Nx # spatial discretization
	L # physical spatial length
	dx # spatial volume element
	x_axis
	Ci # CartesianIndices, from i ∈ [1,Nx^N] returns [x1,…,xN] ∈ [1,Nx]^N, the Nbody coordinates
	Li # LinearIndices, from [x1,…,xN] ∈ [1,Nx]^N returns i ∈ [1,Nx^N]
	ΔmatPerN # N body laplacian matrix operator
	W # diagonal of the interaction operator
	n_basis # = Nx^N, total size of the basis
	basis_one_body # a basis of the one-body space
	basis_N # a basis of the space of N-body wavefunctions, indexed corresponding to Ci
	mymethod # if ture, CartesianIndices / LinearIndices / Flatten / Reshape are functions that we build, otherwise they are from native methods
	function ParamsN(N,Nx,L)
		Ψ = init_tensor(N,Nx)
		p = new(N,Nx,L)
		p.dx = L/Nx; p.x_axis = [i*p.dx for i=0:p.Nx-1]
		p.n_basis = Nx^N
		p.mymethod = false
		build_Ci_Li(p)
		# fill_basis_N(p)
		# fill_basis_1(p)
		p
	end
end

indexof(a,A) = Int(findfirst(x->x==a, A)) # index of a in A
isC(x) = typeof(x) == Complex{Float64}

function Reshape(ϕ,p)
	if !p.mymethod
		return reshape(ϕ,dimsNb(p))
	else
		φ = init_tensor(p.N,p.Nx,isC(ϕ[1]))
		for i=1:p.n_basis
			# px("Complex ? ",isa(ϕ[p.Li[i]],Complex{Int64})," ",ϕ[p.Li[i]])
			φ[i] = ϕ[p.Li[i]]
		end
		# px("typ ",size(Ψ.data[t-1]))
		return φ
	end
end

function Flatten(ϕ,p)
	if !p.mymethod
		return vcat(ϕ...)
	else
		return [ϕ[p.Ci[i]] for i=1:p.n_basis]
	end
end

function build_Ci_Li(p)
	if !p.mymethod
		Ψ = init_tensor(p.N,p.Nx)
		p.Ci = CartesianIndices(Ψ); p.Li = LinearIndices(Ψ)
	else
		Ci = []
		for i=1:p.n_basis
			a = []
			for n=1:p.N
				add = n==1 ? 0 : 1
				v = Int(mod1(1+floor((i-1)/(p.Nx^(n-1))),p.Nx))
				# px("v ",v)
				push!(a,v)
			end
			push!(Ci,CartesianIndex(Tuple(a)))
		end
		p.Ci = Ci
		Li = init_tensor(p.N,p.Nx,false)
		for i=1:p.n_basis
			ci = Ci[i]
			Li[ci] = i
		end
		Li = Int.(Li)
		p.Li = Li
	end
	# println("Ci\n",Ci,"\nLi\n",Li,"\n")
end

function ΔmatPer(n,dx) # One-body Laplacian operator Δ
	Δ = zeros(n,n)
	for i=1:n
		for j=1:n
			Δ[i,j] = i==j ? -2 : (abs(i-j) == 1 ? 1 : 0)
		end
	end
	Δ[1,end] = 1
	Δ[end,1] = 1
	Δ/(dx^2) 
end

function init_N_Laplacian(p) # computes the N-body Laplacian matrix
	Δ = ΔmatPer(p.Nx,p.dx)
	p.ΔmatPerN = one_to_Nb_op(Δ,p).data
end

function init_Nb_interaction_diagonal(w,p)
	if p.N >=2
		p.W = Nb_interaction_diagonal(w,p.N,p)
	end
end

# tuple (Nx,Nx,...,Nx) (N times)
dimsNb(p) = Tuple(fill(p.Nx,p.N))

mutable struct Nfun # N-body function parameters in 1D, static
	data # tensor
	N; Nx
	function Nfun(N,p,complex=true) # initialized to x, Nx x Nx ... x Nx array (N times)
		new(init_tensor(N,p.Nx,complex),N,p.Nx)
	end
end

function basis_fun_N(X,p,complex=true) # vector which is equal to one on X=[x1,...,xN], and 0 everywhere else
	Ψ = init_tensor(p.N,p.Nx,complex)
	Ψ[X] = 1
	Ψ
end

function basis_fun_1(X,p,complex=true) # vector which is equal to one on x, and 0 everywhere else
	Ψ = init_tensor(1,p.Nx,complex)
	Ψ[X] = 1
	Ψ
end

function fill_basis_N(p) # fills the parameters with basis functions
	basis = []
	for i=1:p.n_basis
		Ψ = basis_fun_N(p.Ci[i],p)
		push!(basis,Ψ)
	end
	p.basis_N = basis
end

function fill_basis_1(p) # fills the parameters with basis functions
	basis = []
	for i=1:p.Nx
		Ψ = basis_fun_1(i,p)
		push!(basis,Ψ)
	end
	p.basis_one_body = basis
end

function density(N,Ψ,p) # Ψ is a tensor, p is ParamsN
	ρ = zeros(p.Nx)
	for i in eachindex(Ψ)
		r = p.Ci[i]
		ρ[r[1]] += abs2(Ψ[i])
	end
	N*ρ
end

function pair_density(N,Ψ,p) # Ψ is a tensor, p is ParamsN
	ρ = zeros(p.Nx,p.Nx)
	for i in eachindex(Ψ)
		r = p.Ci[i]
		ρ[r[1],r[2]] += abs2(Ψ[i])
	end
	(N*(N-1)/2)*ρ
end

# Ψ is an Nfun. Gives the one-body density of Ψ
density(Ψ,p) = density(Ψ.N,Ψ.data,p)
pair_density(Ψ,p) = pair_density(Ψ.N,Ψ.data,p)

function antisym_from_orbs(orbs,p) # takes a list of N orthonormal orbitals (arrays) and produces an antisymmetric N-body wavefunction from it. p is a ParamsN
	@assert p.N  == length(orbs)
	@assert p.Nx == length(orbs[1])
	Ψ = Nfun(p.N,p)
	for l in eachindex(Ψ.data)
		r = p.Ci[l]
		A = [orbs[i][r[n]] for i=1:p.N, n=1:p.N]
		Ψ.data[l] = det(A)/sqrt(factorial(p.N))
	end
	Ψ
end

function sym_prod_from_orbs(orbs,p) # n-body symmetric product ψ1(x1)*ψ2(x2)*...*ψN(xN)
	@assert p.N == length(orbs)
	@assert p.Nx == length(orbs[1])
	ψ = Nfun(p.N,p)
	for l in eachindex(ψ.data)
		# px("test ",p.ci[l])
		r1 = p.Ci[l][1]
		ψ.data[l] = orbs[1][r1]
		for j=2:p.N
			rj = p.Ci[l][j]
			ψ.data[l] *= orbs[j][rj]
		end
	end
	ψ
end

function sym_prod_from_orb(orb,p) # n-body symmetric product ψ(x1)*ψ(x2)*...*ψ(xn)
	orbs = [copy(orb) for i=1:p.N]
	sym_prod_from_orbs(orbs,p)
end

function randorbs(p) # obtains N random orbials
	orbs = []
	for n=1:p.N
		push!(orbs,randn(p.Nx))
	end
	orbs
end

### Operators

mutable struct Nop # N-body operators, static
	data # Nx^N × Nx^N matrix, real entries
	N; Nx; n_basis # n_basis = Nx^N
end

function Nop(N,p)
	n_basis = p.Nx^N
	M = zeros(n_basis,n_basis) # real entries
	new(M,N,p.Nx,n_basis)
end

function one_body_mult_op(v,p) # one-body multiplication operator V[x,y] = δ_{xy} v(x). v is a 1d array
	V = Nop(1,p)
	for i=1:p.Nx
		V.data[i,i] = v[i]
	end
	V
end

function Nb_interaction_diagonal(w,N,p) # Nbody multiplication operator ww(x1,…,xN) = Σ_{j < i} w(|xi-xj|), is the diagonal of W[X,Y] = δ_{XY} ww(X). p is ParamsN, w is a function. params_grid has to have a member x_axis
	@assert N>=2
	nb = p.Nx^N
	W = zeros(nb)
	for i=1:nb
		X = p.Ci[i]
		for n=1:N
			Xn = X[n]
			for m=1:N
				Xm = X[m]
				xn = p.x_axis[Xn]; xm = p.x_axis[Xm]
				# W[i] += w(abs(xn-xm)/p.L)
				W[i] += sum([w(abs(mod(xn-xm-q*p.L,p.L))/p.L) for q=-2:2])
			end
		end
	end
	W
end

# vv(x1,…,xN) = Σ_{i} v[xi], diagonal of the Nbody operator V[X,Y] = δ_{XY} vv(X). p has to contain Nx and Ci, v is a 1d array
function Nb_mult_op_diag(v,N,p) 
	V = zeros(p.Nx^N)
	for i=1:p.Nx^N
		for n=1:N
			r = p.Ci[i][n]
			V[i] += v[r]
		end
	end
	V
end

# Second quantization (matrix) of the one-body matrix G
# X = (X1,X2,...,XN), G(X,Y) = Σ(i∈{1,...,N}) G(Xi,Yi) Π(j∈{1,...,N}\{i}) δ(Xj,Yj)
function one_to_Nb_op(G,p) 
	Gsq = zeros(p.n_basis,p.n_basis)
	for i=1:p.n_basis
		Ci_i = p.Ci[i]
		for j=1:p.n_basis
			Ci_j = p.Ci[j]
			for n=1:p.N
				xn = Ci_i[n]; yn = Ci_j[n]
				b = true
				for m=1:p.N
					xm = Ci_i[m]; ym = Ci_j[m]
					if m != n && xm != ym
						b = false
					end
				end
				if b
					Gsq[i,j] += G[xn,yn]
				end
			end
		end
	end
	Nop(Gsq,p.n_basis,p.Nx,p.n_basis)
end

### Examples of objects

# typ ∈ [gaussian,cut_gaussian,sin]
function create_orbital(c,σ,p,typ="cut_gaussian") # creates a gaussianish orbital which has compact support. c is the center in spatial coordinates [0,1], σ is the ecart-type in coordinates [0,1] too
	fun(x) = 0
	if typ in ["gaussian","cut_gaussian"]
		fun(x) = exp(-(x - c)^2 / (2*(σ^2)))
	elseif typ=="sin"
		fun(x) = sin(2π*(x-c))
	end
	g = 0
	if typ=="cut_gaussian"
		g = max.(0,fun.(p.x_axis/p.L) .- exp(-1/2))
	else
		g = fun.(p.x_axis/p.L)
	end
	g/sqrt(norm2(g,p)) # normalize to 1
end

# Create N orbitals, shifted in space
function create_N_orth_orbs(p) # only one and two is implemented
	orbs = []
	dN = 1/N
	c0 = dN
	for n=1:p.N
		c = c0 + dN*(n-1)
		σ = 0.2/p.N
		push!(orbs,create_orbital(c,σ,p))
	end
	orbs
end

### Norms

# norm2 of one-body orbitals
norm2(a,p) = sum(abs2.(a))*p.dx

function norm2_Nb(ψ,p) # ψ[Nx,…,Nx]
	x = 0
	for i in eachindex(ψ)
		x += abs2(ψ[i])
	end
	x*p.L^N/p.n_basis
end

################ DYNAMIC

# pS for pStatic and pD for pDynamic

function init_dyn_tensor(N,Nt,Nx,complex=true) # one should never do fill(zeros(...),p.Nt) because then all elements of Ψ[t] for all t, are equal, because they all have the same pointeur
	φ = init_tensor(N,Nx,complex)
	Ψ = []
	for t=1:Nt
		push!(Ψ,copy(φ))	
	end
	Ψ
end

mutable struct ParamsNdyn # Nbody wave function with time variable
	Nt; T; dt; t_axis; N
	Nx; L; dx; x_axis; n_basis
	Ci; Li
	mymethod
	U_Nb_Δ # unitary evolution operator (matrix) with Laplacian
	U_Nb_W_diag # diagonal of the unitary evolution operator (matrix) with interaction, which is diagonal
	function ParamsNdyn(Nt,T,pS)
		pD = new(Nt,T)
		pD.dt = T/Nt
		pD.t_axis = [i*pD.dt for i=0:pD.Nt-1]
		pD.N = pS.N; pD.Nx = pS.Nx; pD.L = pS.L; pD.dx = pS.dx; pD.x_axis = pS.x_axis; pD.n_basis = pS.n_basis; pD.Ci = pS.Ci; pD.Li = pS.Li
		pD.mymethod = pS.mymethod
		init_Nb_Laplacian_dyn(pD,pS)
		pD
	end
end

# computes the N-body Laplacian evolution operator matrix
init_Nb_Laplacian_dyn(pD,pS) = pD.U_Nb_Δ = exp(im * 0.5 * pD.dt * pS.ΔmatPerN)

function init_Nb_interaction_dyn(pD,pS) # computes the diagonal of the N-body interaction evolution operator matrix
	if pD.N >= 2
		pD.U_Nb_W_diag = exp.(-im*pD.dt*pS.W)
	end
end

struct NfunDyn # N-body wavefunction, the first variable being the time variable. It's not simply a tensor because we need to be able to make tensor products of wavefunctions of different body numbers
	data # it is data[t][x1,...,xN]
	N; Nt; Nx
	function NfunDyn(N,p,complex=true) # p is pD
		Ψ = init_dyn_tensor(N,p.Nt,p.Nx,complex)
		new(Ψ,N,p.Nt,p.Nx)
	end
end

function density_dyn(Ψ,p) # gives the one-body density of Ψ. p is ParamsNdyn
	ρ = init_dyn_tensor(1,p.Nt,p.Nx,false)
	for t=1:p.Nt
		ρ[t] = density(Ψ.N,Ψ.data[t],p)
	end
	ρ
end

function pair_density_dyn(Ψ,p) 
	ρ = init_dyn_tensor(2,p.Nt,p.Nx,false)
	for t=1:p.Nt
		ρ[t] = pair_density(Ψ.N,Ψ.data[t],p)
	end
	ρ
end

function norm2_N_dyn(ψ,p) # ψ[Nt][Nx,…,Nx]
	x = 0
	for t=1:p.Nt
		x += norm2_Nb(ψ[t],p)
	end
	x*p.dt
end

### Time evolution

function evolution_one_step_Nb(ψ,vx,p) # one time step evolution. ψ[Nx,…,Nx] is Nbody array, vx[Nx] is a one-body array. p is ParamsNdyn
	V = Nb_mult_op_diag(vx,p.N,p)
	ψ_lin = Flatten(ψ,p)
	ϕ1 = p.U_Nb_Δ * ψ_lin
	# px(length(ϕ1)," ",length(V)," ",length(p.U_Nb_W_diag)," ",size(V,2))
	ϕ2 = p.N >= 2 ? ϕ1 .* p.U_Nb_W_diag : ϕ1
	ϕ3 = ϕ2 .* exp.(-im*p.dt*V)
	ϕ4 = p.U_Nb_Δ * ϕ3
	Reshape(ϕ4,p)
end

function NbSchro(Ψ0,vxt,p) # Total evolution. Ψ0 is Nfun, vxt[Nt,Nx]
	@assert Ψ0.N==p.N
	Ψ = NfunDyn(Ψ0.N,p)
	Ψ.data[1] = Ψ0.data
	for t=2:p.Nt
		# px("typ ",size(Ψ.data[t-1]))
		Ψ.data[t] = evolution_one_step_Nb(Ψ.data[t-1],vxt[t],p)
	end
	Ψ
end

############################## Tests

function test_unit_norm(ψ,p)
	s = sum([abs(norm2_Nb(ψ.data[t],p)-1) for t=1:p.Nt])
	px("Test unit norm ",s,"\n")
end

function test_vcat_and_Li(ψ,p)
	b = true
	ψ_lin = Flatten(ϕ,p)
	for i=1:p.n_basis
		if ψ_lin[i] != ψ[i]
			b = false
		end
	end
	if p.N == 2
		for i=1:p.Nx
			for j=1:p.Nx
				m = p.Li[i,j]
				if ψ[i,j] != ψ_lin[m]
					b = false
				end
			end
		end
	end
	px("Test vcat Li ",b)
end

function test_Ci(ψ,p)
	b = true
	if p.N==2
		for i_lin=1:p.Nx^2
			ci = p.Ci[i_lin]
			if ψ[ci[1],ci[2]]!=ψ[i_lin]
				b = false
			end
		end
	end
	px("Test Ci ",b)
end

function test_reshape(ψ,p)
	b = true
	Ψ = Reshape(Flatten(ψ,p),p)
	for i=1:p.n_basis
		if ψ[i] != Ψ[i]
			b = false
		end
	end
	px("Test reshape ",b)
end

function test_antisym(Ψ,p) # tests that Ψ(t,x) is antisymmetric, via its pair density
	ρ2 = pair_density_dyn(Ψ,p) 
	s = sum([ρ2[t][x,x] for x=1:p.Nx, t=1:p.Nt])
	px("Test antisymmetry ",s,"\n")
end

############################## Creation of potentials

function plus(f,g)
	sum = []
	for t=1:length(f)
		push!(sum,f[t].+g[t])
	end
	sum
end

function gaussian_fun(freq,τ,σ)
	f(t,x) = exp(-((x-τ)-0.2*(cos(freq*t)-1))^2/(2*σ^2))
	f
end

function fun2pot(f,p) # f is in reduced coordinates (x from 0 to 1)
	v = init_dyn_tensor(1,p.Nt,p.Nx,false)
	for t=1:p.Nt
		for x=1:p.Nx
			v[t][x] = f(p.t_axis[t]/p.T,p.x_axis[x]/p.L)
		end
	end
	v
end

function from_fun_tx(freq,p)
	σ = 0.2; c = 20; τ1 = 1/4; τ2 = 3/4
	V(t,x) = -c*(gaussian_fun(freq,τ1,σ)(t,x) + gaussian_fun(freq,τ2,σ)(t,x))
	Vper(t,x) = V(t,x)+V(t,x+1)+V(t,x+2)+V(t,x+3)+V(t,x+4)+V(t,x-1)+V(t,x-2)+V(t,x-3)+V(t,x-4) # (approximately) periodizes it
	fun2pot(Vper,p)
end

function test_Nb_Lap(f,g,pS) # tests whether Δ(f(x)*g(y)) = (Δf(x))g(y) + f(x)(Δg(y))
	@assert pS.N == 2
	Δ = ΔmatPer(pS.Nx,pS.dx)
	Δ2 = one_to_Nb_op(Δ,pS).data
	fg = sym_prod_from_orbs([f,g],pS).data # f(x)*g(y)
	fg_cat = Flatten(fg,pS)
	a = Reshape(Δ2*fg_cat,pS)
	b = sym_prod_from_orbs([Δ*f,g],pS).data .+ sym_prod_from_orbs([f,Δ*g],pS).data
	px("Test Nb Δ ",norm2_Nb(abs.(a.-b),pS),"\n")
	C = heatmap(abs.(fg),axis=nothing)
	A = heatmap(abs.(a),axis=nothing)
	B = heatmap(abs.(b),axis=nothing)
	pl = plot(C,A,B)
	savefig(pl,"plots/test_lap.png")
end

############################## Plots

function plot2d(f,pD,title="")
	ff = [f[t][x] for x=1:pD.Nx, t=1:pD.Nt]
	heatmap(pD.t_axis,pD.x_axis,ff,title=title)
end

function product_orbs_dyn(o1,o2,p) 
	ψ = init_dyn_tensor(2,p.Nt,p.Nx,false)
	for t=1:p.Nt
		ψ[t] = abs2.(o1.data[t]*o2.data[t]')

	end
	ψ
end

function save_pair_density(ρ2,p,name)
	ts = Int.(floor.([1,2,p.Nt/4,p.Nt/2,3*p.Nt/4,p.Nt]))
	pls = []
	for t=1:length(ts)
		h = heatmap(real.(ρ2[ts[t]]),size=size_plots)
		push!(pls,h)
	end
	pl = plot(pls...)
	savefig(pl,string("plots/pair_density_",name,".png"))
end

############################## Examples of use

###################### PARAMETERS
# N is the number of bodies, Nx the number of spatial points, Nt the number of time points, L the physical spatial length, T the physical time length
N = 2; Nx = 60; Nt = 500; L = 10; T = 3 
w(x) = 10000*exp(-x^2 / (2*(0.1^2))) # Defines the interaction, W(x,y) = w(|x-y|) hence only the positive values of w matter. In reduced (toroidal) coordinates, that is x ∈ [0,1]
###################### END PARAMETERS

# Initialization of static objects
pS = ParamsN(N,Nx,L) # Creates parameters
init_N_Laplacian(pS) # Computes the N-body Laplacian matrix
init_Nb_interaction_diagonal(w,pS) # Computes the diagonal of the interaction matrix operator

# Initialization of dynamic objects
pD = ParamsNdyn(Nt,T,pS)
v = 2*from_fun_tx(7,pD) # Creates a potential
init_Nb_Laplacian_dyn(pD,pS) # computes the N-body Laplacian evolution operator matrix
init_Nb_interaction_dyn(pD,pS) # computes the diagonal of the N-body interaction evolution operator matrix

# Initial state
orbitals = create_N_orth_orbs(pS)
Ψ0 = antisym_from_orbs(orbitals,pS) # One-body state

# Evolution
Ψ = NbSchro(Ψ0,v,pD) # Computation of the evolution
test_antisym(Ψ,pD)
ρ = density_dyn(Ψ,pD) # Computation of the one-body density
ρ2 = pair_density_dyn(Ψ,pD)
save_pair_density(ρ2,pD,"ρ")

# Plots
pl = plot(plot(pD.x_axis,orbitals,title="Initial orbitals"),plot2d(v,pD,"Potential"),plot2d(ρ,pD,"ρ"),size=size_plots) # Plots the initial orbitals, the potential and the density
savefig(pl,"plots/main.png")

### 1 corps, former tests
# pS1 = ParamsN(1,Nx,L); init_N_Laplacian(pS1); pD1 = ParamsNdyn(Nt,T,pS1); init_Nb_Laplacian_dyn(pD1,pS1)
# Ψ01 = antisym_from_orbs([orb],pS1)
# Ψ1 = NbSchro(Ψ01,v,pD1)
# χ = product_orbs_dyn(Ψ1,Ψ1,pD1) 
# save_pair_density(χ,pD,"χ")
### end 1 corps
