function prof_li_hhl_minimal
% Minimal Prof. Li unstable-shear test: ONE frozen-velocity time slab.
% Outputs A*z=b and an HHL-compatible Hermitian embedding H*x=rhs.

%
% Unknowns at each time step j:
%   z_j = [Ustar_j (2M); p_j (M); U_j (2M)],  M=Nx*Ny.
% A frozen field V=U0 is held fixed for N steps.
% No fine reference, repeated slabs, N sweep, or plots.

%% 1. PARAMETERS
Nx = 5; Ny = 5;            % HHL toy test. For original grid use 21,21.
N = 2;                      % Frozen steps in ONE slab (start with 1 or 2).
alpha = 0.70;
rho = 1.0;
nu = 0.01;
epsilon = 0.01;
dt = 0.005;
Lx = 2*pi/alpha;
Ly = 2*pi;

assert(mod(Nx,2)==1 && mod(Ny,2)==1, ...
    'Use odd grid counts: centered periodic D*G has extra null modes on even grids.');
M = Nx*Ny;
q = 5*M;                   % Unknowns per time step
K = q*N;                   % Unknowns over N steps

%% 2. PERIODIC FINITE-DIFFERENCE OPERATORS
hx = Lx/Nx;
hy = Ly/Ny;
D1x = first_periodic(Nx,hx);
D1y = first_periodic(Ny,hy);
D2x = second_periodic(Nx,hx);
D2y = second_periodic(Ny,hy);
Dx = kron(speye(Ny),D1x);
Dy = kron(D1y,speye(Nx));
L = kron(speye(Ny),D2x) + kron(D2y,speye(Nx));
D = [Dx, Dy];               % Divergence: M x 2M
G = [Dx; Dy];               % Gradient: 2M x M

% Pressure equation uses exactly Lp=D*G (not the 5-point Laplacian L).
Ap = D*G;
Ap(1,:) = 0;               % Pressure gauge: p at first node = 0
Ap(1,1) = 1;
Dg = D;
Dg(1,:) = 0;               % Consistent gauge row on pressure RHS

%% 3. UNSTABLE-SHEAR INITIAL VELOCITY AND BODY FORCE
x = (0:Nx-1)'*hx;
y = (0:Ny-1)'*hy;
[X,Y] = ndgrid(x,y);
uBase = -sin(Y);
Ubase = [uBase(:); zeros(M,1)];
U0 = [uBase(:); (epsilon/alpha)*sin(alpha*X(:))];

% Discrete body force balances viscosity for steady base flow.
F = -nu * blkdiag(L,L) * Ubase;
F(1:M) = F(1:M) - mean(F(1:M));
F(M+1:end) = F(M+1:end) - mean(F(M+1:end));

%% 4. FREEZE CONVECTION AND FORM ONE-STEP PROPAGATOR
Vx = U0(1:M);
Vy = U0(M+1:end);
C = spdiags(Vx,0,M,M)*Dx + spdiags(Vy,0,M,M)*Dy;
S = speye(M) - dt*C + nu*dt*L;
B = blkdiag(S,S);          % Ustar_j = B*U_(j-1) + dt*F

%% 5. ASSEMBLE THE GLOBAL NON-HERMITIAN LINEAR SYSTEM A*z=b
% At each step:
%   Ustar_j - B*U_(j-1)             = dt*F
%   -(rho/dt)*Dg*Ustar_j + Ap*p_j  = 0
%   -Ustar_j + (dt/rho)*G*p_j + U_j = 0
I2 = speye(2*M);
A0 = [ I2,                 sparse(2*M,M),  sparse(2*M,2*M);
      -(rho/dt)*Dg,       Ap,             sparse(M,2*M);
      -I2,                (dt/rho)*G,     I2 ];

% Off-diagonal coupling: previous projected velocity U_(j-1).
A1 = sparse(q,q);
A1(1:2*M, 3*M+1:5*M) = -B;
shift = spdiags(ones(N,1),-1,N,N);
A = kron(speye(N),A0) + kron(shift,A1);

b = zeros(K,1);
for j = 1:N
    rows = (j-1)*q + (1:2*M);
    b(rows) = dt*F;
end
b(1:2*M) = b(1:2*M) + B*U0;

%% 6. CLASSICAL VALIDATION (NOT AN HHL COMPUTATION)
z_ref = A\b;
U = U0;
for j = 1:N
    Ustar = B*U + dt*F;
    p = Ap \ ((rho/dt)*Dg*Ustar);
    U = Ustar - (dt/rho)*G*p;
end
U_from_A = z_ref((N-1)*q + (3*M+1:5*M));
rel_error = norm(U_from_A-U)/max(norm(U),eps);
residual = norm(A*z_ref-b)/max(norm(b),eps);
fprintf('Grid %dx%d; N=%d; A=%dx%d; nnz(A)=%d\n',Nx,Ny,N,K,K,nnz(A));
fprintf('Relative difference from sequential solver: %.3e\n',rel_error);
fprintf('Relative linear-system residual: %.3e\n',residual);

%% 7. HHL INPUT: HERMITIAN EMBEDDING
% H0*[0;z] = [b;0] with H0=[0,A;A',0].
% Scaling both sides by the same factor does not change the solution.
H0 = [sparse(K,K), A; A', sparse(K,K)];
scale = norm(H0,1);          % Upper bound for spectral norm
H = H0/scale;
rhs = [b; zeros(K,1)]/scale;

% H is Hermitian but can be poorly conditioned; no HHL is run here.
hermitian_residual = norm(H*[zeros(K,1);z_ref]-rhs)/max(norm(rhs),eps);
fprintf('HHL Hermitian matrix: %dx%d; nnz(H)=%d\n',2*K,2*K,nnz(H));
fprintf('Hermitian embedding residual: %.3e\n',hermitian_residual);

%% 8. EXPORT to ../output (SciPy can open MATLAB -v7 with scipy.io.loadmat)
outputDir = fullfile(fileparts(mfilename('fullpath')),'..','output');
if ~exist(outputDir,'dir')
    mkdir(outputDir);
end
outFile = fullfile(outputDir,'prof_li_hhl_case.mat');
save(outFile,'A','b','H','rhs','z_ref','U0','U', ...
    'Nx','Ny','N','M','dt','rho','nu','alpha','epsilon','scale','-v7');
fprintf('Saved %s\n',outFile);
end

function D = first_periodic(n,h)
e = ones(n,1);
D = spdiags([-e,e],[-1,1],n,n)/(2*h);
D(1,n) = -1/(2*h);
D(n,1) = 1/(2*h);
end

function D2 = second_periodic(n,h)
e = ones(n,1);
D2 = spdiags([e,-2*e,e],[-1,0,1],n,n)/(h*h);
D2(1,n) = 1/(h*h);
D2(n,1) = 1/(h*h);
end
