function prof_li_hhl_reduced_multislab
% PROF_LI_HHL_REDUCED_MULTISLAB
% Demonstrates the two-stage reduction of Prof. Li's frozen-velocity
% Navier-Stokes matrix and rebuilding the matrix after each HHL solve.
%
%   1. U_j = Z*a_j, where D*Z=0 and Z'*Z=I.
%   2. Ared*a = bred, Ared = I - kron(S_N,K), K=Z'*B(V)*Z.
%   3. P5 = I+X+X^2+X^3+X^4, X=kron(S_N,K).
%      Solve A5*a=b5, A5=I-X^5, b5=P5*bred.
%   4. Recover all physical velocities, pressures and intermediate
%      velocities; use LAST velocity to rebuild B(V), K, A5 next slab.
%
% No quantum library is called. Classical A5\b5 is a VERIFIED placeholder
% for the student's HHL platform. Files exported in MATLAB -v7 format
% are readable by scipy.io.loadmat.
%
% IMPORTANT: HHL outputs a NORMALIZED solution state. To propagate between
% slabs, the student must recover a correctly scaled classical velocity
% vector. A state vector with unknown amplitude is insufficient.

%% ----------------------- USER SETTINGS -------------------------------
Nx = 5; Ny = 5;         % Initial HHL test. Change both to 21 later.
Nfreeze = 10;           % Frozen steps per solve (>=6 for nontrivial A5).
totalSteps = 20;        % Two slabs: solve, reconstruct, REBUILD, solve.
alpha = 0.70;
rho = 1.0;
nu = 0.01;
epsilon = 0.01;
dt = 0.005;
powerK = 5;            % P5; keep as 5 for our previous transformation.
useHHLadapter = false; % Change to true after implementing stub below.
exportMatrices = true; % One .mat file per slab, for Python/HHL.
verifyFullMatrix = true; % Full original system check ONLY on small grids.
outputDir = fullfile(fileparts(mfilename('fullpath')),'..','output','hhl_reduced_cases');

assert(mod(Nx,2)==1 && mod(Ny,2)==1, ...
    'Use odd Nx,Ny: centered periodic D*G has extra null modes for even grids.');
assert(Nfreeze >= 1 && totalSteps >= 1 && powerK >= 1);
if Nfreeze <= powerK
    warning(['With Nfreeze <= %d, X^%d=0 and A%d=I; ' ...
             'this is a TRIVIAL HHL test. Use Nfreeze>%d.'], ...
             powerK,powerK,powerK,powerK);
end
if ~exist(outputDir,'dir')   % summary.mat is always written
    mkdir(outputDir);
end
M = Nx*Ny;
q = 5*M;
Lx = 2*pi/alpha; Ly = 2*pi;
dx = Lx/Nx; dy = Ly/Ny;

%% --------------------- SPACE OPERATORS ------------------------------
Dx1 = periodic_first(Nx,dx);
Dy1 = periodic_first(Ny,dy);
Dxx1 = periodic_second(Nx,dx);
Dyy1 = periodic_second(Ny,dy);
Dx = kron(speye(Ny),Dx1);
Dy = kron(Dy1,speye(Nx));
L = kron(speye(Ny),Dxx1)+kron(Dyy1,speye(Nx));
D = [Dx Dy];             % M-by-2M: divergence
G = [Dx;Dy];             % 2M-by-M: pressure gradient
Ap = D*G;                % PRESSURE Laplacian, not 5-point L
Ap(1,:) = 0; Ap(1,1) = 1;
Dg = D; Dg(1,:) = 0;     % Must be modified in SAME row as Ap
I2 = speye(2*M);

% Compute divergence-free orthonormal basis ONCE (geometry is unchanged).
% null(full(D)) is deliberately simple; suitable for small demonstration
% grids, but scales poorly. It is NOT an efficient quantum encoding.
Z = null(full(D));
r = size(Z,2);
assert(norm(D*Z,'fro')/max(norm(D,'fro'),eps)<1e-10, ...
    'Z fails divergence-free check.');
assert(norm(Z'*Z-eye(r),'fro')/max(sqrt(r),eps)<1e-10, ...
    'Z is not orthonormal.');
fprintf('Grid %d x %d: M=%d, rank(D)=%d, reduced velocity DOF=%d\n', ...
    Nx,Ny,M,2*M-r,r);

%% -------------- UNSTABLE SHEAR + DISCRETE BODY FORCE ----------------
[xg,yg] = ndgrid((0:Nx-1)'*dx,(0:Ny-1)'*dy);
uBase = -sin(yg);
Ubase = [uBase(:);zeros(M,1)];
U0 = [uBase(:);(epsilon/alpha)*sin(alpha*xg(:))];
F = -nu*blkdiag(L,L)*Ubase;
F(1:M)=F(1:M)-mean(F(1:M));
F(M+1:end)=F(M+1:end)-mean(F(M+1:end));
f = dt*F;               % Known constant contribution to Ustar

% Initial field must lie in ker(D); reconstruct to remove roundoff.
a_in = Z'*U0;
U_in = Z*a_in;
assert(norm(U_in-U0)/max(norm(U0),eps)<1e-9, ...
    'Initial velocity is not discretely divergence-free.');

Uhistory = zeros(2*M,totalSteps+1);
Uhistory(:,1) = U_in;
Phistory = zeros(M,totalSteps);
residualReduced = [];
residualFull = [];
recoveryError = [];
kappaRed = [];
kappa5 = [];

%% ------------------ ONE HHL SOLVE PER FROZEN SLAB --------------------
nDone = 0;
slab = 0;
while nDone < totalSteps
    slab = slab + 1;
    n = min(Nfreeze,totalSteps-nDone);  % Steps in this slab
    V = U_in;                          % Freeze THIS velocity for n steps

    % REBUILD the convection operator and propagator each slab.
    C = spdiags(V(1:M),0,M,M)*Dx + spdiags(V(M+1:end),0,M,M)*Dy;
    Svel = speye(M) - dt*C + nu*dt*L;
    B = blkdiag(Svel,Svel);

    % Spatial reduction: U=Z*a, so a_j=K*a_(j-1)+g.
    K = Z'*(B*Z);
    g = Z'*f;
    bred = repmat(g,n,1);
    bred(1:r) = K*a_in + g;

    % Ared = I-X, X = S_n (subdiagonal shift) kron K.
    S_n = spdiags(ones(n,1),-1,n,n);
    X = kron(S_n,sparse(K));
    nr = n*r;
    Ared = speye(nr)-X;

    % LEFT transformation P5:
    % (I+X+...+X^4) (I-X) = I-X^5.
    % Transform RHS too. Construct without dense P5.
    A5 = speye(nr) - kron(S_n^powerK,sparse(K^powerK));
    b5 = bred;
    term = bred;
    for k = 1:powerK-1
        term = X*term;
        b5 = b5 + term;
    end

    % Hermitian embedding for HHL: H*[0;a]=[b5;0].
    H0 = [sparse(nr,nr),A5;A5',sparse(nr,nr)];
    scale = norm(H0,1);     % Bound ||H0||_2, for spectral normalization.
    H = H0/scale;
    rhsH = [b5;zeros(nr,1)]/scale;

    % This is the REFERENCE for checking the HHL solution.
    a_classical = A5\b5;

    if exportMatrices
        filename = fullfile(outputDir,sprintf('slab_%03d.mat',slab));
        save(filename,'Nx','Ny','M','n','r','Nfreeze','powerK', ...
            'dt','rho','nu','epsilon','alpha','V','a_in','Z','K','g', ...
            'Ared','bred','A5','b5','H','rhsH','scale', ...
            'a_classical','-v7');
    end

    % ----- Replace this ONE solve with student HHL output -----
    if useHHLadapter
        a = student_hhl_adapter(H,rhsH,A5,b5,a_classical,slab);
    else
        a = a_classical;
    end
    % ----------------------------------------------------------

    a = a(:);
    assert(numel(a)==nr,'HHL solver must return n*r reconstructed amplitudes.');
    rred = norm(Ared*a-bred)/max(norm(bred),eps);
    r5 = norm(A5*a-b5)/max(norm(b5),eps);
    errA = norm(a-a_classical)/max(norm(a_classical),eps);
    residualReduced(end+1,1) = r5; %#ok<AGROW>

    % Recover every physical state from a_j, and the full [Ustar;q;U].
    % This recovery is also a physical consistency test (pressure gauge).
    full_z = zeros(q*n,1);
    Uprev = U_in;
    errRecover = 0;
    for j = 1:n
        idx = (j-1)*r+(1:r);
        Uj = Z*a(idx);                % Corrected velocity U_j
        Ustar = B*Uprev+f;           % Intermediate velocity
        qj = Ap\(Dg*Ustar);          % q_j = (dt/rho)*p_j
        pj = (rho/dt)*qj;            % Physical pressure
        Uproject = Ustar-G*qj;       % Independent Chorin reconstruction

        errRecover = max(errRecover, ...
           norm(Uj-Uproject)/max(norm(Uproject),eps));
        idxfull = (j-1)*q+(1:q);
        full_z(idxfull) = [Ustar;pj;Uj];

        Uhistory(:,nDone+j+1)=Uj;
        Phistory(:,nDone+j)=pj;
        Uprev = Uj;
    end
    recoveryError(end+1,1) = errRecover; %#ok<AGROW>

    % Optional ORIGINAL 5M-by-5M block matrix check, FIRST SLAB only.
    fullResidual = NaN;
    fullDiff = NaN;
    if verifyFullMatrix && slab == 1 && q*n<=1500
        [Afull,bfull] = build_original_full(B,f,Ap,Dg,G,rho,dt,M,n,U_in);
        fullResidual = norm(Afull*full_z-bfull)/max(norm(bfull),eps);
        zfull_ref = Afull\bfull;
        fullDiff = norm(zfull_ref-full_z)/max(norm(zfull_ref),eps);
    end
    residualFull(end+1,1)=fullResidual; %#ok<AGROW>

    % Exact 2-norm condition numbers feasible ONLY on small matrices.
    condRed = NaN; condA5 = NaN;
    if nr <= 350
        condRed = cond(full(Ared));
        condA5 = cond(full(A5));
    end
    kappaRed(end+1,1) = condRed; %#ok<AGROW>
    kappa5(end+1,1) = condA5; %#ok<AGROW>

    fprintf(['Slab %d: steps %d-%d | full %d, reduced %d | ',...
             'kappa(Ared)=%.4g, kappa(A5)=%.4g\n'], ...
             slab,nDone+1,nDone+n,q*n,nr,condRed,condA5);
    fprintf(['  transformed residual %.3e, Ared residual %.3e, ',...
             'HHL/classical difference %.3e\n'],r5,rred,errA);
    fprintf(['  recovered projection mismatch %.3e; ',...
             'full residual %.3e; full solution mismatch %.3e\n'], ...
             errRecover,fullResidual,fullDiff);

    % KEY REBUILD STEP: updated physical U starts the NEXT frozen block.
    U_in = Uprev;
    a_in = Z'*U_in;
    nDone = nDone+n;
end

t = (0:totalSteps)*dt;
save(fullfile(outputDir,'summary.mat'),'Uhistory','Phistory','t', ...
    'residualReduced','residualFull','recoveryError', ...
    'kappaRed','kappa5','Nx','Ny','Nfreeze','totalSteps','-v7');
fprintf('DONE. Updated B(V), K, Ared, A5 after every solved slab.\n');
fprintf('All physical states and slab HHL matrices exported to %s\n',outputDir);

end

function a = student_hhl_adapter(H,rhsH,A5,b5,a_ref,slab) %#ok<INUSD>
% TO BE REPLACED BY THE STUDENT:
% 1. Send H, rhsH to HHL (Hermitian system).
% 2. HHL should return the direction of [0;a], NOT [a;0].
% 3. Recover its physical amplitude and global phase/sign so that
%    A5*a approximately equals b5. A normalized state is NOT enough.
% 4. Return a (length size(A5,1)) in correct MATLAB ordering.
% For classical prototyping, an approximate direction a_unit may be
% rescaled with t=( (A5*a_unit)'*b5 )/||A5*a_unit||^2.
% This uses a FULL classical vector and is only a verification technique,
% NOT an end-to-end quantum advantage.
error('Implement student_hhl_adapter for slab %d.',slab);
end

function [A,b] = build_original_full(B,f,Ap,Dg,G,rho,dt,M,n,U0)
% Exactly matches the original full per-step unknowns [Ustar;p;U].
q = 5*M;
I2 = speye(2*M);
A0 = [I2,sparse(2*M,M),sparse(2*M,2*M); ...
      -(rho/dt)*Dg,Ap,sparse(M,2*M); ...
      -I2,(dt/rho)*G,I2];
A1 = sparse(q,q);
A1(1:2*M,3*M+1:5*M)=-B;
S = spdiags(ones(n,1),-1,n,n);
A = kron(speye(n),A0)+kron(S,A1);
b = zeros(q*n,1);
for j=1:n
    b((j-1)*q+(1:2*M))=f;
end
b(1:2*M)=b(1:2*M)+B*U0;
end

function D = periodic_first(n,h)
e=ones(n,1);
D=spdiags([-e,e],[-1,1],n,n)/(2*h);
D(1,n)=-1/(2*h);
D(n,1)= 1/(2*h);
end

function D2 = periodic_second(n,h)
e=ones(n,1);
D2=spdiags([e,-2*e,e],[-1,0,1],n,n)/h^2;
D2(1,n)=1/h^2;
D2(n,1)=1/h^2;
end
