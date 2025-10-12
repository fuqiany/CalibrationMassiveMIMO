clear all
clc
close all
% This function is for demo of truncated BGPC algorithm in [1].
% signal model: Y = diag(lambda)*A*X, -> diag(gamma)*Y = A*X, Y:n*N, A:n*m, X:m*N
% the algorithm performs good with Gaussian random matrix A, but performs badly with DFT matrix
% [1] Yanjun Li, Kiryung Lee, and Yoranm Bresler, ''Blind gain and phase calibration via sparse spetral method'', IEEE Trans. Inf. Theory, vol 65, no. 5, May 2019.

mode = 0;           % 0: A is complex Gaussian randn matrix; 1: A is DFT matrix
n = 128;            % the number of antennas
m = 256;            % the number of atoms
N = 16;             % the number of samples
sparsity = 8;       % sparsity level
sigmaW = 0.01;      % noise variance

alpha = sqrt(n);    % tuning parameters

if 0 == mode
    A = sqrt(1/n)/sqrt(2)*(randn(n,m) + 1j*randn(n,m));
else
    theta = 0:2*pi/(m-1):2*pi;
    fsteer = @(theata)exp(-1j*2*pi*cos(theata)*1*(0:1:n-1)'/2)/sqrt(n); % steer vector, M x 1
    for k = 1:length(theta)
        A(:,k) = fsteer(theta(k));
    end
end



psi_1 = 2*pi*rand(n,1);
psi_2 = 2*pi*rand(n,1);

delta = 0.1;
lambda  = exp(1j*psi_1).*(1+(sqrt(1 + delta) -1)*exp(1j*psi_2));
gamma = 1./lambda;
% scatter(real(lambda),imag(lambda));


% generate X with CN(0,1/(N*s))
Xtemp = sqrt(1/N/sparsity)/sqrt(2)*(randn(sparsity,N) + 1j*randn(sparsity,N));
X = zeros(m,N);
for k = 1:N
    index = randperm(m,sparsity);
    X(index,k) = Xtemp(:,k);
end
norm(X,'fro')

W = sqrt(sigmaW/N/n)/sqrt(2)*(randn(n,N) + 1j*randn(n,N));
sigmaW_est = mean(mean(abs(W).^2))

Y = diag(lambda)*A*X + W;

MSNR = 20*log10(norm(diag(lambda)*A*X,'fro')/norm(W,'fro'))


% algorithm
eta_true = [X(:);-gamma/alpha]; % N*m + n

D = zeros(N*n, N*m);
I_N = eye(N);
for k = 1:n
    D(N*(k-1)+1:N*k,:) = kron(I_N,A(k,:));
end

E = zeros(N*n,n);

for k = 1:n
    E(N*(k-1)+1:N*k,k) = Y(k,:).';
end

temp = [D alpha*E];
B = temp'*temp;
abs(mean(B*eta_true))

beta = norm(B);
G = beta*eye(m*N+n) - B;

% power iteration for BGPC
eta_est = zeros(m*N+n,1);
eta_est(end-n+1:end) = exp(-1j*psi_1);
iter = 0;
Iter_Max = 100;
THRESHOLD = 1e-10;
norm1 = norm(eta_est);
thre = 1;
while((iter < Iter_Max)&&(thre > THRESHOLD))
    eta_est = G*eta_est./norm(G*eta_est);

    % sparsity projection
    for k = 1:N
        eta_est(m*(k-1)+1:m*k) = fun_sparse_proj(eta_est(m*(k-1)+1:m*k),sparsity);
    end

    norm2 = norm(G*eta_est);
    thre = abs(norm2 - norm1);
    norm1 = norm2;
    iter = iter + 1;
    mse(iter) = thre;
end
iter
thre

% results
eta_true = eta_true./norm(eta_true);
eta_est = eta_est./norm(eta_est);
RSNR = -10*log10(2-2*abs(eta_true'*eta_est))

lambda_est = eta_est(end-n+1:end);
subplot(2,2,1);
plot(abs(lambda_est./lambda_est(1))); hold on
plot(abs(gamma./gamma(1)),'--');
legend('est','true');
subplot(2,2,2);
plot(angle(lambda_est));hold on
plot(angle(gamma),'--');
legend('est','true');

subplot(2,2,3);
stem(abs(eta_est(1:end-n))); hold on;
stem(abs(eta_true(1:end-n)),'--');



% only s largest elements are nonzero
function x = fun_sparse_proj(x,s)
    abs_x = abs(x);
    sorted_x = sort(abs_x,'descend');
    threshold = sorted_x(s);
%     index = find(abs_x < threshold);
    x(abs_x < threshold) = 0;
end

