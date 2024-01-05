function [output] = main_gl(seed, DIM, NUM, opt)
addpath("D:\user\OneDrive - The Chinese University of Hong Kong\Document(OneDrive)\CUHK\2022-23 Summer Research\code\func")
addpath("D:\user\OneDrive - The Chinese University of Hong Kong\Document(OneDrive)\CUHK\2022-23 Summer Research\code\cvx")

% clear;
close all;
% seed = 30;
rng(seed);
cvx = true;
SGL = true;
% cvx_time = 0;
% generate a graph
% DIM = 100;
try
    switch opt
        case 'gaussian'
            [A, XCoords, YCoords] = construct_graph(DIM, 'gaussian', seed, 0.7, 0.5);
        case 'er'
            [A, XCoords, YCoords] = construct_graph(DIM, 'er', seed, 0.2);
        case 'pa'
            [A, XCoords, YCoords] = construct_graph(DIM, 'pa', seed, round(DIM*0.1));
        otherwise
            error('wrong opt');
    end
catch exception
    disp(exception.message);
    output = NaN;
    return;
end

% [A,XCoords, YCoords] = construct_graph(DIM,'gaussian', 0.75, 0.5);
% [A,XCoords, YCoords] = construct_graph(DIM, 'er', seed, 0.2);
% [A,XCoords, YCoords] = construct_graph(DIM,'pa',1);
density_p = sum(sum(A>1e-5))/(DIM^2-DIM);
density_n = sum(sum(A<-1e-5))/(DIM^2-DIM);
disp(density_p);
disp(density_n);
% generate graph signals
% NUM = 100;
Ap =  A.*(A>0);
Dp = diag(sum(full(Ap)));
Lp = Dp-full(Ap);
An = -A.*(A<0);
Dn = diag(sum(full(An)));
Ln = Dn-full(An);

% D = diag(sum(full(A)));
% L0 = D - full(A);
% [V, D] = eig(full(L0));
% Dp = D.*(D>0);
% Dn = -D.*(D<0);
% sigma = pinv(Dp)/norm(Dp,'fro') + Dn/norm(Dn,'fro');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Lp = Lp/trace(Lp)*DIM;
Ln = Ln/trace(Ln)*DIM;
mid = round(DIM / 2);
[Vp,Dp] = eig(full(Lp));
hp = zeros(size(Dp));
hp(Dp>1e-10) = 1./sqrt(Dp(Dp>1e-10));
% hp = exp(-lambda * Dp);
dim_of_null_space = sum(diag(hp)==0);
perm = [zeros(DIM-dim_of_null_space,dim_of_null_space),eye(DIM-dim_of_null_space);eye(dim_of_null_space),zeros(DIM-dim_of_null_space,dim_of_null_space)'];
hp = diag(perm * diag(hp));
Vp = Vp * perm';
hp(mid+1:DIM,mid+1:DIM) = 0;
hp = hp / norm(hp, 'fro');

[Vn,Dn] = eig(full(Ln));
hn = zeros(size(Dn));

hn(Dn>1e-10) = sqrt(Dn(Dn>1e-10));
% hn = exp(-lambda * Dn);
hn(1:mid,1:mid) = 0;
hn = hn / norm(hn, 'fro');
mu = zeros(1,DIM);
X = mvnrnd(mu, eye(DIM), NUM)';
X = 0.5 * (Vp*hp*Vp'+Vn*hn*Vn')*X;
L0 = Lp - Ln;

noise = randn(size(X));
X_noisy = X + 0.01 * noise / norm(noise, 'fro') * norm(X, 'fro');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Lp = Lp/trace(Lp)*DIM;
% Ln = Ln/trace(Ln)*DIM;
% mid = round(DIM / 2);
% [Vp,Dp] = eig(full(Lp));
% hp = pinv(Dp);
% hp = hp / norm(hp, 'fro');

% [Vn,Dn] = eig(full(Ln));
% hn = Dn / norm(Dn, 'fro');
% mu = zeros(1,DIM);
% X = mvnrnd(mu, eye(DIM), NUM)';
% X = (Vp*hp*Vp'+Vn*hn*Vn')*X;
% L0 = Lp - Ln;

% noise = randn(size(X));
% X_noisy = X + 0.1 * noise / norm(noise, 'fro') * norm(X, 'fro');


file_name = sprintf("data/X_s_%d.csv", seed);
writematrix(X_noisy,file_name);


Z = zeros(DIM,DIM);
for i = 1 : DIM
    for j = 1 : DIM
       Z(i,j) = norm(X_noisy(i,:)-X_noisy(j,:),2)^2;
    end
end
% disp(0.5 * sum(sum(Z.*A)));

%% common parameters

if DIM==100
    alpha = .19; 
    beta = .9;
    delta = -6; 
    rho = .05;
    epsilon = 2e-4;
elseif DIM==80
    alpha = .25; 
    beta = 1;
    delta = -4;
    rho = .08;
    epsilon = 5e-4;
elseif DIM==50
    alpha = .45; 
    beta = 1.5;
    delta = -3;
    rho = .1;
    epsilon = 1e-5;
elseif DIM==20
    alpha = 2.2; 
    beta = 3;
    delta = -1; 
    rho = .2;
    epsilon = 1e-6;
end
% epsilon = 1e-6;
max_iter = 1e5;


%% obtain optimal solution via ADMM solver
fprintf('solving...\n');

tau1 = 1/(rho*(sqrt(2*(DIM-1))+sqrt(DIM*(DIM-1)/2))^2);
tau2 = 1/rho;
% tic
max_try = 50;
alpha_opt = alpha;
beta_opt = beta;
delta_opt = delta;
[W_opt, density_p, density_n] = gl_admm_solver(X_noisy, alpha, beta, delta, rho, tau1, tau2, 1e6, 1e-13);
error_p = abs(density_p - .11);
error_n = abs(density_n - .11);
% D = diag(sum(full(W_opt)));
% L_opt = D-full(W_opt);
% L_opt(abs(L_opt)<10^(-4))=0;
% [precision_opt_p,recall_opt_p,f_opt_p,precision_opt_n,recall_opt_n,f_opt_n,~] = graph_learning_perf_eval(L0,L_opt);
% f = 0.5*(f_opt_p+f_opt_n);
dp_best = density_p;
dn_best = density_n;
for i = 1:max_try
    alpha = alpha_opt * (1 + 0.1*randn());
    beta = beta_opt * (1 + 0.1*randn());
    delta = delta_opt * (1 + 0.1*randn());
    [W_opt, density_p, density_n] = gl_admm_solver(X_noisy, alpha, beta, delta, rho, tau1, tau2, 1e6, 1e-13);
    % D = diag(sum(full(W_opt)));
    % L_opt = D-full(W_opt);
    % L_opt(abs(L_opt)<10^(-4))=0;
    % [precision_opt_p,recall_opt_p,f_opt_p,precision_opt_n,recall_opt_n,f_opt_n,~] = graph_learning_perf_eval(L0,L_opt);
    % f_new = 0.5*(f_opt_p+f_opt_n);

    error_new_p = abs(density_p - .11);
    error_new_n = abs(density_n - .11);
    if (error_new_p < error_p && error_new_n < error_n)
    % if (f_new > f)
        alpha_opt = alpha;
        beta_opt = beta;
        delta_opt = delta;
        error_p = error_new_p;
        error_n = error_new_n;
        % f = f_new;
        dp_best = density_p;
        dn_best = density_n;
    end
    if error_p < 0.002 && error_n < 0.002
        break
    end
end
alpha = alpha_opt;
beta = beta_opt;
delta = delta_opt;

disp(dp_best);
disp(dn_best);
fprintf('solved!\n');
% toc
% load('W_opt.mat')

%% CVX
if cvx
    tic
    [W_cvx, ~] = gl_cvx(X_noisy, alpha, beta, delta); % run algorithm
    % [W_cvx, ~] = gl_cvx(X_noisy, alphap, alphan, betap, betan); % run algorithm
    cvx_time = toc;
    % S = (W_cvx>0.75)|(W_cvx<-0.75);
    % W_cvx = S.*W_cvx;
    D = diag(sum(full(W_cvx)));
    L_cvx = D-full(W_cvx);
    L_cvx(abs(L_cvx)<10^(-4))=0;
    [precision_cvx_p,recall_cvx_p,f_cvx_p,precision_cvx_n,recall_cvx_n,f_cvx_n,~] = graph_learning_perf_eval(L0,L_cvx);
    fval_cvx = 0.5*trace(W_cvx*Z); % + 0.5*beta*(norm(W_cvx,'fro'))^2;
end
% disp(density_p);
% disp(density_n);
%% PDS
% tau = 0.9;
% Z0 = 1/sqrt(alpha*beta)*Z;
% [W_pds, stat_pds, primal_gap_iter_pds] = gsp_learn_graph_log_degrees_(Z0, 1, 1, alpha, beta, tau, max_iter, epsilon, W_opt);
% W_pds = sqrt(alpha/beta)*W_pds;
% 
% D = diag(sum(full(W_pds)));
% L_pds = D-full(W_pds);
% L_pds(abs(L_pds)<10^(-4))=0;
% fval_pds = trace(W_pds*Z) - alpha*sum(log(sum(W_pds,2))) + 0.5*beta*(norm(W_pds,'fro'))^2;
% [precision_pds, recall_pds, Fmeasure_pds, NMI_pds, num_of_edges_pds] = graph_learning_perf_eval(L0,L_pds);

%% SGL
if SGL
    if pyenv().Version == ""
        pyenv(Version="C:\ProgramData\Anaconda3\envs\dynSGL\python.exe",ExecutionMode="InProcess");
    end
    command = sprintf("SGL.py -s %d", seed);
    SGL_time  = pyrunfile(command,'toc');
    file_name = sprintf("data/W_SGL_%d.csv", seed);
    W_SGL = readmatrix(file_name);    
    D = diag(sum(full(W_SGL)));
    L_SGL = D-full(W_SGL);
    L_SGL(abs(L_SGL)<10^(-4))=0;
    
    [precision_SGL_p,recall_SGL_p,f_SGL_p,precision_SGL_n,recall_SGL_n,f_SGL_n,~] = graph_learning_perf_eval(L0,L_SGL);
end
%% ADMM
% rho = .01;
% tau1 = 1/(rho*5148);
% tau2 = 1/rho;
tic
[W_admm, fval_admm, primal_gap_iter_admm] = gl_admm(X_noisy, alpha, beta, delta, rho, tau1, tau2, max_iter, epsilon, W_opt);
admm_time = toc;
D = diag(sum(full(W_admm)));
L_admm = D-full(W_admm);
L_admm(abs(L_admm)<10^(-4))=0;
[precision_admm_p,recall_admm_p,f_admm_p,precision_admm_n,recall_admm_n,f_admm_n,~] = graph_learning_perf_eval(L0,L_admm);
%% FDPG
% reset = 50;
% tic
% [W_FDPG, fval_fdpg, primal_gap_iter_fdpg] = gl_fdpg(X_noisy, alpha, beta, max_iter, reset, W_opt);
% fdpg_time = toc;
% D = diag(sum(full(W_FDPG)));
% L_fdpg = D-full(W_FDPG);
% L_fdpg(abs(L_fdpg)<10^(-4))=0;
% [precision_fdpg, recall_fdpg, Fmeasure_fdpg, NMI_fdpg, num_of_edges_fdpg] = graph_learning_perf_eval(L0,L_fdpg);

%% MM
% tic
% DIMw = DIM*(DIM-1)/2;
% w_0 = ones(DIMw,1);
% [w_mm, stat_mm, fval_mm, primal_gap_iter_mm] = MM_gl(X_noisy, alpha, beta, w_0, epsilon, NUM, max_iter, W_opt);
% mm_time = toc;
% W_mm  = linear_operator_vec2mat(w_mm, DIM);
% D = diag(sum(full(W_mm)));
% L_mm = D-full(W_mm);
% L_mm(abs(L_mm)<10^(-4))=0;
% [precision_mm, recall_mm, Fmeasure_mm, NMI_mm, num_of_edges_mm] = graph_learning_perf_eval(L0,L_mm);

%% outputs
% fprintf('alphap=%.2f, betap=%.2f\n', alphap, betap);
% fprintf('alphan=%.2f, betan=%.2f\n', alphan, betan);
fprintf('seed=%d, alpha=%.2f, beta=%.2f, delta=%.2f, rho=%.2f\n', seed, alpha, beta, delta, rho);
if cvx
    fprintf('----- CVX  Time needed is %f -----\n', cvx_time);
end
if SGL
    fprintf('----- SGL  Time needed is %f -----\n', SGL_time);
end
fprintf('----- ADMM Time needed is %f -----\n', admm_time);

if cvx
    f_cvx = 0.5*(f_cvx_p+f_cvx_n);
    fprintf('CVX               | fval_cvx=%f\n', fval_cvx);
    fprintf('CVX measurements  | precision_cvx_p=%f,recall_cvx_p=%f,f_cvx_p=%f\n                  | precision_cvx_n=%f,recall_cvx_n=%f,f_cvx_n=%f\n                  | f_cvx=%f\n\n' ...
       ,precision_cvx_p,recall_cvx_p,f_cvx_p,precision_cvx_n,recall_cvx_n,f_cvx_n,f_cvx);
end
if SGL
    f_SGL = 0.5*(f_SGL_p+f_SGL_n);
    % fprintf('SGL               | fval_SGL=%f\n', fval_SGL);
    fprintf('SGL measurements  | precision_SGL_p=%f,recall_SGL_p=%f,f_SGL_p=%f\n                  | precision_SGL_n=%f,recall_SGL_n=%f,f_SGL_n=%f\n                  | f_SGL=%f\n\n' ...
        ,precision_SGL_p,recall_SGL_p,f_SGL_p,precision_SGL_n,recall_SGL_n,f_SGL_n,f_SGL);
end
f_admm = 0.5*(f_admm_p+f_admm_n);
fprintf('ADMM               | fval_admm=%f\n', fval_admm(end));
fprintf('ADMM measurements  | precision_admm_p=%f,recall_admm_p=%f,f_admm_p=%f\n                  | precision_admm_n=%f,recall_admm_n=%f,f_admm_n=%f\n                  | f_admm=%f\n\n' ...
    ,precision_admm_p,recall_admm_p,f_admm_p,precision_admm_n,recall_admm_n,f_admm_n,f_admm);

output = [cvx_time,SGL_time,admm_time,f_SGL_p,f_SGL_n,f_SGL,f_admm_p,f_admm_n,f_admm];
%% figures

% figure;
% semilogy(primal_gap_iter_admm,'-r','LineWidth',1.5);
% xlabel('iteration $k$','Interpreter','latex','FontSize',20);
% ylabel('{$\|w^k-w^*\|_2$}','Interpreter','latex','FontSize',20);
% lgd = legend('pADMM-SGL','location','northeast');
% lgd.FontSize = 14;
% beep on; beep;
