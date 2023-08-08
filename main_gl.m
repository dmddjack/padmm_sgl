clear;
close all
seed = 30;
rng(seed);

% generate a graph
DIM = 20;
% [A,XCoords, YCoords] = construct_graph(DIM,'gaussian', 0.75, 0.5);
[A,XCoords, YCoords] = construct_graph(DIM,'er',seed , 0.2);
% [A,XCoords, YCoords] = construct_graph(DIM,'pa',1);
density_p = sum(sum(A>1e-5))/(DIM^2-DIM);
density_n = sum(sum(A<-1e-5))/(DIM^2-DIM);
disp(density_p);
disp(density_n);
% generate graph signals
NUM = 100;
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



lambda = 1000;
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
% Vp_tmp = [Vp(:,dim_of_null_space+1:DIM),Vp(:,1:dim_of_null_space)];
Vp = Vp * perm';
% assert(all(all(Vp==Vp_tmp)));
% fprintf("dim_of_null_space=%d\n",dim_of_null_space);
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

% gftcoeff = mvnrnd(mu,sigma,NUM);
% X = V*gftcoeff';

% Dp = Dp.*(Dp>0);
% Dn = Dn.*(Dn>0);
% sigmap = pinv(Dp);
% sigman = Dn;
% gftcoeffp = mvnrnd(mu,sigmap,NUM);
% gftcoeffn = mvnrnd(mu,sigman,NUM);
% X = Vn*gftcoeffn'+Vp*gftcoeffp';
noise = randn(size(X));
X_noisy = X + 0.1 * noise / norm(noise) * norm(X);



Z = zeros(DIM,DIM);
for i = 1 : DIM
    for j = 1 : DIM
       Z(i,j) = norm(X_noisy(i,:)-X_noisy(j,:),2)^2;
    end
end
disp(0.5 * sum(sum(Z.*A)));

%% common parameters
alphap = 160;
betap = 180;
alphan = 1;
betan = 1.5;


alpha = 1.8 % 1.8;
beta = 1 % 3;
gamma = -4.2 % -1.5;
max_iter = 1000;
epsilon = 1e-10;

%% obtain optimal solution via ADMM solver
% fprintf('solving...\n');
% t = 10;
% tau1 = 1e-4;
% tau2 = 1e-4;
% tic
% [W_opt] = gl_admm_solver(X_noisy, alpha, beta, t, tau1, tau2, 1e6, 1e-14);
% fprintf('solved!\n');
% toc
% load('W_opt.mat')

% %% CVX
tic
[W_cvx, ~] = gl_cvx(X_noisy, alpha, beta, gamma); % run algorithm
% [W_cvx, ~] = gl_cvx(X_noisy, alphap, alphan, betap, betan); % run algorithm
cvx_time = toc;
% S = (W_cvx>0.75)|(W_cvx<-0.75);
% W_cvx = S.*W_cvx;
D = diag(sum(full(W_cvx)));
L_cvx = D-full(W_cvx);
L_cvx(abs(L_cvx)<10^(-4))=0;
[precision_cvx_p,recall_cvx_p,f_cvx_p,precision_cvx_n,recall_cvx_n,f_cvx_n,~] = graph_learning_perf_eval(L0,L_cvx);
fval_cvx = 0.5*trace(W_cvx*Z); % + 0.5*beta*(norm(W_cvx,'fro'))^2;
disp(density_p);
disp(density_n);
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

%% ADMM
% t = 50;
% tau1 = 0.0005;
% tau2 = 0.02;
% tic
% [W_admm, fval_admm, primal_gap_iter_admm] = gl_admm(X_noisy, alpha, beta, t, tau1, tau2, max_iter, epsilon, W_opt);
% admm_time = toc;
% D = diag(sum(full(W_admm)));
% L_admm = D-full(W_admm);
% L_admm(abs(L_admm)<10^(-4))=0;
% [precision_admm, recall_admm, Fmeasure_admm, NMI_admm, num_of_edges_admm] = graph_learning_perf_eval(L0,L_admm);

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
fprintf('alpha=%.2f, beta=%.2f\n', alpha, beta);
fprintf('----- CVX  Time needed is %f -----\n', cvx_time);
% fprintf('----- PDS  Time needed is %f -----\n', stat_pds.time);
% fprintf('----- ADMM Time needed is %f -----\n', admm_time);
% fprintf('----- FDPG Time needed is %f -----\n', fdpg_time);
% fprintf('----- MM Time needed is %f -----\n', mm_time);

fprintf('CVX               | fval_cvx=%f\n', fval_cvx);
fprintf('CVX measurements  | precision_cvx_p=%f,recall_cvx_p=%f,f_cvx_p=%f\n                  | precision_cvx_n=%f,recall_cvx_n=%f,f_cvx_n=%f\n                  | f_cvx=%f\n\n' ...
    ,precision_cvx_p,recall_cvx_p,f_cvx_p,precision_cvx_n,recall_cvx_n,f_cvx_n,0.5*(f_cvx_p+f_cvx_n));

% fprintf('PDS               | fval_pds=%f \n', fval_pds);
% fprintf('PDS measurements  | Fmeasure_pds=%f, precision_pds=%f, recall_pds=%f, NMI_pds=%.4f\n\n', Fmeasure_pds, precision_pds, recall_pds, NMI_pds);
% 
% fprintf('ADMM              | fval_admm=%f, t=%f, tau1=%f, tau2=%f, max_iter=%d\n', fval_admm(end), t, tau1, tau2, max_iter);
% fprintf('ADMM measurements | Fmeasure_admm=%f, precision_admm=%f, recall_admm=%f, NMI_admm=%.4f\n\n', Fmeasure_admm, precision_admm, recall_admm, NMI_admm);
% 
% fprintf('FDPG              | fval_fdpg=%f, max_iter=%d\n', fval_fdpg(end), max_iter);
% fprintf('FDPG measurements | Fmeasure_fdpg=%f, precision_fdpg=%f, recall_fdpg=%f, NMI_fdpg=%.4f\n\n', Fmeasure_fdpg, precision_fdpg, recall_fdpg, NMI_fdpg);
% 
% fprintf('MM              | fval_mm=%f, max_iter=%d\n', fval_mm(end), max_iter);
% fprintf('MM measurements | Fmeasure_mm=%f, precision_mm=%f, recall_mm=%f, NMI_mm=%.4f\n\n', Fmeasure_mm, precision_mm, recall_mm, NMI_mm);
       
% fprintf('num of edges: %f, %f, %f, %f\n\n', num_of_edges_pds, num_of_edges_admm, num_of_edges_fdpg, num_of_edges_mm)
%% figures
% figure;
% semilogy(primal_gap_iter_admm,'-r','LineWidth',1.5);
% hold on;
% semilogy(primal_gap_iter_pds,'-b','LineWidth',1.5,'LineStyle',"--");
% hold on;
% semilogy(primal_gap_iter_fdpg,'-g','LineWidth',1.5,'LineStyle',"-.");
% hold on;
% semilogy(primal_gap_iter_mm,'-k','LineWidth',1.5,'LineStyle',":");
% hold on;
% xlabel('iteration $k$','Interpreter','latex','FontSize',23);
% ylabel('{$\|w-w^*\|_2$}','Interpreter','latex','FontSize',23);
% lgd = legend('pADMM-GL','Primal-Dual','FDPG','location','southwest');
% lgd.FontSize = 15;
% title('Static Graph Learning','Interpreter','latex','FontSize',20);

% figure;
% semilogy(fval_admm,'-r','LineWidth',1);
% hold on;
% semilogy(stat_pds.fgh_eval,'-b','LineWidth',1, 'LineStyle',"--");
% hold on;
% semilogy(fval_fdpg,'-g','LineWidth',1, 'LineStyle',"-.");
% hold on;
% % semilogy(fval_mm,'-k','LineWidth',1, 'LineStyle',":");
% % hold on;
% xlabel('iteration','FontSize',30);
% ylabel('{$f(w)$}','Interpreter','latex','FontSize',30);
% lgd = legend('ADMM','Primal-Dual','FDPG','location','southwest');
% lgd.FontSize = 20;

% figure;
% subplot(2,2,1)
% imagesc(L0)
% colorbar
% title('Groundtruth')
% subplot(2,2,2)
% imagesc(L_cvx)
% colorbar
% title('CVX')
% subplot(2,2,3)
% imagesc(L_pds)
% colorbar
% title('PDS')
% subplot(2,2,4)
% imagesc(L_admm)
% colorbar
% title('ADMM')