%% README
% Construct graph: function construct_tvg
% Generate signal: function generate_graph_signals
% ADMM solver: function dgl_admm

clear;
close all
seed = 30;
rng(seed);
cvx = 0;
dynSGL = 0;
%% common parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
alpha = 1.8;
beta = 1;
gamma = 5.6;
delta = -7;
% m=50
% alpha = .4;
% beta = 1;
% gamma = .41;
% delta = -4.2;
fprintf('alpha=%.3f, beta=%.3f, gamma=%.3f, delta=%.3f\n', alpha, beta, gamma, delta);

%% generate a graph %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% synthetic data
DIM = 20;
NUM = 100;
time_slots = 10;
p_connect = 0.2;
p_resample = 0.05;
[A,XCoords, YCoords] = construct_tvg(DIM,'tver',seed, p_connect,time_slots,p_resample);
L0 = cell(time_slots, 1);
[X_noisy, L0{1}] = generate_graph_signals(NUM,A{1},DIM,randi(2^16));
similarity_ground_truth = 0;
for t=2:time_slots
    [X_new, L0{t}] = generate_graph_signals(NUM,A{t},DIM,randi(2^16));
    X_noisy = cat(2,X_noisy,X_new);
    corr = corrcoef(squareform(A{t-1}),squareform(A{t}));
    similarity_ground_truth = similarity_ground_truth + corr(1,2);
end
similarity_ground_truth = similarity_ground_truth / (time_slots - 1);
file_name = sprintf("X_d_%d_%d.csv", time_slots, seed);
writematrix(X_noisy,file_name);



% 
% % % real data (the users need to specify the number of time slots)
% down_sample = 100; % randomly select a point every down_sample points (determines the number of graph nodes)
% NUM_frames = 100; % overall number of graph signals (determines the number of graph signals within each time slot)
% time_slots = 10; 
% X_noisy = tvg_realdata('Data/dance_mesh', down_sample, NUM_frames);
% X_noisy = (X_noisy-min(X_noisy)) ./ (max(X_noisy)-min(X_noisy)); % normalizing each column
% DIM = size(X_noisy,1);
% NUM = floor(size(X_noisy,2)/time_slots); % add floor by wxl
%% CVX %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if cvx
    tic
    [W_cvx, w_cvx, fval_cvx] = dgl_cvx(X_noisy, alpha, beta, gamma, delta, time_slots); % run algorithm
    cvx_time = toc;


    [precision_cvx_p,recall_cvx_p,f_cvx_p,precision_cvx_n,recall_cvx_n,f_cvx_n] = dgl_perf_eval(L0,W_cvx, time_slots);
    
    
    % fval_cvx = 0.5*trace(W_cvx*Z); % + 0.5*beta*(norm(W_cvx,'fro'))^2;
    % disp(density_p);
    % disp(density_n);
end


%% obtain optimal solution via ADMM %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rho = 1;
tau1 = 1/(rho*500);
tau2 = 1/rho;
max_iter_opt = 1e6;
epsilon_opt = 1e-13;
fprintf('solving...\n');
tic
[W_opt, w_opt] = dgl_admm_solver(X_noisy, alpha, beta, gamma, delta, rho, tau1, tau2, max_iter_opt, epsilon_opt, time_slots);
toc
fprintf('optimal solution obtained\n');

%% for comparing ADMM & PDS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
max_iter = 1e6;
epsilon = 1e-10;

%% ADMM %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% rho = .5;
% tau1 = 1e-2;
% tau2 = 2;
tic
[W_admm, fval_admm, fval_admm_iter, primal_gap_iter_admm] = dgl_admm(X_noisy, alpha, beta, gamma, delta, rho, tau1, tau2, max_iter, epsilon, w_opt, time_slots);
admm_time = toc;
[precision_admm_p,recall_admm_p,f_admm_p,precision_admm_n,recall_admm_n,f_admm_n] = dgl_perf_eval(L0, W_admm, time_slots);

%% dynSGL %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if dynSGL
    if pyenv().Version == ""
        pyenv(Version="C:\Users\dmddj.PRO13\.conda\envs\dynSGL\python.exe",ExecutionMode="OutOfProcess");
    end
    command = sprintf("dynSGL.py -t %d -s %d", time_slots, seed);
    dynSGL_time  = pyrunfile(command,'toc');
    file_name = sprintf("W_dynSGL_%d_%d.csv", time_slots, seed);
    data = readmatrix(file_name);    
    W_dynSGL = cell(time_slots,1);
    for t=1:time_slots
        W_dynSGL{t} = data(:,1+(t-1)*DIM:t*DIM);
        % w_t = squareform(W_dynSGL{t});
        % density_t = nnz(w_t)/max(size(w_t));
        % disp(density_t)
    end
    [precision_dynSGL_p,recall_dynSGL_p,f_dynSGL_p,precision_dynSGL_n,recall_dynSGL_n,f_dynSGL_n] = dgl_perf_eval(L0, W_dynSGL, time_slots);
end

%% primal-dual %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% params.maxit = max_iter;
% params.tol = epsilon;
% params.step_size = 0.95;
% % distance
% T = time_slots;
% Z = zeros(DIM,DIM);
% for i = 1 : DIM
%     for j = 1 : DIM
%         Z(i,j) = norm(X_noisy(i,1:NUM)-X_noisy(j,1:NUM),2)^2;
%     end
% end
% z_all = squareform(Z)';
% 
% for k = 1:T-1
%     Z = zeros(DIM,DIM);
%     for i = 1 : DIM
%         for j = 1 : DIM
%             Z(i,j) = norm(X_noisy(i,1+k*NUM:(k+1)*NUM)-X_noisy(j,1+k*NUM:(k+1)*NUM),2)^2;
%         end
%     end
%     z_new = squareform(Z)';
%     z_all = cat(2,z_all,z_new);
% end
% 
% tic
% [W_pds, fval_pds, primal_gap_iter_pds, ~] = tvglearn_fusedlasso(z_all, alpha, beta, gamma, w_opt, params);
% pds_time = toc;
% 
% fprintf('----- PDS Time needed is %f -----\n', pds_time);
% fprintf('PDS | fval_pds=%f, max_iter=%d\n', fval_pds, max_iter);

%% outputs %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('similarity_ground_truth = %f\n', similarity_ground_truth);
fprintf('alpha=%.3f, beta=%.3f, gamma=%.3f, delta=%.3f\n', alpha, beta, gamma, delta);
if cvx
    fprintf('----- CVX  Time needed is %f -----\n', cvx_time);
    % fprintf('CVX               | fval_cvx=%f\n', fval_cvx);
    fprintf('CVX measurements  | precision_cvx_p=%f,recall_cvx_p=%f,f_cvx_p=%f\n                  | precision_cvx_n=%f,recall_cvx_n=%f,f_cvx_n=%f\n                  | f_cvx=%f\n\n' ...
        ,precision_cvx_p,recall_cvx_p,f_cvx_p,precision_cvx_n,recall_cvx_n,f_cvx_n,0.5*(f_cvx_p+f_cvx_n));
end
if dynSGL
    fprintf('----- dynSGL Time needed is %f -----\n', dynSGL_time);
    % fprintf('dynSGL | fval_dynSGL=%f, t=%f, tau1=%f, tau2=%f, max_iter=%d\n', fval_dynSGL, rho, tau1, tau2, max_iter);
    fprintf('dynSGL measurements  | precision_dynSGL_p=%f,recall_dynSGL_p=%f,f_dynSGL_p=%f\n                  | precision_dynSGL_n=%f,recall_dynSGL_n=%f,f_dynSGL_n=%f\n                  | f_dynSGL=%f\n\n' ...
        ,precision_dynSGL_p,recall_dynSGL_p,f_dynSGL_p,precision_dynSGL_n,recall_dynSGL_n,f_dynSGL_n,0.5*(f_dynSGL_p+f_dynSGL_n));
end
fprintf('----- ADMM Time needed is %f -----\n', admm_time);
fprintf('ADMM | fval_admm=%f, t=%f, tau1=%f, tau2=%f, max_iter=%d\n', fval_admm, rho, tau1, tau2, max_iter);
fprintf('ADMM measurements  | precision_admm_p=%f,recall_admm_p=%f,f_admm_p=%f\n                  | precision_admm_n=%f,recall_admm_n=%f,f_admm_n=%f\n                  | f_admm=%f\n\n' ...
    ,precision_admm_p,recall_admm_p,f_admm_p,precision_admm_n,recall_admm_n,f_admm_n,0.5*(f_admm_p+f_admm_n));
%% figures %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure;
semilogy(primal_gap_iter_admm,'-r','LineWidth',1.5);
% hold on;
% semilogy(primal_gap_iter_pds,'-b','LineWidth',1.5);
% hold on;
xlabel('iteration $k$','Interpreter','latex','FontSize',20);
ylabel('{$\|w^k-w^*\|_2$}','Interpreter','latex','FontSize',20);
lgd = legend('pADMM-SGL','location','northeast');
lgd.FontSize = 14;
beep on; beep;