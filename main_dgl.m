%% README
% Construct graph: function construct_tvg
% Generate signal: function generate_graph_signals
% ADMM solver: function dgl_admm
function [output, primal_gap_iter_admm] = main_dgl(seed, DIM, NUM, time_slots, opt)
addpath(genpath("D:\user\OneDrive - The Chinese University of Hong Kong\Document(OneDrive)\CUHK\2022-23 Summer Research\code\func"));
addpath(genpath("D:\user\OneDrive - The Chinese University of Hong Kong\Document(OneDrive)\CUHK\2022-23 Summer Research\code\cvx"));
% clear;
close all
% seed = 30;
rng(seed);
cvx = true;
dynSGL = true;
% cvx_time = 0;
%% common parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if DIM==100
    alpha = .13;
    beta = .8;
    gamma = .3;
    delta = -1;
    rho = 0.07;
    epsilon = 5e-3;
    %alpha=0.093; beta=0.570; gamma=1.695; delta=-14.268; rho=.5;
elseif DIM==80
    alpha = .184;
    beta = .8;
    gamma = .445;
    delta = -2;
    rho = .12;
    %alpha=0.099; beta=0.628; gamma=1.682; delta=-23.490; rho=.5;
    epsilon = 1e-3;
elseif DIM==50
    alpha = .4;
    beta = 1;
    gamma = .41;
    delta = -4.2;
    rho = 0.2;
    epsilon = 1e-5;
elseif DIM==20
    alpha = 1.8;
    beta = 1;
    gamma = 5.6;
    delta = -6.8;
    rho = 1;
    epsilon = 1e-6;
end

fprintf('alpha=%.3f, beta=%.3f, gamma=%.3f, delta=%.3f\n', alpha, beta, gamma, delta);

%% generate a graph %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% synthetic data
disp(opt);
try
    switch opt
        case 'tver'
            p_connect = 0.2;
            p_resample = 0.05;
            [A, ~] = construct_tvg(DIM,'tver',seed,p_connect,time_slots,p_resample);
        case 'tvpa'
            m = round(DIM*0.1);
            p_resample = 0.05;
            [A, ~] = construct_tvg(DIM,'tvpa',seed,m,time_slots,p_resample);
        otherwise
            error('wrong opt');
    end
catch exception
    disp(exception.message);
    output = NaN;
    primal_gap_iter_admm = NaN;
    return;
end

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
file_name = sprintf("data/X_d_%d_%d.csv", time_slots, seed);
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



%% obtain optimal solution via ADMM %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% rho = 1;
fprintf('solving...\n');
tau1 = 1/(rho*(sqrt(2*(DIM-1))+2+sqrt(DIM*(DIM-1)/2))^2);
tau2 = 1/rho;
max_iter_opt = 1e6;
epsilon_opt = 1e-13;

max_try = 50;
alpha_opt = alpha;
beta_opt = beta;
beta_original = beta;
gamma_opt = gamma;
delta_opt = delta;
rho_opt = rho;
fprintf("try #0\n");
tic
[W_opt, w_opt, density_p, density_n, similarity] = dgl_admm_solver(X_noisy, alpha, beta, gamma, delta, rho, tau1, tau2, max_iter_opt, epsilon, time_slots);
% w_opt = zeros(DIM*(DIM-1)/2*time_slots,1);
last_time = toc
error_p = (density_p - .11)^2;
error_n = (density_n - .11)^2;
error_s = (similarity - .95)^2;

dp_best = density_p;
dn_best = density_n;
sim_best = similarity;
disp(dp_best);
disp(dn_best);
disp(sim_best);
W_opt_best = W_opt;
for i = 1:max_try
    if (error_p + error_n + error_s < 0.015^2)
        break
    end
    fprintf("try #%d\n", i);
    %rho = rho_opt * (1 + 0.1*randn());
    %tau1 = 1/(rho*(sqrt(2*(DIM-1))+2+sqrt(DIM*(DIM-1)/2))^2);
    %tau2 = 1/rho;
    alpha = alpha_opt * (1 + 0.1*randn());
    if density_p + density_n > .22
        beta = beta_opt * (1 + 0.1*abs(randn()));
    else
        beta = beta_opt * (1 - 0.1*abs(randn()));
    end
    
    if similarity < .95
        gamma = gamma_opt * (1 + 0.1*abs(randn()));
    else
        gamma = gamma_opt * (1 - 0.1*abs(randn()));
    end

    if dp_best < dn_best
        delta = delta_opt + abs(randn());
    else
        delta = delta_opt - abs(randn());
    end
    %if beta < beta_original
    %    rho = rho_opt * max(beta / beta_opt, 1) * gamma / gamma_opt;
    %else
    rho = rho_opt * beta / beta_opt * gamma / gamma_opt;
    %end
    tau1 = 1/(rho*(sqrt(2*(DIM-1))+2+sqrt(DIM*(DIM-1)/2))^2);
    tau2 = 1/rho;
    % [rho, alpha, beta, gamma, delta]
    tic
    [W_opt, w_opt, density_p, density_n, similarity] = dgl_admm_solver(X_noisy, alpha, beta, gamma, delta, rho, tau1, tau2, max_iter_opt, epsilon, time_slots);
    time = toc
    error_new_p = (density_p - .11)^2;
    error_new_n = (density_n - .11)^2;
    error_new_s = (similarity - .95)^2;
    %if time < last_time
    %    rho_opt = rho;
    %    tau1 = 1/(rho_opt*(sqrt(2*(DIM-1))+2+sqrt(DIM*(DIM-1)/2))^2);
    %    tau2 = 1/rho_opt;
    %end
    if (error_new_p + error_new_n  < error_p + error_n)
        alpha_opt = alpha;
        %if beta < beta_original
        %    rho_opt = rho_opt * max(beta / beta_opt, 1);
        %else
            rho_opt = rho_opt * beta / beta_opt;
        %end
        tau1 = 1/(rho_opt*(sqrt(2*(DIM-1))+2+sqrt(DIM*(DIM-1)/2))^2);
        tau2 = 1/rho_opt;
        beta_opt = beta;
        delta_opt = delta;
        error_p = error_new_p;
        error_n = error_new_n;
        dp_best = density_p;
        dn_best = density_n;
        disp(dp_best);
        disp(dn_best);
    end
    if error_new_s < error_s
        rho_opt = rho_opt * gamma / gamma_opt;
        tau1 = 1/(rho_opt*(sqrt(2*(DIM-1))+2+sqrt(DIM*(DIM-1)/2))^2);
        tau2 = 1/rho_opt;
        gamma_opt = gamma;
        error_s = error_new_s;
        sim_best = similarity;
        disp(sim_best);
        W_opt_best = W_opt;
    end

end
alpha = alpha_opt;
beta = beta_opt;
gamma = gamma_opt;
delta = delta_opt;
rho = rho_opt;
tau1 = 1/(rho*(sqrt(2*(DIM-1))+2+sqrt(DIM*(DIM-1)/2))^2);
tau2 = 1/rho;
%[W_opt, w_opt, density_p, density_n, similarity] = dgl_admm_solver(X_noisy, alpha, beta, gamma, delta, rho, tau1, tau2, max_iter_opt, epsilon_opt, time_slots);

disp(dp_best);
disp(dn_best);
disp(sim_best);
fprintf('optimal solution obtained\n');

%% CVX %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if cvx
    tic
    [W_cvx, ~] = dgl_cvx(X_noisy, alpha, beta, gamma, delta, time_slots); % run algorithm
    cvx_time = toc;
    [precision_cvx_p,recall_cvx_p,f_cvx_p,precision_cvx_n,recall_cvx_n,f_cvx_n] = dgl_perf_eval(L0,W_cvx, time_slots);
    % fval_cvx = 0.5*trace(W_cvx*Z); % + 0.5*beta*(norm(W_cvx,'fro'))^2;
    % disp(density_p);
    % disp(density_n);
end

%% for comparing ADMM & dynSGL %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
max_iter = 1e6;

%% ADMM %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic
[W_admm, fval_admm, fval_admm_iter, primal_gap_iter_admm] = dgl_admm(X_noisy, alpha, beta, gamma, delta, rho, tau1, tau2, max_iter, epsilon, w_opt, time_slots);
admm_time = toc;
[precision_admm_p,recall_admm_p,f_admm_p,precision_admm_n,recall_admm_n,f_admm_n] = dgl_perf_eval(L0, W_admm, time_slots);

%% dynSGL %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if dynSGL
    if pyenv().Version == ""
        pyenv(Version="C:\Users\dmddj.PRO13\.conda\envs\dynSGL\python.exe",ExecutionMode="InProcess");
    end
    command = sprintf("dynSGL.py -t %d -s %d", time_slots, seed);
    dynSGL_time  = pyrunfile(command,'toc');
    file_name = sprintf("data/W_dynSGL_%d_%d.csv", time_slots, seed);
    data = readmatrix(file_name);    
    W_dynSGL = cell(time_slots,1);
    for t=1:time_slots
        W_dynSGL{t} = data(:,1+(t-1)*DIM:t*DIM);
    end
    [precision_dynSGL_p,recall_dynSGL_p,f_dynSGL_p,precision_dynSGL_n,recall_dynSGL_n,f_dynSGL_n] = dgl_perf_eval(L0, W_dynSGL, time_slots);
end

%% outputs %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('similarity_ground_truth = %f\n', similarity_ground_truth);
fprintf('alpha=%.3f, beta=%.3f, gamma=%.3f, delta=%.3f\n', alpha, beta, gamma, delta);
if cvx
    f_cvx = 0.5*(f_cvx_p+f_cvx_n);
    fprintf('----- CVX  Time needed is %f -----\n', cvx_time);
    % fprintf('CVX               | fval_cvx=%f\n', fval_cvx);
    fprintf('CVX measurements  | precision_cvx_p=%f,recall_cvx_p=%f,f_cvx_p=%f\n                  | precision_cvx_n=%f,recall_cvx_n=%f,f_cvx_n=%f\n                  | f_cvx=%f\n\n' ...
        ,precision_cvx_p,recall_cvx_p,f_cvx_p,precision_cvx_n,recall_cvx_n,f_cvx_n,f_cvx);
end
if dynSGL
    f_dynSGL = 0.5*(f_dynSGL_p+f_dynSGL_n);
    fprintf('----- dynSGL Time needed is %f -----\n', dynSGL_time);
    % fprintf('dynSGL | fval_dynSGL=%f, t=%f, tau1=%f, tau2=%f, max_iter=%d\n', fval_dynSGL, rho, tau1, tau2, max_iter);
    fprintf('dynSGL measurements  | precision_dynSGL_p=%f,recall_dynSGL_p=%f,f_dynSGL_p=%f\n                  | precision_dynSGL_n=%f,recall_dynSGL_n=%f,f_dynSGL_n=%f\n                  | f_dynSGL=%f\n\n' ...
        ,precision_dynSGL_p,recall_dynSGL_p,f_dynSGL_p,precision_dynSGL_n,recall_dynSGL_n,f_dynSGL_n,f_dynSGL);
end
f_admm = 0.5*(f_admm_p+f_admm_n);
fprintf('----- ADMM Time needed is %f -----\n', admm_time);
fprintf('ADMM | fval_admm=%f, t=%f, tau1=%f, tau2=%f, max_iter=%d\n', fval_admm, rho, tau1, tau2, max_iter);
fprintf('ADMM measurements  | precision_admm_p=%f,recall_admm_p=%f,f_admm_p=%f\n                  | precision_admm_n=%f,recall_admm_n=%f,f_admm_n=%f\n                  | f_admm=%f\n\n' ...
    ,precision_admm_p,recall_admm_p,f_admm_p,precision_admm_n,recall_admm_n,f_admm_n,f_admm);

output = [cvx_time,dynSGL_time,admm_time,f_dynSGL_p,f_dynSGL_n,f_dynSGL,f_admm_p,f_admm_n,f_admm];
% output = [0, dynSGL_time,admm_time, f_dynSGL_p,f_dynSGL_n,f_dynSGL, f_admm_p,f_admm_n,f_admm];