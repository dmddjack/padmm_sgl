function [W, fval_admm, fval_iter, primal_gap_iter] = dgl_admm(X, alpha, beta, gamma, delta, rho, tau1, tau2, max_iter, epsilon, w_opt,T)

% min_{w,v} 2*v'*w + beta*w'*w - alpha*ones'*log(v_1)+ gamma*|v_2|_{l1}
% s.t.      Q_dw-v=0, w>=0

%% initialization

%% initialization
[DIM,NUM] = size(X); % m in paper
DIMw = DIM*(DIM-1)/2; % p in paper
NUM = NUM/T;
% B_1
[S, St] = sum_squareform(DIM);
BCell = repmat({S}, 1, T);
B_1 = blkdiag(BCell{:});
B_1t = B_1';
% B_2
[tm,tp] = size(B_1);
% construct step by step
B_211 = zeros(DIMw,tp);
B_212 = -speye(tp-DIMw,tp);
B_21 = cat(1,B_211,B_212);
B_22 = speye(tp);
B_22(1:DIMw,:) = 0;
B_2 = B_21 + B_22;
B_2t = B_2';
% B_3
B3_Cell = repmat({ones(1,DIMw)}, 1, T);
B_3 = blkdiag(B3_Cell{:});
B_3t = B_3';
% Q_d
C_1 = cat(1,B_1,B_2);
C_1t = C_1';
C_2 = cat(1,C_1,B_3);
C_2t = C_2';
one = ones(T,1);
D = zeros(DIM,DIM);
for i = 1 : DIM
    for j = 1 : DIM
        D(i,j) = norm(X(i,1:NUM)-X(j,1:NUM),2)^2;
    end
end
% disp(D);
d = squareform(D)';
% disp(d);
for k = 1:T-1
    D = zeros(DIM,DIM);
    for i = 1 : DIM
        for j = 1 : DIM
            D(i,j) = norm(X(i,1+k*NUM:(k+1)*NUM)-X(j,1+k*NUM:(k+1)*NUM),2)^2;
        end
    end
    d_new = squareform(D)';
    d = cat(1,d,d_new);
end


%% iterations
w = zeros(tp,1);
% v = randn(tm+tp,1);
% y = randn(tm+tp+T,1);
v = zeros(tm+tp,1);
y = zeros(tm+tp+T,1);
fval_iter = zeros(max_iter,1);
primal_res_iter = zeros(max_iter,1); 
dual_res_iter = zeros(max_iter,1);
primal_gap_iter = zeros(max_iter,1);

% for varying t
% uncomment the below for varying penaly param
% miu = 10;
% tau_incr = 2;
% tau_decr = 2;
for k = 1 : max_iter
    
    % added by wxl // commented when comparing runtime
    % primal_gap_iter(k) = norm(w-w_opt);
    
    % update w
    p = w - tau1*rho*C_2t*(C_2*w - [v;delta*one] - y/rho);
    % w = (p-tau1*d)/(2*tau1*beta+1);
    w = sign(p-tau1*d).*(max(abs(p-tau1*d)-tau1*alpha,0))/(2*tau1*beta+1);
    
    % update v
    v_tmp = v;
    C_1w = C_1*w;
    C_2w = C_2*w;
    y1 = y(1:tm+tp);


    q = (1-tau2*rho)*v + tau2*rho*C_1w - tau2*y1;
    v_1_tmp = q(1:tm);
    v_2_tmp = q(tm+1:tm+tp);
    v_1 = v_1_tmp / (2*tau2*beta + 1);
    v_2 = sign(v_2_tmp).*(max(abs(v_2_tmp)-tau2*gamma,0)); % 'gamma' added by wxl
    v = cat(1,v_1,v_2);

    % updata y
    y = y - rho*(C_2w - [v;delta*one]);
    
    % suboptimality measurements
    primal_res_iter(k) = norm(C_2w - [v;delta*one]);
    dual_res_iter(k) = norm(rho*C_1t*(v-v_tmp));
    
    % uncomment the below for varying penalty param
%     if primal_res_iter(k)>miu*dual_res_iter(k)
%         t = tau_incr*t;
%     else
%         if dual_res_iter(k) >miu*primal_res_iter(k)
%             t = t/tau_decr;
%         end
%     end
    
    %fval_iter(k) = (d'*w + alpha*(norm(w,1)) + beta*(w'*w + (B_1*w)'*(B_1*w)) + gamma*norm(B_2*w, 1)); % commented when comparing runtime
    
    % stopping criterion
    if (primal_res_iter(k) < epsilon) && (dual_res_iter(k) < epsilon)
        % fprintf('primal_gap_iter(%d)=%f',k,primal_res_iter(k));
        % fprintf('dual_gap_iter(%d)=%f\n',k,dual_res_iter(k));
        fval_admm = fval_iter(k);
        break;
    end
end
if k == max_iter
    fval_admm = fval_iter(max_iter);
end
% fprintf('primal_gap_iter(%d)=%f',k,primal_res_iter(k));
% fprintf('dual_gap_iter(%d)=%f\n',k,dual_res_iter(k));
W = cell(T,1);
w_1 = w(1:DIMw);
W{1} = squareform(w_1);
% density_p = sum(w_1>1e-4)/max(size(w_1));
% density_n = sum(w_1<-1e-4)/max(size(w_1));
% similarity = 0;
% w_last = w_1;

for t=2:T
    w_t = w((t-1)*DIMw+1:t*DIMw);
    W{t} = squareform(w_t);
    % density_p = density_p + sum(w_t>1e-4)/max(size(w_t));
    % density_n = density_n + sum(w_t<-1e-4)/max(size(w_t));
    % corr = corrcoef(w_t,w_last);
    % similarity = similarity + corr(1,2);
    % w_last = w_t;
end
% density_p = density_p / T;
% density_n = density_n / T;
% similarity = similarity / (T - 1);
% fprintf("density_p = %.4f, density_n = %.4f \nsimilarity_admm = %.4f\n",density_p, density_n, similarity);
