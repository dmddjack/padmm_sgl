function [W, density_p, density_n] = gl_admm_solver(X, alpha, beta, delta, rho, tau1, tau2, max_iter, epsilon)


% min_{w,v} d'*w + beta*w'*w + alpha*||w||_1 + beta*v'*v
% s.t.      [S;B]w-[v;delta]=0
% 
%% initialization
DIM = size(X,1);
DIMw = DIM*(DIM-1)/2;
D = zeros(DIM,DIM);
for i = 1 : DIM
    for j = 1 : DIM
       D(i,j) = norm(X(i,:)-X(j,:),2)^2; 
    end 
end
% D = D / norm(D,'fro') * DIM;
d = squareform(D)';

% z = z/size(X,2); %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[S, St] = sum_squareform(DIM);
% S = [eye(DIMw);S];
% St = S';
d = d / norm(d) * 100;
%% iterations
%w = randn(DIMw,1);
w = ones(DIMw,1); %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
v = randn(DIM,1);
one = ones(DIMw,1);
y = randn(DIM+1,1);
C = [S;one'];
Ct = C';

fval_iter = zeros(max_iter,1);
primal_res_iter = zeros(max_iter,1);
dual_res_iter = zeros(max_iter,1);

for k = 1 : max_iter
    fval_iter(k) = d'*w + beta*(w'*w) + beta*((S*w)'*(S*w)) + alpha*norm(w,1); % commented when comparing runtime
    
    % update w
    
    p = w - tau1*rho*Ct*(C*w - [v;delta] - y/rho);
    % w = (p-tau1*d)/(2*tau1*beta+1);
    w = sign(p-tau1*d).*(max(abs(p-tau1*d)-tau1*alpha,0))/(2*tau1*beta+1);
    
    % update v
    v_tmp = v;
    Sw = S*w;
    y1 = y(1:DIM);
    
    q = (1-tau2*rho)*v + tau2*rho*Sw - tau2*y1;
    % v1_tmp = q(1:DIMw);
    % v2_tmp = q(DIMw+1:DIMw+DIM);
    % v1 = ;
    v = q/(2*tau2*beta+1);
    % v = cat(1,v1,v2);
    
    % updata y
    y = y - rho*(C*w - [v;delta]);
    
    % suboptimality measurements
    primal_res_iter(k) = norm(C*w - [v;delta]);
    dual_res_iter(k) = norm(rho*St*(v-v_tmp));
    % if mod(k, 1000) == 0
    %     disp(primal_res_iter(k));
    %     disp(dual_res_iter(k));
    % end
    
    % stopping criterion
    if (primal_res_iter(k) < epsilon) && (dual_res_iter(k) < epsilon)
%         fprintf('primal_gap_iter(%d)=%f',k,primal_res_iter(k));
%         fprintf('dual_gap_iter(%d)=%f',k,dual_res_iter(k));
        break;
    end
end
% disp(k);
W = squareform(w);

density_p = sum(w>1e-4)/max(size(w));
density_n = sum(w<-1e-4)/max(size(w));
% disp(density_p);
% disp(density_n);
