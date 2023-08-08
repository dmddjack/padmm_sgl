function [W, fval_cvx] = gl_cvx(X, alpha, beta, gamma)

% min_{W} trace(W*D) - alpha*sum(log(sum(W0,2))) + 0.5*beta*(norm(W0,'fro'))^2
% s.t.    W>=0, W=W', diag(W)=0

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

% disp(d);

d = d / norm(d) * 100;
[S, St] = sum_squareform(DIM);
p = [eye(DIMw),zeros(DIMw,DIMw)];
n = [zeros(DIMw,DIMw),eye(DIMw)];
% cvx

cvx_begin

cvx_precision best

variable w(DIMw)

minimize (d'*w + beta*square_pos(norm(w))+ beta*square_pos(norm(S*w)) + alpha*(norm(w,1)))

subject to
    ones(DIMw,1)'*w == gamma;
    % norm(w,1) <= 2*DIM;
    % S*w == zeros(DIM, 1);

cvx_end

% cvx_begin
% 
% cvx_precision best
% 
% variable w(2*DIMw)
% 
% minimize (d'*(p*w-n*w) - alphap*sum_log(S*p*w) + betap*square_pos(norm(n*w)) ...
%     + alphan*norm(n*w,1) + betan*square_pos(norm(n*w)))
% 
% subject to
%     % ones(DIMw,1)'*w == 0;
%     p*w >= 0;
%     n*w >= 0;
%     % norm(w,1) <= 2*DIM;
%     % S*w == zeros(DIM, 1);
% 
% cvx_end

% cvx_begin
% 
% cvx_precision best
% 
% variable W(DIM,DIM)
% 
% minimize (sum(sum(D.*W)) + beta*square_pos(norm(W,'fro'))+ alpha*(sum(sum(abs(W)))))
% 
% subject to
%     sum(sum(W)) == 0;
%     % norm(w,1) <= 2*DIM;
%     % S*w == zeros(DIM, 1);
% 
% cvx_end
% 
fval_cvx = cvx_optval;
W = squareform(w);
% W = squareform(p*w-n*w);
% disp(w);
% density_p = sum(p*w>1e-4)/max(size(p*w));
% density_n = sum(n*w>-1e-4)/max(size(n*w));
% disp(w);
% disp(p*w);
% disp(n*w);
density_p = sum(w>1e-4)/max(size(w));
density_n = sum(w<-1e-4)/max(size(w));
disp(density_p);
disp(density_n);


% % test
% fval_cvx_opt = trace(W*D) - alpha*sum(log(sum(W,2))) + 0.5*beta*(norm(W,'fro'))^2;
% fprintf('\nfval_cvx_opt=%f-------------------------------------\n', fval_cvx_opt);