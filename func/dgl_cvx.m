function [W, w, fval_cvx] = dgl_cvx(X, alpha, beta, gamma, delta, T)

% min_{w,v} 2*v'*w + beta*w'*w - alpha*ones'*log(v_1)+ gamma*|v_2|_{l1}
% s.t.      Q_dw-v=0, w>=0

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
C = cat(1,B_1,B_2);
Ct = C';

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

cvx_begin

cvx_precision default

variable w(tp)

minimize (d'*w + alpha*(norm(w,1)) + beta*(square_pos(norm(w)) + square_pos(norm(B_1*w))) + gamma*norm(B_2 * w, 1))

subject to
    B_3*w == delta * ones(T,1);

cvx_end

W = cell(T,1);
w_1 = w(1:DIMw);
W{1} = squareform(w_1);
density_p = sum(w_1>1e-4)/max(size(w_1));
density_n = sum(w_1<-1e-4)/max(size(w_1));
similarity = 0;
w_last = w_1;

for t=2:T
    w_t = w((t-1)*DIMw+1:t*DIMw);
    W{t} = squareform(w_t);
    density_p = density_p + sum(w_t>1e-4)/max(size(w_t));
    density_n = density_n + sum(w_t<-1e-4)/max(size(w_t));
    corr = corrcoef(w_t,w_last);
    % disp(corr);
    similarity = similarity + corr(1,2);
    w_last = w_t;
end
density_p = density_p / T;
density_n = density_n / T;
similarity = similarity / (T - 1);
fprintf("density_p = %.4f, density_n = %.4f \nsimilarity = %.4f\n",density_p, density_n, similarity);

fval_cvx = cvx_optval;


