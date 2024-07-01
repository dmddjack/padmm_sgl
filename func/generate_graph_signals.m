function [X_noisy, L] = generate_graph_signals(N,A,DIM,seed)
% generate noisy signal
rng(seed);
NUM = N;
Ap =  A.*(A>0);
Dp = diag(sum(full(Ap)));
Lp = Dp-full(Ap);
An = -A.*(A<0);
Dn = diag(sum(full(An)));
Ln = Dn-full(An);
% disp(A);

Lp = Lp/trace(Lp)*DIM;
Ln = Ln/trace(Ln)*DIM;
L = Lp - Ln;

mid = round(DIM / 2);
[Vp,Dp] = eig(full(Lp));
hp = zeros(size(Dp));
hp(Dp>1e-10) = 1./sqrt(Dp(Dp>1e-10));

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
%X = 2./(1+exp(-X))-1;
%[val, indx] = max(X);
%[val, indy] = max(val);
%disp([indy, indx(indy)])
%disp(val)
%disp(sum(sum(X)));

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
