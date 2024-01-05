function [G, XCoords, YCoords, sign_] = construct_graph(N,opt,seed,varargin1,varargin2)
% Graph construction

rng(seed);

%% check inputs
if nargin == 3
    if strcmp(opt,'chain') == 0
        error('number of input variables not correct :(')
    end
elseif nargin == 4
    if strcmp(opt,'gaussian') || strcmp(opt,'ff')
        error('number of input variables not correct :(')
    end
elseif nargin == 5
    if strcmp(opt,'er') || strcmp(opt,'pa')
        error('number of input variables not correct :(')
    end
end

%% generate coordinates of vertices
plane_dim = 1;
XCoords = plane_dim*rand(N,1);
YCoords = plane_dim*rand(N,1);

%% construct the graph
switch opt
    case 'gaussian' % random graph with Gaussian weights
        T = varargin1; 
        s = varargin2;
        d = distanz([XCoords,YCoords]'); 
        W = exp(-d.^2/(2*s^2)); 
        W(W<T) = 0; % Thresholding to have sparse matrix
        W = W-diag(diag(W));

        count = nnz(W);
        % disp(count);
        sign_ = [ones(round(count/2),1);-ones(count-round(count/2),1)];
        sign_ = sign_(randperm(count));
        disp(sum(sign_));
        tmp = zeros(size(W));
        tmp(W>0) = sign_ .* W(W>0);

        W = tmp + tmp';

        G = W-diag(diag(W));
        % disp(count);
        
    case 'er' % Erdos-Renyi random graph
        p = varargin1;
        G = erdos_reyni(N,p);
        count = nnz(G);

        sign_ = [ones(round(count/2),1);-ones(count-round(count/2),1)];
        sign_ = sign_(randperm(count));
        % disp(G(G>0));
        
        tmp = zeros(size(G));
        tmp(G>0) = sign_ .* G(G>0);
        G = tmp;
        % disp(G);

        G = G + G';
        G = sparse(G);
        % S = rand(N)*2-1;
        % S = triu((S>0)-(S<0));
        % S = S + S';
        % G = G.*S;
        
    case 'pa' % scale-free graph with preferential attachment
        m = varargin1;
        G = preferential_attachment_graph(N,m);
        count = nnz(G);

        sign_ = [ones(round(count/2),1);-ones(count-round(count/2),1)];
        sign_ = sign_(randperm(count));
        % disp(G(G>0));
        
        tmp = zeros(size(G));
        tmp(G>0) = sign_ .* G(G>0);
        G = tmp;
        % disp(G);

        G = G + G';
        G = sparse(G);
        % S = rand(N)*2-1;
        % S = triu((S>0)-(S<0));
        % S = S + S';
        % G = G.*S;
        
    case 'ff', % forest-fire model
        p = varargin1;
        r = varargin2;
        G = forest_fire_graph(N,p,r);
        
    case 'chain' % chain graph
        G = spdiags(ones(N-1,1),-1,N,N);
        G = G + G';
end

% plot(graph(G),'EdgeLabel',graph(G).Edges.Weight);

if ~all(conncomp(graph(G))==1)
    
    G = NaN;
    
    error('G is not connected!');

    return;
end


%% plot the graph
% figure();wgPlot(G+diag(ones(N,1)),[XCoords YCoords],2,'vertexmetadata',ones(N,1));