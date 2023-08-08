function [G, XCoords, YCoords] = tvg_tver(N,varargin1,varargin2,varargin3)
% Erdos-Renyi random graph with temporal homogeneity

% Time-Varying Graph Learning
% with Constraints on Graph Temporal Variation

% Graph construction
% N num of nodes
% varargin1 the probability of each edge p
% varargin2 num of time slots
% varargin3 portion of resampling
max_attempt = 10;
rng(30);
G = cell(varargin2,1);

%% generate coordinates of vertices
plane_dim = 1;
XCoords = plane_dim*rand(N,1);
YCoords = plane_dim*rand(N,1);

%% construct the graph

% base graph
p = varargin1;
G_1= double(triu(rand(N)<p,1));
[rindex,cindex] = find(G_1);
% disp(rindex);
% disp(cindex);
G_1 = sparse(G_1);
weights = rand(nnz(G_1),1);

edge_count = nnz(G_1);
% disp(count);
sign_ = [ones(round(edge_count/2),1);-ones(edge_count-round(edge_count/2),1)];

sign_ = sign_(randperm(edge_count));


% disp(full(G_1));
G_1((cindex - 1) * N + rindex) = sign_ .* weights;
% disp(full(G_1));
if ~all(conncomp(graph(G_1+G_1'))==1)
    
    G = NaN;
    
    error('G1 is not connected!')
end

G_1 = G_1+G_1';
G{1} = G_1;
% disp(full(G_1));

% density_p = sum(sum(G{1}>1e-5));
% density_n = sum(sum(G{1}<-1e-5));
% disp(density_p);
% disp(density_n);
% num resample
% disp(edge_count*varargin3/2);
nr = ceil(edge_count*varargin3/2);
% disp(nr);
for t=2:varargin2
    d = sum(full(G_1~=0));
    % disp(t);
    % disp(d);
    for j=1:nr
        % disp(j);
        assert(all(d == sum(full(G_1~=0))));
        for i=1:max_attempt
            if i == max_attempt
                disp('max attemp reached!')
            end
            index = randperm(edge_count,2);
            r1 = rindex(index(1));
            c1 = cindex(index(1));
            r2 = rindex(index(2));
            c2 = cindex(index(2));
            % fprintf("r1 c1 r2 c2 = %d %d %d %d\n",r1, c1, r2, c2);
            if r1 == r2 || r1 == c2 || c1 == r2 || c1 == c2
                fprintf("duplicate node\n");
                continue
            end
            if any(d([r1 c1 r2 c2]) < 2)
                fprintf("node degree less than 2\n");
                continue
            end
            if G_1((r2 - 1) * N + r1) == 0 && G_1((c2 - 1) * N + c1) == 0
                cindex(index(1)) = r2;
                rindex(index(2)) = c1;
            elseif G_1((c2 - 1) * N + r1) == 0 && G_1((c1 - 1) * N + r2) == 0
                cindex(index(1)) = c2;
                cindex(index(2)) = c1;
            else
                fprintf("target edge is connected\n");
                continue
            end
            % disp(weights);
            weights(index) = rand(2,1);
            % disp(weights);

            G_1 = zeros(N);
            G_1((cindex - 1) * N + rindex) = sign_ .* weights;
            G_1 = sparse(G_1);
            G_1 = G_1 + G_1';
            break
        end
    end
    G{t}=G_1;
    % disp(G{t}-G{t-1});
end
%% plot the graph
% for t=1:varargin2
%     subplot(2,2,t);
%     H = plot(graph(G{t}),'EdgeLabel',graph(G{t}).Edges.Weight);
%     % H.XData = [1 2 4 5 3];
% end
% figure();wgPlot(G+diag(ones(N,1)),[XCoords YCoords],2,'vertexmetadata',ones(N,1));