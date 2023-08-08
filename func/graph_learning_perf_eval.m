function [precision_p,recall_p,f_p, precision_n,recall_n,f_n,NMI_p,num_of_edges_p,NMI_n,num_of_edges_n] = graph_learning_perf_eval(L_0,L)
% evaluate the performance of graph learning algorithms

L_0tmp = L_0-diag(diag(L_0));
Lp_0tmp =  L_0tmp.*(L_0tmp<0);
Ln_0tmp =  L_0tmp.*(L_0tmp>0);
edges_groundtruth_p = squareform(Lp_0tmp)~=0;
edges_groundtruth_n = squareform(Ln_0tmp)~=0;


Ltmp = L-diag(diag(L));
Lp_tmp =  Ltmp.*(Ltmp<0);
Ln_tmp =  Ltmp.*(Ltmp>0);
edges_learned_p = squareform(Lp_tmp)~=0;
edges_learned_n = squareform(Ln_tmp)~=0;

num_of_edges_p = sum(edges_learned_p);
num_of_edges_n = sum(edges_learned_n);


if num_of_edges_p > 0
    [precision_p,recall_p] = perfcurve(double(edges_groundtruth_p),double(edges_learned_p),1,'Tvals',1,'xCrit','prec','yCrit','reca');
    if precision_p == 0 && recall_p == 0
        f_p = 0;
    else
        f_p = 2*precision_p*recall_p/(precision_p+recall_p);
    end
    NMI_p = perfeval_clus_nmi(double(edges_groundtruth_p),double(edges_learned_p));
else
    precision_p = 0;
    recall_p = 0;
    f_p = 0;
    NMI_p = 0;
end

if num_of_edges_n > 0
    [precision_n,recall_n] = perfcurve(double(edges_groundtruth_n),double(edges_learned_n),1,'Tvals',1,'xCrit','prec','yCrit','reca');
    if precision_n == 0 && recall_n == 0
        f_n = 0;
    else
        f_n = 2*precision_n*recall_n/(precision_n+recall_n);
    end
    NMI_n = perfeval_clus_nmi(double(edges_groundtruth_n),double(edges_learned_n));
else
    precision_n = 0;
    recall_n = 0;
    f_n = 0;
    NMI_n = 0;
end