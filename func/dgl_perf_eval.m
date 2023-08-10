function [precision_p, recall_p, f_p, precision_n, recall_n, f_n] = dgl_perf_eval(L_0, W, T)

precision_p_all = 0;
recall_p_all = 0;
f_p_all = 0;
precision_n_all = 0;
recall_n_all = 0;
f_n_all = 0;

for t=1:T
    W_t = W{t};
    D = diag(sum(full(W_t)));
    L = D-full(W_t);
    L(abs(L)<10^(-4))=0;
    [precision_p_tmp,recall_p_tmp,f_p_tmp,precision_n_tmp,recall_n_tmp,f_n_tmp,~] = graph_learning_perf_eval(L_0{t},L);
    
    precision_p_all = precision_p_all + precision_p_tmp;
    recall_p_all = recall_p_all + recall_p_tmp;
    f_p_all = f_p_all + f_p_tmp;
    precision_n_all = precision_n_all + precision_n_tmp;
    recall_n_all = recall_n_all + recall_n_tmp;
    f_n_all = f_n_all + f_n_tmp;
end

precision_p = precision_p_all / T;
recall_p = recall_p_all / T;
f_p = f_p_all / T;
precision_n = precision_n_all / T;
recall_n = recall_n_all / T;
f_n = f_n_all / T;