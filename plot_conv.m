clear;
close all;
seed = 114535;
DIM = 100;
NUM = 100; 
time_slots = 10; 
opts = ["tver", "tvpa"];
legends = ["ER", "PA"];
colors = ["-r", "-b"];
% opts = ["gaussian", "er", "pa"];
% legends = ["gaussian", "ER", "PA"];

primal_gap = cell(2);
for i = 1:length(opts)
    [output, primal_gap{i}] = main_dgl(seed, DIM, NUM, time_slots, opts(i));
end
figure('Position', [100, 100, 600, 450]);
set(gcf, 'renderer', 'painters');  % Change renderer to 'painters'
for i = 1:length(opts)
    semilogy(primal_gap{i},colors(i),'LineWidth',1.5);
    hold on;
end

xlabel('iteration $k$','Interpreter','latex','FontSize',20);
ylabel('{$\|w^k-w^*\|_2$}','Interpreter','latex','FontSize',20);
lgd = legend(legends,'location','northeast');
lgd.FontSize = 18;
beep on; beep;