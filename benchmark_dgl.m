terminate(pyenv);
opts = ["tvpa"];
% DIMs = [20, 50, 80, 100];
DIMs = [50];
seeds = 114510:1:114559;
% seeds = 114511:1:114512;
NUM = 100;
time_slots = 10;
%opt = 'gaussian';
for opt = opts
    disp(opt);
    for DIM = DIMs
        outputs = zeros(length(seeds)+2, 10);
        i = 1;
        for seed = seeds
            % alpha = randn * .019 + .18; 
            % beta = randn * .1 + 1;
            % delta = randn * .6 + -5.5; 
            % rho = randn * .005 + .05;
            fprintf('seed: %d\n', seed);
            output = main_dgl(seed, DIM, NUM, time_slots, opt);
            if ~isnan(output)
                outputs(i, :) = [seed, output];
                i = i + 1;
            end
        end
        mean_out = mean(outputs(1:i-1, 2:end));
        std_out = std(outputs(1:i-1, 2:end));
        outputs(end-1:end, :) = [[-1,mean_out]; [-2,std_out]];
        filename = sprintf('experiments/dgl_%s_%d.csv', opt, DIM);
        T = array2table(outputs, 'VariableNames', {'seed', 'cvx time', 'dynsgl time', 'admm time', 'dynsgl F1+','dynsgl F1-','dynsgl F1','admm F1+', 'admm F1-', 'admm F1'});
        writetable(T, filename);
    end
end