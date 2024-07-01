% terminate(pyenv);
opts = ["gaussian", "er", "pa"];
DIMs = [20, 50, 80, 100];
% DIMs = [80];
seeds = 114510:1:114559;
NUM = 100;
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
            output = main_gl(seed, DIM, NUM, opt);
            if ~isnan(output)
                outputs(i, :) = [seed, output];
                i = i + 1;
            end
        end
        mean_out = mean(outputs(1:i-1, 2:end));
        std_out = std(outputs(1:i-1, 2:end));
        outputs(end-1:end, :) = [[-1,mean_out]; [-2,std_out]];
        filename = sprintf('experiments/gl_%s_%d.csv', opt, DIM);
        T = array2table(outputs, 'VariableNames', {'seed', 'cvx time', 'sgl time', 'admm time', 'sgl F1+','sgl F1-','sgl F1','admm F1+', 'admm F1-', 'admm F1'});
        writetable(T, filename);
    end
end