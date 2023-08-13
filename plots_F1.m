clear;
close all;
% Create bar data
% F_score = [0.4275546 0.441271 0.4344127; 0.4752857 0.4551524 0.4652188]'; 
F_score = [0.4273837 0.5188922 0.4731379; 0.6218447 0.537915 0.5798798]'; 
% F_score = [0.3537008	0.4171802	0.3854407;0.6027939	0.5468985	0.5748461]';

% Create error data (using random data for this demo)
% err  = [0.030814475	0.021108031	0.018332177; 0.022046061 0.018892471 0.015853581]';
err  = [0.148728673	0.080341148	0.105168064; 0.09051367	0.078886056	0.073440446]';
% err = [0.058266957	0.044686729	0.030076139; 0.082361468	0.070272023	0.049945575]';
% Plot bars

figure;
x0=100;
y0=100;
width=700;
height=500;
set(gcf,'position',[x0,y0,width,height])
X_label = categorical({'F1+','F1-','F1'});
X_label = reordercats(X_label, {'F1+','F1-','F1'});
h = bar(X_label, F_score, 0.5, 'grouped');
set(h, 'BarWidth', 1); 

xCnt = ([1:3] + cell2mat(get(h,'XOffset'))).';

hold on
errorbar(xCnt(:), F_score(:), err(:), err(:), 'k', 'LineStyle','none');

% Add values on top of each bar
for i = 1:numel(F_score)
    text(xCnt(i), F_score(i)+err(i), num2str(F_score(i), '%0.4f'), 'HorizontalAlignment','center', 'VerticalAlignment','bottom');
end

% axis square
% Set the y-axis range
yMin = 0.25; % Minimum value
yMax = 0.78; % Maximum value
ylim([yMin, yMax]);

lgd = legend('SGL','pADMM-SGL','location','northeast');
lgd.FontSize = 14;
saveas(gcf,'s_m=20.png')