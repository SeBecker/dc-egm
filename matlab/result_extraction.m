addpath('utils')

%%
% Solve model
m5=model_retirement;
m5.ngridm=500;
m5.df=1/(1+m5.r); %flat consumption hopefully
m5.sigma=0.35;
m5.lambda=0.2; %some EV taste shocks
m5.nsims=50;
m5.init=[5 20];

%%
tic
m5.solve_dcegm;

%%
%Save result
for k1=1:25
   writematrix(m5.policy(1, k1).x, sprintf('policym5/x%d_%d.txt',k1, 1))
   writematrix(m5.policy(1, k1).y, sprintf('policym5/y%d_%d.txt', k1, 1))
   writematrix(m5.policy(2, k1).x, sprintf('policym5/x%d_%d.txt',k1, 2))
   writematrix(m5.policy(2, k1).y, sprintf('policym5/y%d_%d.txt', k1, 2))
   writematrix(m5.value(1, k1).x, sprintf('valuem5/x%d_%d.txt',k1, 1))
   writematrix(m5.value(1, k1).y, sprintf('valuem5/y%d_%d.txt', k1, 1))
   writematrix(m5.value(2, k1).x, sprintf('valuem5/x%d_%d.txt',k1, 2))
   writematrix(m5.value(2, k1).y, sprintf('valuem5/y%d_%d.txt', k1, 2))
end