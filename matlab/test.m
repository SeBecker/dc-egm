%% Nice simulation graphics using retirement model
m5=model_retirement;
m5.ngridm=500;
m5.df=1/(1+m5.r); %flat consumption hopefully
m5.sigma=0.35;
m5.lambda=0.2; %some EV taste shocks
m5.nsims=50;
m5.init=[5 20];
tic
m5.solve_dcegm;
