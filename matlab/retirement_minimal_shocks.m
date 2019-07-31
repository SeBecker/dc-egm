addpath('utils')

%%
%%%%%%%%%%% Run solve_m0 DC-EGM %%%%%%%%%%%%%%%%%
%% Model with minimal shocks, benchmark for usage of DC step
m0=model_retirement;
m0.sigma=0;
m0.lambda=eps; 
%no EV taste shocks => value functions are not smoothed out

tic
m0.solve_dcegm;

%%
%Default parameters of the model
label		= 'Consumption model with retirement'; %name of this model
Tbar		= 25			; %number of periods (fist period is t=1) 
ngridm	= 500		; %number of grid points over assets
mmax		= 50		; %maximum level of assets
expn		=	5 		; %number of quadrature points used in calculation of expectations
nsims		= 10		; %number of simulations
init    =[10 30] ; %interval of the initial wealth
r = 0.05	; %interest rate
df 			= 0.95	; %discount factor
sigma   = 0.25	; %sigma parameter in income shocks
duw     =	0.35	; %disutility of work
theta		= 1.95 	; %CRRA coefficient (log utility if ==1)
inc0		= 0.75	; %income equation: constant
inc1		= 0.04  ; %income equation: age coef
inc2		= 0.0002;	%income equation: age^2 coef
cfloor  =	0.001	; %consumption floor (safety net in retirement)
lambda  = 0.02  ; %scale of the EV taste shocks 

%%
% Reparametrise to minimal shocks
sigma=0;
lambda=eps;

%%
% retirement_model.m line 86ff.
policy = polyline;
value = polyline;

[quadp quadw]=quadpoints(expn,0,1);
quadstnorm=norminv(quadp,0,1);
savingsgrid=linspace(0,mmax,ngridm);

%%
    for id=1:2 %1=remain worker, 2=retire
        policy(id,25)=polyline([0 mmax],[0 mmax],'Policy function in period T');
        value(id,25)=polyline([0 mmax],[0 NaN],'Value function in period T');
    end

    
%%
% Plug in values from m0 for periods 24 and 23
% Since DC step comes in only in T-3 we skip the line by line solution
% for the 1st two periods.
    for id=1:2 %1=remain worker, 2=retire
        for it = [23, 24]
            policy(id,it)=m0.policy(id, it);
            value(id,it)=m0.value(id, it);
        end
    end
%%
% Run line-by-line for T-3, period 22 in MatLab = 21 in Python
it = 22;

%% 
% M_{t+1} implied by the exogenous grid
wk1 = budget(it, savingsgrid, quadstnorm*sigma, 1);
wk1 = max(cfloor, wk1);
wk1_reshape = wk1(1:end);

%%
% Value function working in t+1
vl1 = value_function(1, it+1, wk1_reshape, value);

%%
% Value function retirement in t+1
vl1(2, :) = value_function(2, it+1, wk1_reshape, value);

%%
% Choice probability of working, for a worker
id_id = 1;
pr1 = chpr(vl1, lambda)*(id_id == 1);
% Switches from 1 to 0 at index 130-131

%%
% Next period consumption
cons1 = policy(:, it+1).interpolate(wk1);
% Consumption for choice retirement is uniformly lower

%%
% Marginal utility by choice probabilities
mu1=pr1.*mutil(cons1(1,:))+(1-pr1).*mutil(cons1(2,:));

%%
% Next period marginal wealth
% Constant for this model
mwk1 = mbudget(it, savingsgrid, quadstnorm*sigma, 1);

%%
% RHS of the Euler equation
rhs=quadw'*(reshape(mu1,size(wk1)).*mwk1);

%%
% Consumption in the current period
cons0=imutil(df*rhs);
% Drop at index 26-27

%%
% Fill in information on endogenous grid and consumption for it
policy(1,it)=polyline(savingsgrid+cons0,cons0,sprintf('period %d, choice %d (1=working)',it,1));

%%
% Compute expected value functions
% worker
ev=quadw'*reshape(logsum(vl1, lambda),size(wk1));%expected value function for 1=worker

% retiree
% ev=quadw'*value_function(2, 25, wk1, value);%expected value function for 2=retiree

%%
% Save into value function object
value(1,it)=polyline(savingsgrid+cons0,util(cons0,1)+df*ev,sprintf('VF in period %d, choice %d (1=working)',it,1));%uncleaned

%%
%%%%%%%% Secondary envelope %%%%%%%%%%%%

minx = min(value(1, it).x);
is_true = value(1, 22).x(1) <= minx;
% Start secondary envelope here if true
% Analyses if there is a discontinuity problem in credit constrained region
% To Do: Handling of discontinuity in credit constrained region

%%
%%%%%%%% As function %%%%%%%%
[value(1, it) indxdel, newdots] = value(1, it).secondary_envelope;

%%
%%%%%%%% Line-by-line %%%%%%%%
% Start secondary envelope
% Here numel(obj) = numel(value(1, it)) = 1;
cur = value(1, it);

%% 
ii=cur.x(2:end)>cur.x(1:end-1);
% compare emelemntwise wether i+1 element is greater than i element
% 1 if true, i.e., no discontinuity
% 0 if false, i.e., discontinuity

%%
sect = polyline;
i = 1;

%%
% Chops value both x and y in tree sections, i.e., at every swith,
% both from 1 to zero and from zero to 1
% Last element of previous section is always included in next section
while true
    j=find(ii~=ii(1),1,'first');
    if isempty(j)
        %exit the loop if there are no more loop-backs
        if i>1
            %if sections already started, add the last one
            sect(i)=cur;
        end
        break;
    end
    [sect(i) cur]=cur.chop(j,true); %true to repeat the boundary points
    ii(1:j-1)=[];%chop ii array accordingly
    i=i+1;
end

%%
% numel(sect) = 3 => sort
sect=sect.sort;

%%
%%%%%%%%%% Upper enelope %%%%%%%%%%%
%%%%%%%%%% As function %%%%%%%%%%%

[res newdots] = sect.upper_envelope(true);

%%
%%%%%%%%%% Line-by-line %%%%%%%%%%%
% stuff
obj = sect;
l=obj.len;%check that x and y are of the same size
fullinterval=exist('fullinterval');%when second arg given, full interval
obj=obj(l>0); %disregard all polylines of zero length

%%
% 500 x points now sorted
pt = sort(unique([obj.x]));

%%
% Interpolation for the same 500 sorted points in pt
% using each of the three polyline sections in sect
% outputs a 3x500 matrix
[intr extr] = obj.interpolate(pt);

%% NOOOOOOOO %%
% Running no fullinterval condition
%disregard points where at least one line is extrapolated
mask=sum(extr,1)>0;
intr(:,mask)=[];
pt(mask)=[];
n=sum(~mask);%number of point in the overlap region

%%
% Full interval condition
intr(extr) = -Inf;
n = numel(pt);

% Wherever extrapolation flag was equal to true, now -Inf in inter.

%%
%find lines on the top
maxintr=repmat(max(intr),size(intr,1),1);
top=intr==maxintr;

% Shape num sections by num pt (for the case when fullinterval is true)
% Each column is full with the max of sect.y for the column
% top indicates which row the value was taken from
%%
% Polilyne with one point - the first point!

res=polyline(pt(1),maxintr(1,1),'upper envelope');


%%
% intersections=polyline([],[],'intersection points');

%%
k0 = find(top(:, 1), 1, 'first'); % index of first nonzero element in col 1
% i.e., which polyline is on top at the 1st point

%%
for j=2:n
    k1=find(top(:,j),1,'first');%index of next top line
    if k1~=k0
        %check if there is an intersection point
        %between the lines:
        ln1=k0;ln2=k1; %intersections between these lines
        pt1=pt(j-1);pt2=pt(j); %which lies between these points
        [y1 extr1]=obj(ln1).interpolate([pt1 pt2]);
        [y2 extr2]=obj(ln2).interpolate([pt1 pt2]); %and these function values (maybe extrapolated)
        %check that neither is extrapolated in both points,
        %and that intersection point is inside the interval <= func values are different at the borders
        if all(~[extr1 extr2]) & all(abs(y1-y2)>0)
            %find the intersection point or points 
            while true
                pt3=fzero(@(x) obj(ln2).interpolate(x)-obj(ln1).interpolate(x),[pt1 pt2]);
                pt3f=obj(ln1).interpolate(pt3);
                %check if there are lines above the found intersection
                [intr2 exrt2]=obj.interpolate(pt3);%interpolate all lines in the new point
                intr2(exrt2)=-Inf; %disregard the extrapolated points
                maxintr2=repmat(max(intr2),size(intr2,1),1);
                ln3=find(intr2==maxintr2,1,'first');
                if ln3==ln1 | ln3==ln2
                    %there are no other functions above!
                    %add the intersection point
                    res=res.inset(pt3,pt3f,res.len);
                    
                    %intersections=intersections.inset(pt3,pt3f); %inset in the end
                    
                    %maybe there are some more intersections before next point?
                    if ln2==k1
                        %no actually, because the left line is the one we started with
                        break;
                    else
                        %indeed, so update the interval of new search
                        ln1=ln2;
                        pt1=pt3;
                        ln2=k1;
                        pt2=pt(j);
                    end
                else
                    %there is line ln3 above the found intersection point
                    %so, it is not on the upper envelope
                    %need to search again
                    ln2=ln3;%new candidate
                    pt2=pt3;%new border
                end
            end
        end
    end
    if any(abs(obj(k1).x-pt(j))<eps) || j==n
    % if ismember(pt(j),obj(k1).x) || j==n
        %add new grid point to the end
        res=res.inset(pt(j),maxintr(1,j)); %inset in the end
    end
    k0=k1;%next step
end


%%
indexremoved = obj.diff(res, 10);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Back to retirement_model.m solve_dcegm line 149ff.

% lable
value(1,it).label=sprintf('VF in period %d, choice %d (1=working), cleaned',it,1);

%%
newpolicy=[];

%%
% Loop over new points, for this case only one new point
newpolicy(1,1) = newdots.x(1);

%%
% Index of first section of points in old policy polyline.x 
% smaller than new point
j = find(policy(1, it).x < newdots.x(1));

% it is not monotonically increasing, there are some indexes missing
% since there was the discontinuity there before

%%
% All indexes of points that were deleted are dropped
j = j(~ismember(j, indxdel));

%%
% Last not deleted point that was member of original polyline x 
% before new point
j = max(j);
% index 14

%%
% Get another x point as the interpolation at the new point
% between point with index 14 and next point of original policy.x
newpolicy(1, 2) = interp1(policy(1, it).x(j:j+1), policy(1, it).y(j:j+1), newdots.x(1), 'linear');

%%
% Same for point greater / to the right
j = find(policy(1, it).x > newdots.x(1));
j = j(~ismember(j, indxdel));
j = min(j);
newpolicy(1, 3) = interp1(policy(1, it).x(j-1:j), policy(1, it).y(j-1:j), newdots.x(1), 'linear');

%%
% Remove points deleted in dc step from original polyline
policy(1, it) = policy(1, it).thinout(indxdel);

%%
%
j = find(policy(1, it).x > newpolicy(1,1), 1, 'first');

%%
policy(1,it)=policy(1,it).inset(newpolicy(1,1)-1e3*eps,newpolicy(1,2),j-1);
policy(1,it)=policy(1,it).inset(newpolicy(1,1),newpolicy(1,3),j);

%%%%%%%%%%% END %%%%%%%%%%

%%
%%%%%%%%%%% Function definitions %%%%%%%%%%%%%%%

function u=util(consumption,working) %utility

    u=(consumption.^(1-1.95)-1)/(1-1.95);

    u=u-0.35*(working==1);
end %util

function mu=mutil(consumption) %marginal utility
    if 1.95==1
        mu=1./consumption;
    else
        mu= consumption.^(-1.95);
    end
end %mutil

function cons=imutil(mutil) %inverse marginal utility
    if 1.95 == 1
        cons=1./mutil;
    else
        cons= mutil.^(-1/1.95);
    end
end %imutil

function mw1=mbudget(it,savings,shocks,working) 
    %derivative of wealth in t+1 w.r.t. savings
    %inputs and outputs as above
    mw1=repmat(1+0.05,size(shocks,1),size(savings,2));
end %mbudget

%Calculator of value functions that uses the analytical part in credit constrained region
function res =value_function(working,it,x, value)
    %interpolates value function at period t=it using analytical part
    %working =1 for workers, =2 for retirees
    res=nan(size(x)); %output of the same size as x
    mask=x<value(working,it).x(2); %all points in credit constrained region
    mask=mask | it==25; %in the terminal period all points are in the constrained region
    res(mask)=util(x(mask),working)+0.95*value(working,it).y(1); %the first value in me.value(working,it) is EV from zero savings!
    res(~mask)=value(working,it).interpolate(x(~mask));
end

function res=chpr(x, lambda)
    %choice probability of the first row in multirow matrix
    mx=max(x,[],1);
    mxx=x-repmat(mx,size(x,1),1);
    res=exp(mxx(1,:)/lambda)./sum(exp(mxx/lambda),1);
end

%Logsum and choice probability calculators, assume numel(x)=2
function res=logsum(x, lambda)
    %logsum by columns
    mx=max(x,[],1);
    mxx=x-repmat(mx,size(x,1),1);
    res=mx+lambda*log(sum(exp(mxx/lambda),1));
end

function w=income(it,shock) %income in period it with given normal shock
    %assume it=1 is age=20
    age=it+19;
    w=exp(0.75 + 0.04*age - 0.0002*age.*age + shock);
end %income


function w1=budget(it, savings,shocks,working) 
    %wealth in period t+1, where it=t
    %inputs: savings = 1x(ngridm) row vector of savings
    %				 shocks = (expn)x1 column vector of shocks
    %output: w1 = (expn)x(ngridm) matrix of all possible next period wealths
    w1=ones(size(shocks,1),1)*savings(1,:)*(1+0.05)+ ...
         (working==1)*income(it+1,shocks(:,1))*ones(1,size(savings,2));
end %budget

function vfres=vfcalc(it,x, value)
    vfres=nan(size(x)); %output of the same size as x
    mask=x<value(it).x(2); %all points in credit constrained region
    mask=mask | it==25; %in the terminal period all points are in the constrained region
    vfres(mask)=util(x(mask),2)+0.95*value(it).y(1); %the first value in me.value(working,it) is EV from zero savings!
    vfres(~mask)=value(it).interpolate(x(~mask));

end