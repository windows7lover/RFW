% Generate the problem data.
d = 50;
n = 25; % Slower convergence when n < d -> non-strongly convex case
nIter = 500;
dualgap =  zeros(1, nIter+1);

% Manifold: this code uses manop, see https://www.manopt.org
% Boumal, Nicolas, et al. "Manopt, a Matlab toolbox for optimization on 
% manifolds." The Journal of Machine Learning Research 15.1 (2014).
manifold = spherefactory(d); 

% Define the problem cost function and its derivatives.
A = randn(n,d);
A = A'*A;
A = A/norm(A);
xstar = manifold.rand();
f = @(x) 0.5*(x-xstar)'*A*(x-xstar);
egrad = @(x) A*(x-xstar);
mgrad = @(x) manifold.egrad2rgrad(x, egrad(x));
L = norm(A); % Worst case

% Create the problem set
x_center = manifold.rand();
radius_ratio = 0.9;
radius_max = manifold.dist(x_center,xstar)*radius_ratio;
setFunction = @(x) manifold.dist(x_center, x)<=radius_max;

% Find an initial point in the set
% Avoid a bug if we start exactly at c_center, implementation issue
start_step_size = 0.1;
x = manifold.exp(x_center, -mgrad(x_center), start_step_size);
while(manifold.dist(x, x_center) > radius_max)
    start_step_size = start_step_size/2;
    x = manifold.exp(x_center, -mgrad(x_center), start_step_size);
end


% Main loop RFW
for i=1:nIter
    
    gradx = mgrad(x);
    
    v = linear_max_oracle(-gradx, x, radius_max, x_center, manifold);
    dualgap(i) = -manifold.inner(x, gradx, manifold.log(x, v));
    
    step_size = -manifold.inner(x, gradx, manifold.log(x, v)) / (L*manifold.dist(x, v)^2);
    step_size = min(step_size, 1);
    x = manifold.exp(x, manifold.log(x, v), step_size);
end
v = linear_max_oracle(-gradx, x, radius_max, x_center, manifold);
dualgap(end) = -manifold.inner(x, gradx, manifold.log(x, v));

%% 
figure
% Max at eps, otherwise the result is numerically meaningless
semilogy(1:length(dualgap), max(dualgap,eps))
legend({'FW Dual Gap'})

    
    
