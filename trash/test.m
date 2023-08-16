% test

rng(1)
d = 200;
n = 100;
nIter = 100;
radius_ratio = 0.9;

A = randn(n,d);
A = A'*A;
A = A/norm(A);
accuracy = zeros(1, nIter+1);
 
% Create the problem structure.
manifold = spherefactory(n);
 
% Define the problem cost function and its derivatives.
xstar = ones(d, 1);
xstar = xstar / norm(xstar);
f = @(x) 0.5*(x-xstar)'*A*(x-xstar);
egrad = @(x) A*(x-xstar);
mgrad = @(x) manifold.egrad2rgrad(x, egrad(x));

x_center = abs(rand(d,1));
x_center = x_center/norm(x_center);

radius_max = manifold.dist(x_center,xstar)*radius_ratio;
setFunction = @(x) manifold.dist(x_center, x)<=radius_max;

L = norm(A);
x = x_center;

x = rand(d,1); x = x/norm(x);
w = rand(d,1); w = manifold.retr(x, w);



v = linear_max_oracle_v2(w, x, radius_max, x_center, manifold);