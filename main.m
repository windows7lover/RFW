% Generate the problem data.
rng(1)
d = 500;
n = 250;
nIter = 150;
radius_ratio = 0.9;

A = randn(n,d);
A = A'*A;
A = A/norm(A);
accuracy = zeros(1, nIter+1);
dualgap =  zeros(1, nIter+1);
 
% Create the problem structure.
manifold = spherefactory(d);
 

x_center = abs(rand(d,1));
x_center = x_center/norm(x_center);

% Define the problem cost function and its derivatives.
xstar = ones(d, 1);
% xstar = -x_center;
xstar = xstar / norm(xstar);
f = @(x) 0.5*(x-xstar)'*A*(x-xstar);
egrad = @(x) A*(x-xstar);
mgrad = @(x) manifold.egrad2rgrad(x, egrad(x));


radius_max = manifold.dist(x_center,xstar)*radius_ratio;
setFunction = @(x) manifold.dist(x_center, x)<=radius_max;

L = norm(A);
x = (0.99*x_center+0.01*xstar)/norm((0.99*x_center+0.01*xstar));

[manifold.dist(x, x_center), radius_max]

for i=1:nIter
    i
    f(x)
    gradx = mgrad(x);
    
    v = linear_max_oracle(-gradx, x, radius_max, x_center, manifold);
    
%     v = linear_max_oracle(-gradx, x, radius_max, x_center, manifold);
    accuracy(i) = -manifold.inner(x, gradx, manifold.log(x, v));
    dualgap(i) = f(x);
    
    step_size = -manifold.inner(x, gradx, manifold.log(x, v)) / (L*manifold.dist(x, v)^2);
    step_size = min(step_size, 1);
    x = manifold.exp(x, manifold.log(x, v), step_size);
    
    
%     x = line_search(x, v, f, i, manifold);
end
v = linear_max_oracle(-gradx, x, radius_max, x_center, manifold);
accuracy(end) = -manifold.inner(x, gradx, manifold.log(x, v));

%% 
% semilogy(accuracy(1:1e3)-min(accuracy))
% close all

figure
semilogy(real(accuracy))
hold on
semilogy(real(dualgap))

fig = figure('Renderer', 'painters', 'Position', [10 10 700 500]);
facealpha = 0.5;
fs = 16;
fs2 = 12;

subplot(1,2,1)
X = [1 1:20 20];
Y = [1e-7 accuracy(2:21) 1e-7];
fill(X, Y, [0 0.4470 0.7410],'FaceAlpha',facealpha)
hold on
X = [20 20:nIter nIter];
Y = [1e-7 accuracy(21:nIter+1) 1e-7];
fill(X, Y, [0.4660 0.6740 0.1880],'FaceAlpha',facealpha)
set(gca,'YScale','log','Xscale','log','fontsize',fs2)
loglog(0:nIter, accuracy, 'color', 'k', 'linewidth', 3)

subplot(1,2,2)
X = [0 0:20 20];
Y = [1e-7 accuracy(1:21) 1e-7];
fill(X, Y, [0 0.4470 0.7410],'FaceAlpha',facealpha)
hold on
X = [20 20:nIter nIter];
Y = [1e-7 accuracy(21:nIter+1) 1e-7];
fill(X, Y, [0.4660 0.6740 0.1880],'FaceAlpha',facealpha)
set(gca,'YScale','log','fontsize',fs2)
semilogy(0:nIter, accuracy, 'color', 'k', 'linewidth', 3)
set(gca,'ytick',[])

legend({'$\frac{1}{t}$ convergence' 'Linear Convergence' 'RFW Dual Gap'}, 'interpreter', 'latex', 'fontsize', 12)

han=axes(fig,'visible','off');
han.Title.Visible='off';
han.XLabel.Visible='on';
han.YLabel.Visible='on';
ylabel(han,'RFW Duality Gap', 'interpreter', 'latex');
set(gca, 'fontsize', fs)
xlabel(han,'\#Gradient Oracle Calls', 'interpreter', 'latex');
    
    
