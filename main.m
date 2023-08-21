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
x = (0.99*x_center+0.01*xstar)/norm((0.99*x_center+0.01*xstar));

[manifold.dist(x, x_center), radius_max]

for i=1:nIter
    i
    f(x)
    gradx = mgrad(x);
    
    v = linear_max_oracle_v2(-gradx, x, radius_max, x_center, manifold);
    
%     v = linear_max_oracle(-gradx, x, radius_max, x_center, manifold);
    accuracy(i) = -manifold.inner(x, gradx, manifold.log(x, v));
    
    step_size = -manifold.inner(x, gradx, manifold.log(x, v)) / (L*manifold.dist(x, v)^2);
    step_size = min(step_size, 1);
    x = manifold.exp(x, manifold.log(x, v), step_size);
    
%     x = line_search(x, v, f, i, manifold);
end
v = linear_max_oracle_v2(-gradx, x, radius_max, x_center, manifold);
accuracy(end) = -manifold.inner(x, gradx, manifold.log(x, v));

%% 
% semilogy(accuracy(1:1e3)-min(accuracy))
close all

fig = figure;
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
xlabel(han,'\#Gradient Oracle Calls', 'interpreter', 'latex');

set(gca, 'fontsize', fs)

%     % plot the manifold
%     [X,Y,Z] = sphere(100);
%     surf(X,Y,Z)
%     hold on
%     r = 1;
%     list_point = [];
%     for theta = 0:0.01:2*pi
%         theta
%         for phi = 0:0.01:pi
%             point = [sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta)];
%             
%             if(dist(point, x_center) < radius_max)
%                 list_point = [list_point ; r*point];
%             end
%                 
%         end
%     end
%     plot3(list_point(:,1),list_point(:,2),list_point(:,3),'.r');
%     plot3(x_center(1),x_center(2),x_center(3),'.g','markersize',15)
%     hold on
%     % Plot grad direction and x0
%     vecGrad = [];
%     vecx0 = [];
%     for rho = -3:0.01:3
%         vec1 = x+rho*gradx/norm(gradx);
%         vecGrad = [vecGrad;vec1'];
%         vec2 = x + rho*(manifold.log(x,x_center))/norm(manifold.log(x,x_center));
%         vecx0 = [vecx0; vec2'];
%     end
%     plot3(vecGrad(:,1),vecGrad(:,2),vecGrad(:,3),'.')
%     plot3(vecx0(:,1),vecx0(:,2),vecx0(:,3),'.')
    
    
