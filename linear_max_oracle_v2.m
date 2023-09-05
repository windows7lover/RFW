function v = linear_max_oracle_v2(w, x, R, x0, manifold)

% Solve the problem
% max w^Tz : z \in exp_x^{-1}(C),
% where C := {x:dist(x,x_0)\leq R}
% Assume that the ball has radius = 1!

% Define the two basis vector of the subspace u_1 and u_2

% u1 = manifold.log(x, x0);
u1 = w;
u1 = u1/norm(u1);
u2 = manifold.log(x, x0);
u2 = u2-u1*(u1'*u2);
u2 = u2/norm(u2);
p = @(phi) cos(phi)*u1 + sin(phi)*u2;

alpha = @(p) solve_sincoseq(x0'*x,x0'*p,(1-2*sin(R/2)^2));

% fhandle = @(phi) -w'*(alpha(p(phi))*p(phi));
fhandle = @(phi) -(alpha(p(phi))*cos(phi)); %same minimum since w*u2 = 0.
options = optimset('TolX',eps);
[best_phi] = fminbnd(fhandle, -pi, pi, options);
v = manifold.exp(x, p(best_phi), alpha(p(best_phi)));



% fhandle2 = @(phi) -(alpha(p(phi))*(phi)); 
% [best_phi2] = fminbnd(fhandle2, -pi, pi, options);
% 
% [best_phi, best_phi2]

% figure
% fplot(@(phi) -alpha(p(phi)), [-2*pi, 2*pi])
% 
% figure
% fplot(@(phi) -(alpha(p(phi))*cos(phi)), [-pi, pi])


% figure
% fplot( @(phi) -test(x0'*x, @(x) x0'*p(x), (1-2*sin(R/2)^2), phi) / ( -test(x0'*x, @(x) x0'*p(x), (1-2*sin(R/2)^2), 1)), [-pi,pi])
% hold on
% dx = 1e5*eps
% fplot(@(phi)((fhandle(phi+dx)-fhandle(phi))/dx) / ((fhandle(1+dx)-fhandle(1))/dx), [-pi, pi])
% 
% 1



