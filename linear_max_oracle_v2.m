function v = linear_max_oracle_v2(w, x, R, x0, manifold)

% Solve the problem
% max w^Tz : z \in exp_x^{-1}(C),
% where C := {x:dist(x,x_0)\leq R}
% Assume that the ball has radius = 1!

% Define the two basis vector of the subspace u_1 and u_2

u1 = manifold.log(x, x0);
u1 = u1/norm(u1);
u2 = w-u1*(u1'*w);
u2 = u2/norm(u2);
p = @(phi) cos(phi)*u1 + sin(phi)*u2;

alpha = @(p) solve_sincoseq(x0'*x,x0'*p,(1-2*sin(R/2)^2));

fhandle = @(phi) -w'*(alpha(p(phi))*p(phi));
options = optimset('TolX',eps);
[best_phi] = fminbnd(fhandle, -pi, pi, options);
v = manifold.exp(x, p(best_phi), alpha(p(best_phi)));


