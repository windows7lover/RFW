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

max_val = -inf;

delta_phi = 0.001;


for phi = -pi:delta_phi:pi
    v_plus = alpha_plus(p(phi))*p(phi);
    v_minus = alpha_minus(p(phi))*p(phi);
    
    if (w'*v_plus > max_val) && (~any(isnan(v_plus)))
        max_val = w'*v_plus;
        best_phi = phi;
        best_alpha = @(p) alpha_plus(p);
    end
    if (w'*v_minus > max_val) && (~any(isnan(v_minus)))
        max_val = w'*v_minus;
        best_phi = phi;
        best_alpha = @(p) alpha_minus(p);
    end
    
end


fhandle = @(phi) w'*(best_alpha(p(phi))*p(phi));
options = optimset('TolX',1e-8);
best_phi = fminbnd(fhandle, best_phi-delta_phi, best_phi+delta_phi,options);
v = manifold.exp(x, p(best_phi), best_alpha(p(best_phi)));

    function best_alpha = alpha_plus(p)
        best_alpha = max(solve_sincoseq(x0'*x,x0'*p,(1-2*sin(R/2)^2)));
    end


    function best_alpha = alpha_minus(p)
        best_alpha = min(solve_sincoseq(x0'*x,x0'*p,(1-2*sin(R/2)^2)));
    end

    
    function plot_alpha_sphere(p)
        
    % plot the manifold
    [X,Y,Z] = sphere(100);
    surf(X,Y,Z)
    hold on
    list_point = [];
    for theta = 0:0.01:2*pi
        for phi = 0:0.01:pi
            point = [sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta)];
            
            if(dist(point, x0) < R)
                list_point = [list_point ; point];
            end
                
        end
    end
    plot3(list_point(:,1),list_point(:,2),list_point(:,3),'.r');
    plot3(x0(1),x0(2),x0(3),'.g','markersize',15)
    hold on
    
    % Plot alpha
    vecalpha = [];
    for alpha = 0:0.001:pi
        vec2=manifold.exp(x,p,alpha);
        vecalpha = [vecalpha; vec2'];
    end
    plot3(vecalpha(:,1),vecalpha(:,2),vecalpha(:,3),'.')

    end


end


