function v = linear_max_oracle_v2(w, x, R, x0, manifold)

% Solve the problem
% max w^Tz : z \in exp_x^{-1}(C),
% where C := {x:dist(x,x_0)\leq R}

% Define the two basis vector of the subspace u_1 and u_2
u1 = manifold.log(x, x0);
u1 = u1/norm(u1);
u2 = w-u1*(u1'*w);
u2 = u2/norm(u2);

[manifold.dist(x,x0), R]

p_plus = @(gamma) gamma*u1 + (1-gamma^2)*u2;
p_minus = @(gamma) gamma*u1 - (1-gamma^2)*u2;

max_val = -inf;

% gamma = 1;
% plot_alpha_sphere(p_plus(gamma))
% figure
% plot_alpha(p_plus(gamma))
% alpha(p_plus(gamma))

best_gamma = [];
% Initial trial for gamma
delta_gamma = 0.1;
for gamma = -1:delta_gamma:1
    v_plusplus = alpha_plus(p_plus(gamma))*p_plus(gamma);
    v_minusplus = alpha_minus(p_plus(gamma))*p_plus(gamma);
    
    v_plusminus = alpha_plus(p_minus(gamma))*p_minus(gamma);
    v_minusminus = alpha_minus(p_minus(gamma))*p_minus(gamma);
    
    if (w'*v_plusplus > max_val) && (~any(isnan(v_plusplus)))
        max_val = w'*v_plusplus;
        best_gamma = gamma;
        best_alpha = @(p) alpha_plus(p);
        best_p = p_plus;
        best_v = v_plusplus;
    end
    
    
    if (w'*v_minusplus > max_val) && ~any(isnan(v_minusplus))
        max_val = w'*v_minusplus;
        best_gamma = gamma;
        best_alpha = @(p) alpha_minus(p);
        best_p = p_plus;
        best_v = v_minusplus;
    end
    
    if (w'*v_plusminus > max_val) && ~any(isnan(v_plusminus))
        max_val = w'*v_plusminus;
        best_gamma = gamma;
        best_alpha = @(p) alpha_plus(p);
        best_p = p_minus;
        best_v = v_plusminus;
    end
    
    if (w'*v_minusminus > max_val) && ~any(isnan(v_minusminus))
        max_val = w'*v_minusminus;
        best_gamma = gamma;
        best_alpha = @(p) alpha_minus(p);
        best_p = p_minus;
        best_v = v_minusminus;
    end
    
end

fhandle = @(gamma) w'*(best_alpha(best_p(gamma))*best_p(gamma));
options = optimset('TolX',1e-8);
best_gamma = fminbnd(fhandle, best_gamma-delta_gamma, best_gamma+delta_gamma,options);

v = manifold.exp(x, best_p(best_gamma), best_alpha(best_p(best_gamma)));

% if(manifold.dist(v,x0) > R)
%     display('wtf')
% end

    function best_alpha = alpha_plus(p)
        
        fun_alpha = @(alpha) manifold.dist( manifold.exp(x,p,alpha), x0)-R;
        
        fun_fzero = @(alpha) fun_alpha(alpha) + (fun_alpha(alpha)>0);
        try
            best_alpha=fzero(fun_alpha, [0,pi]);
            while(fun_alpha(best_alpha)>0)
                best_alpha = best_alpha-1e-8;
            end
        catch
            best_alpha = NaN;
        end
        
%         best_alpha = NaN;
%         best_fun_val = inf;
%         
%         for alpha_try=0:0.001:pi
%             funval = fun_alpha(alpha_try);
%             if abs(funval) < best_fun_val
%                 best_fun_val = abs(funval);
%                 best_alpha = alpha_try;
%             end
%         end
        
%         if best_fun_val > 1e-1 % No intersection found
%             best_alpha=0;
%         end
        
%         alpha_out = fzero(fun_alpha, [0,2*R]);
    end


    function best_alpha = alpha_minus(p)
        fun_alpha = @(alpha) manifold.dist( manifold.exp(x,p,alpha), x0)-R;
%         best_alpha=fzero(fun_alpha, [-pi,0]);
        try
            best_alpha=fzero(fun_alpha, [0,pi]);
        catch
            best_alpha = NaN;
        end
        
%         best_alpha = NaN;
%         best_fun_val = inf;
%         
%         for alpha_try=0:-0.1:-pi
%             funval = fun_alpha(alpha_try);
%             if abs(funval) < best_fun_val
%                 best_fun_val = abs(funval);
%                 best_alpha = alpha_try;
%             end
%         end
    end

    function plot_alpha(p)
        
        alpha_vec = -pi:0.1:pi;
        fun_alpha = @(alpha) manifold.dist( manifold.exp(x,p,alpha), x0)-R;
        
        for i=1:length(alpha_vec)
            alpha_try = alpha_vec(i);
            funval(i) = fun_alpha(alpha_try);
        end
        plot(alpha_vec, funval)
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


