function out = solve_sincoseq(a,b,c)
% solve a*sin(x)+b*cos(x)=c
% return the positive solution

middle1 = (b-sqrt(a^2+b^2-c^2))/(a+c);
middle2 = (b+sqrt(a^2+b^2-c^2))/(a+c);
out = [2*atan(middle1), 2*atan(middle2)];

