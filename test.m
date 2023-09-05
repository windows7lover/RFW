function out = test(a, b, c, phi)

out = -sin(phi)*atan((sqrt(a^2 + b(phi)^2 - c^2) + b(phi))/(a + c));