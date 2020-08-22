function [gradient] = cal_f_a_s(Theta,action, s)
%calculate pi(a|s)
gradient = zeros(6,1);
if action == 0
    gradient(s) = 1 / Theta(s);
else
    gradient(s) = -1 / (1 - Theta(s));
end