function [Policy_gradient] = Policy_baseline(Theta,Action, State)
m = length(Action);
Policy_gradient = zeros(6,1);
r_i = zeros(1,m);
for i = 1 : m
    if State(i) == 1
            r = 0.0005;
    else
        if State(i) == 6
            r = 1;
        else
            r = 0;
        end
    end
    r_i(i) = r;
end
s2 = sum(r_i);
for i = 1 : m
    s1 = cal_f_a_s(Theta,Action(i),State(i));
    Policy_gradient = Policy_gradient + s1 * s2;
    s2 = s2 - r_i(i);
end
Policy_gradient = Policy_gradient / m;
    