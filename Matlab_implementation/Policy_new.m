function [policy_gradient] = Policy_new(Theta,Action,State,n)
m = length(Action);
metric = zeros(m,6);
coefficient = compute_all_coeff(State,n,Action,Theta);
w = base_function(State(1:m),n) * coefficient;
for i = 1 : m
    f = cal_f_a_s(Theta,Action(i), State(i))';
    if State(i) == 1
        r = 0.0005;
    else
        if State(i) == 6
            r = 1;
        else
            r = 0;
        end
    end
    metric(i,:) = (w(i,:) + f) * r;
end
policy_gradient = mean(metric, 1);
