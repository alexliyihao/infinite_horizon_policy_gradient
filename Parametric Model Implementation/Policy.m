function [policy_gradient] = Policy(Theta,Action,State)
m = length(Action);
metric = zeros(6,m);
w = optimalw(Theta,Action,State);
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
    metric(:,i) = (w(:,State(i)) + f') * r;
end
policy_gradient = mean(metric, 2);

