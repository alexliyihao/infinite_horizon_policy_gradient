function [Mean,Bias,Variance,MSE] = Variance_control(M, N, Theta)
m = length(Theta);
Policy_True = True_Policy(Theta);
metric = zeros(m,N);
b = 0;
for i = 1 : N
    [State, Action] = Generate_MC(M,Theta);
    r_i = zeros(1,m);
    for j = 1 : m
        if State(j) == 1
                r = 0.005;
        else
            if State(j) == 6
                r = 1;
            else
                r = 0;
            end
        end
        r_i(j) = r;
    end
    s2 = sum(r_i);
    b = b + s2 / m;
end
b = b / N;
for i = 1 : N
    [State, Action] = Generate_MC(M,Theta);
    r_i = zeros(1,m);
    for j = 1 : m
        if State(j) == 1
                r = 0.005;
        else
            if State(j) == 6
                r = 1;
            else
                r = 0;
            end
        end
        r_i(j) = r;
    end
    s2 = sum(r_i);
    Policy_gradient = zeros(6,1);
    for j = 1 : m
        s1 = cal_f_a_s(Theta,Action(j),State(j));
        Policy_gradient = Policy_gradient + s1 * (s2 - b);
        s2 = s2 - r_i(j);
    end
    Policy_gradient = Policy_gradient / m;
    metric(:,i) = Policy_gradient;
end
Mean = mean(metric,2);
Variance = var(metric',1)';
Bias = Mean - Policy_True;
MSE = Bias.^2 + Variance;
end