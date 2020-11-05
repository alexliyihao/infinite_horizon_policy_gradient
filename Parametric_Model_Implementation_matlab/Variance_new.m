function [Mean,Bias, Variance,MSE] = Variance_new(M, N, Theta)
m = length(Theta);
Policy_True = True_Policy(Theta);
metric = zeros(m,N);
for i = 1 : N
    [State, Action] = Generate_MC(M,Theta);
    policy_gradient = Policy(Theta,Action,State);
    metric(:,i) = policy_gradient;
end
Mean = mean(metric,2);
Variance = var(metric',1)';
Bias = Mean - Policy_True;
MSE = Bias.^2 + Variance;