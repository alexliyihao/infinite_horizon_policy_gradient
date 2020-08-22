function [Variance,Policy_Grad_True] = True_Policy_baseline(Theta,K)
Policy_Grad_True = zeros(6,K);
B = 1e4;
for i = 1 : K
    [State, Action] = Generate_MC(B,Theta);
    Policy_Grad_True(:,i) = Policy_baseline(Theta,Action, State);
end
Variance = var(Policy_Grad_True',1) / K;
Policy_Grad_True = mean(Policy_Grad_True,2);
end