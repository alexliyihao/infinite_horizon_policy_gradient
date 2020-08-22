function [Policy_Grad_True] = True_Policy(Theta)
[State, Action] = Generate_MC(1e6,Theta);
Policy_Grad_True = Policy(Theta,Action,State);
end