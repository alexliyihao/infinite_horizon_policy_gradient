function [coefficient] = compute_all_coeff(State,n,Action,Theta)
m = length(Action);
G = zeros(6,m);
coefficient = zeros(n + 1,6);
for i = 1 : m
    gradient = cal_f_a_s(Theta,Action(i),State(i));
    G(:,i) = gradient;
end
for i = 1 : 6
    y = G(i,:)';
    q = Optimumq(y,State,n);
    coefficient(:,i) = q;
end