function [J,Gradient] = Costfunction(Theta,Action, State,X)
m = length(Action);
J = 0;
Gradient = zeros(6);
X = reshape(X,6,6);
for i = 1 : m
    state1 = State(i);
    state2 = State(i + 1); 
    item = X(:,state2) - X(:,state1) - cal_f_a_s(Theta,Action(i), State(i));
    J = J + item' * item;
    if state1 ~= state2
        Gradient(:,state2) = Gradient(:,state2) + 2 * item;
        Gradient(:,state1) = Gradient(:,state1) - 2 * item;
    end
end
Gradient = Gradient(:);
        
    
    
   