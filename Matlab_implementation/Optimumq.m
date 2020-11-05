function [q] = Optimumq(y,State,n)
m = length(State) - 1;
x2 = base_function(State(2:m + 1),n);
x1 = base_function(State(1:m),n);
X = x2 - x1;
q = pinv(X' * X) * (X' * y);
