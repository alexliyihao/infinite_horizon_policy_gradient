function [X] = optimalw(Theta,Action,State)
initial_X = rand(36,1);
    options = optimset('GradObj', 'on', 'MaxIter', 50);
    [X] = ...
        fmincg (@(t)(Costfunction(Theta,Action, State,t)), ...
                  initial_X, options);
              X = reshape(X,6,6);
m = length(State);
cnt = zeros(6,1);
for i = 1:m
    cnt = cnt + X(:,State(i));     
end
expectations = cnt / m;
for i = 1 : 6
    X(i,:) = X(i,:) - expectations(i);
end
end