function [basis] = base_function(x,n)
m = length(x);
basis = zeros(m,n + 1);

for i = 1:n + 1
    basis(:,i) = x'.^(i - 1);
end
