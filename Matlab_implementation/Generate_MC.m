function [State, Action] = Generate_MC(N,Theta)
Action = zeros(1,N);
State = zeros(1,N + 1);
s_old = 1;
for i = 1 : N + 1000
    prob1 = rand(1);
    if prob1 < Theta(s_old)
        action = 0;
    else
        action = 1;
    end
    if i > 1000
        Action(i - 1000) = action;
    end
    if action == 0
        if s_old == 1
            s_new = s_old;
        else
            s_new = s_old - 1;
        end
    else
        prob2 = rand(1);
        if s_old == 6
            if prob2 < 0.6
                s_new = s_old;
            else
                s_new = s_old - 1;
            end
        else 
            if s_old == 1
                if prob2 < 0.6
                    s_new = s_old + 1;
                else
                    s_new = s_old;
                end
            else
                if prob2 < 0.3
                    s_new = s_old;
                else 
                    if prob2 < 0.9
                        s_new = s_old + 1;
                    else
                        s_new = s_old - 1;
                    end
                end
            end
        end
    end
    if i > 1000
        State(i - 1000) = s_old;
    end
    s_old = s_new;
end
State(N + 1) = s_new;
                  
            
end