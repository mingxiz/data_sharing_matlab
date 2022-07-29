function sol = newton_log(A, b, time_limit, MAX_ITER)

    %% cite 
    %% http://www.stanford.edu/~boyd/papers/distr_opt_stat_learning_admm.html
    %% partially use the code provided in https://web.stanford.edu/~boyd/papers/admm/logreg-l1/logreg.html
    
    time_start = tic;
    alpha = 0.1;
    BETA  = 0.5;
    TOLERANCE = 1e-5;
    [m n] = size(A);
    I = eye(n+1);
    if exist('x0', 'var')
        x = x0;
    else
        x = zeros(n+1,1);
    end
    C = [-b -A];
    f = @(w) (sum(log(1 + exp(C*w))) );
    iter = 1;
    while iter <= MAX_ITER
        fx = f(x);
        g = C'*(exp(C*x)./(1 + exp(C*x))) ;
        H = C' * diag(exp(C*x)./(1 + exp(C*x)).^2) * C + TOLERANCE*I;
        dx = -H\g;   % Newton step
        dfx = g'*dx; % Newton decrement
        if abs(dfx) < TOLERANCE
            break;
        end
        % backtracking
        t = 1;
        while f(x + t*dx) > fx + alpha*t*dfx
            t = BETA*t;
        end
        x = x + t*dx;
        crt_time = toc(time_start);
        if crt_time>= time_limit
            break
        end
        iter = iter + 1;
    end
    sol.num_iter = iter;
    sol.beta = x;
end
