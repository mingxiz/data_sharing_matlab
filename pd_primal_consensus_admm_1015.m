function sol = pd_primal_consensus_admm_1015(X, y, block_num, lambda, max_iter, tol, time_limit)

%%  Consensus ADMM
%%  Solver for estimation problem with n>>p  
%%  min (1/2)* (y-X*beta)^T(y-X*beta)? data could not be shared across solver)

%%  Written for MATLAB_R2016b
%%  Written by Mingxi Zhu, Graduate School of Business,
%%  Stanford University, Stanford, CA 94305.
%%  June, 2021.
%%  mingxiz@stanford.edu

    pd_time_start = tic;
    pd_preparing_time_start = tic;

    % starting point beta = 0;
    [n, p] = size(X);
    % initialize starting point
    x_k = zeros(p, block_num);
    z_k = zeros(p, 1);
    y_k = zeros(p, block_num);
    c = zeros(p, block_num);

    % Preparing matrix (with choleskey)
    % change block structure;
    order = 1:n;
    block_size = floor(n/block_num);
    store_R = cell(1, block_num);
    store_XTX = cell(1, block_num);
    for i_block = 1:block_num-1
        X_sub = X(order(((i_block - 1)*block_size + 1):i_block*block_size),:);
        c(:, i_block) = -X_sub'*y(order(((i_block - 1)*block_size + 1):i_block*block_size),1);
        store_XTX{i_block} = X_sub'*X_sub;
        store_R{i_block} = chol(store_XTX{i_block}+lambda*eye(p));
    end
    X_sub = X(order(((block_num - 1)*block_size + 1):n),:);
    c(:, block_num) = -X_sub'*y(order(((block_num - 1)*block_size + 1):n),1);
    store_XTX{block_num} =  X_sub'*X_sub;
    store_R{block_num} = chol(store_XTX{block_num}+lambda*eye(p));

    tol_temp_vec = zeros(block_num, 1);
    num_iter = 0;
    tol_temp = inf;
    %prepare z_k
    sub_model_time_1 = 0;
    %prepare right-hand-side
    sub_model_time_2 = 0;
    %update z_k
    sub_model_time_3 = 0;
    %update y_k
    sub_model_time_4 = 0;
    %solve linear system
    sub_solver_time = 0;
    pd_preparing_time = toc(pd_preparing_time_start);

    while tol_temp>tol
    
        sub_model_time_1_start = tic;    
        %change block-wise randomization
        x_block_index_perm = 1:block_num;
        %update x_k first
        lambda_z = lambda*z_k;
        sub_model_time_1 = toc(sub_model_time_1_start) + sub_model_time_1;
    
        for i = 1: block_num
        
            sub_model_time_2_start = tic;
            block_i = x_block_index_perm(1, i);
            R = store_R{block_i};
            b = -(c(:, block_i) + y_k(:, block_i) - lambda_z);
            sub_model_time_2 = toc(sub_model_time_2_start) + sub_model_time_2;
        
            sub_solver_time_start = tic;
            x_k(:, block_i) =  R\(R'\b);
            sub_solver_time = toc(sub_solver_time_start) + sub_solver_time;
        end
    
        sub_model_time_3_start = tic;
        z_k = 1/block_num*(sum(x_k,2)+1/lambda*sum(y_k,2));
        sub_model_time_3 = toc(sub_model_time_3_start) + sub_model_time_3;
    
        sub_model_time_4_start = tic;    
    
        for i = 1: block_num
            block_i = x_block_index_perm(1, i);
            diff = x_k(:, block_i)-z_k;
            tol_temp_vec(block_i, 1) = max(abs(diff));
            y_k(:, block_i) = y_k(:, block_i) + lambda*diff;
        end
    
        tol_temp = max(tol_temp_vec);
 
        num_iter = num_iter + 1;
        if num_iter == max_iter
            break
        end
        sub_model_time_4 = toc(sub_model_time_4_start) + sub_model_time_4;
    
        pd_crt_time = toc(pd_time_start);
        if pd_crt_time > time_limit
            break
        end
    end

    sol.beta = z_k;
    sol.y_k = y_k;
    sol.x_k = x_k;
    sol.tol = tol_temp;
    sol.total_time = toc(pd_time_start);
    sol.solver_time = sub_solver_time;
    sol.model_time_1 = sub_model_time_1;
    sol.model_time_2 = sub_model_time_2;
    sol.model_time_3 = sub_model_time_3;
    sol.model_time_4 = sub_model_time_4;
    sol.prepare_time = pd_preparing_time;
    sol.num_iter = num_iter;
