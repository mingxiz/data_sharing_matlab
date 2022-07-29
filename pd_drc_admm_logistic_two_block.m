function sol = pd_drc_admm_logistic_two_block(x, y, block_num, percentage, lambda_1, lambda_2, max_iter, tol, time_limit)

%%  DRC ADMM
%%  Solver for estimation problem with n>>p
%%  min sum(log(1+exp(-yi*xi*beta))) data precentage% permuted
%%  the first argument of beta is intersection

%%  Written for MATLAB_R2016b
%%  Written by Mingxi Zhu, Graduate School of Business,
%%  Stanford University, Stanford, CA 94305.
%%  July, 2021.
%%  mingxiz@stanford.edu


    drc_time_start = tic;
    drc_preparing_time_start = tic;


    % startiing point
    [n, p] = size(x);
    % initialilze x
    x = [ones(n, 1), x];
    x = y.*x;
    p = p + 1;


    % initliaze starting popint 
    A = x';
    % x_k each time use sub solver to create new vector
    x_k = 1/2*ones(n ,1);
    z_k = x_k;
    beta_k = zeros(p, 1);
    xi_k = zeros(n, 1);
    Az_k = A*z_k;

    tol_temp = inf;
    sub_model_prepare_time = 0;
    sub_t_time = 0;
    sub_beta_time = 0;
    num_iter = 0;

    % Preselect those who willing to provide data
    numelements = round(percentage*n);
    selected_indices = sort(randperm(n, numelements));
    block_size = floor(n/block_num);
    AAt_store = cell(1, block_num);
    A_sub_store = cell(1, block_num);
    changed_index = cell(1, block_num);
    changed_index_sub = cell(1, block_num);
    unchanged_index_sub = cell(1, block_num);
    unchanged_index = cell(1, block_num);
    changed_index_num = cell(1, block_num);
    for i = 1: block_num-1
        changed_index{1, i} = selected_indices(selected_indices<=i*block_size & selected_indices>=((i - 1)*block_size+1));
        changed_index_sub{1, i} = changed_index{1, i} - (i-1)*block_size;
        unchanged_index_sub{1, i} = setdiff(1:block_size, changed_index_sub{1, i});
        unchanged_index{1, i} = (i - 1)*block_size + unchanged_index_sub{1, i};
        changed_index_num{1, i} = length(changed_index{1, i});
        A_local = A(:, ((i - 1)*block_size+1):i*block_size);
        A_local(:, changed_index_sub{1, i}) = [];
        A_sub_store{1, i} = A_local;
        AAt_store{1, i} = A_local*A_local';
    end
    changed_index{1, block_num} = selected_indices(selected_indices>=((block_num - 1)*block_size+1));
    changed_index_sub{1, block_num} = changed_index{1, block_num} - (block_num-1)*block_size;
    changed_index_num{1, block_num} = length(changed_index{1, block_num});
    A_local = A(:, ((block_num - 1)*block_size+1):end);
    curr_size = size(A_local, 2);
    unchanged_index_sub{1, block_num} = setdiff(1:curr_size, changed_index_sub{1, block_num});
    unchanged_index{1, block_num} = (block_num - 1)*block_size + unchanged_index_sub{1, block_num};
    A_local(:, changed_index_sub{1, block_num}) = [];
    A_sub_store{1, block_num} = A_local;
	AAt_store{1, block_num} = A_local*A_local';
    drc_preparing_time = toc(drc_preparing_time_start);
 
    while tol_temp > tol
    
        sub_t_time_start = tic;
        % randperm select idx
        index_perm = randperm(numelements);
        selected_indices_perm = selected_indices(index_perm);
        crt_start = 1;
        crt_end = changed_index_num{1};
        crt_perm = cell(1, block_num);
        for i = 1: block_num - 1
            crt_perm{1, i} = selected_indices_perm(crt_start:crt_end);
            crt_start = crt_end + 1;
            crt_end = crt_end + changed_index_num{i+1};
        end
        crt_perm{1, block_num} = selected_indices_perm(crt_start:crt_end);

    
        x_visited_block = 1;
        update_order = randperm(block_num, block_num);
        sub_model_prepare_time = toc(sub_t_time_start) + sub_model_prepare_time;
        
        while x_visited_block <= block_num
            
           % select current blox index
            x_crt_block =  update_order(x_visited_block);
            curr_x_global_index = crt_perm{1, x_crt_block};
            curr_x_index = [unchanged_index{x_crt_block}, curr_x_global_index];
            % update x_k
            curr_c_vec = xi_k(curr_x_index) + lambda_2*z_k(curr_x_index); 
            curr_x_k = admm_subsolver_vpa(curr_c_vec, lambda_2);
            x_k(curr_x_index) = curr_x_k;
            
            % update z_k
            A_sub_global = A(:, curr_x_global_index);
            A_sub_A_sub_t = AAt_store{1, x_crt_block} + A_sub_global*A_sub_global';
            curr_A_sub = [A_sub_store{1, x_crt_block}, A_sub_global];
            curr_z_k = z_k(curr_x_index);
            A_subzi = curr_A_sub*curr_z_k;
            diff_A_sub = Az_k - A_subzi;
            c_curr_sub = - lambda_1*curr_A_sub'*diff_A_sub - xi_k(curr_x_index) + curr_A_sub'*beta_k + lambda_2*curr_x_k;
            c_curr_sub_p = - lambda_1*A_sub_A_sub_t*diff_A_sub  - curr_A_sub*xi_k(curr_x_index) + A_sub_A_sub_t*beta_k + lambda_2*curr_A_sub*curr_x_k;
            curr_lhs = lambda_2*eye(p) + lambda_1*A_sub_A_sub_t;
            curr_ui = curr_lhs\c_curr_sub_p;
            z_k(curr_x_index) = (1/lambda_2)*c_curr_sub - (lambda_1/lambda_2)*curr_A_sub'*curr_ui;
            Az_k = Az_k - A_subzi + curr_ui;
        
            x_visited_block = x_visited_block + 1;
        end
        sub_t_time = sub_t_time + toc(sub_t_time_start);
    
        sub_beta_time_start = tic;
        diff_xz = x_k - z_k;
        tol_temp = max(max(abs(Az_k)), max(abs(diff_xz)));
        beta_k = beta_k - lambda_1*Az_k;
        xi_k = xi_k - lambda_2*diff_xz;
    
        num_iter = num_iter+1;
        if num_iter == max_iter
            break
        end
        drc_crt_time = toc(drc_time_start);
        if drc_crt_time > time_limit
            break
        end
        sub_beta_time = sub_beta_time + toc(sub_beta_time_start);
  
    end 

    drc_time = toc(drc_time_start);

    sol.beta = -beta_k;
    sol.x = x_k;
    sol.z = z_k;
    sol.total_time = drc_time;
    sol.drc_preparing_time = drc_preparing_time;
    sol.sub_t_time = sub_t_time;
    sol.sub_beta_time = sub_beta_time;
    sol.num_iter = num_iter;
    sol.prim_tol = tol_temp;
    %sol.store_lambda = store_y;

    
    