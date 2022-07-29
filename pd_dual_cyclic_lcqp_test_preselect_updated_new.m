function sol = pd_dual_cyclic_lcqp_test_preselect_updated_new(c, A, block_num, percentage, lambda, tol, max_iter, time_limit)

%%  cyclic ADMM with data sharing
%%  Solver for following Quadratic Optimization Problem 
%%  min 1/2*x^T x + c^T x
%%  s.t.  A x = 0
%%  Input (H,c,A,b,block_num, dual penalty parameter lambda,tolerence of ||Ax-b||_inf) 

%%  Written for MATLAB_R2016b
%%  Written by Mingxi Zhu, Graduate School of Business,
%%  Stanford University, Stanford, CA 94305.
%%  June, 2021.
%%  mingxiz@stanford.edu
%%  H = speye(dim)

    rac_time_start = tic;
    rac_preparing_time_start = tic;

    dim = size(A,2);
    num_cons = size(A,1);
%     if num_cons > dim
%         fprintf('need p < n')
%         return
%     end
    
    tol_temp = inf;
    sub_model_time = 0;
    sub_update_time = 0;
    sub_model_prepare_time = 0;
    sub_dual_update_time = 0;
    sub_select_time = 0;
    sub_multiply_time = 0;
    
    sub_solver_time = 0;
     
    num_iter = 0;
    
    % construct necessary variabbles 
    Ax = zeros(num_cons,1);
    y_k = zeros(num_cons,1);
    Ax_sub = zeros(num_cons, block_num);
    Ay_sub = zeros(num_cons, block_num);

    % Preselect those who willing to provide data
    numelements = round(percentage*dim);
    selected_indices = sort(randperm(dim, numelements));
    block_size = floor(dim/block_num);
    AAt_store = cell(1, block_num);
    changed_index = cell(1, block_num);
    changed_index_sub = cell(1, block_num);
    unchanged_index_sub = cell(1, block_num);
    changed_index_num = cell(1, block_num);
    for i = 1: block_num-1
        changed_index{1, i} = selected_indices(selected_indices<=i*block_size & selected_indices>=((i - 1)*block_size+1));
        changed_index_sub{1, i} = changed_index{1, i} - (i-1)*block_size;
        unchanged_index_sub{1, i} = setdiff(1:block_size, changed_index_sub{1, i});
        changed_index_num{1, i} = length(changed_index{1, i});
        A_local = A(:, ((i - 1)*block_size+1):i*block_size);
        A_local(:, changed_index_sub{1, i}) = [];
        AAt_store{1, i} = A_local*A_local';
        Ay_sub(:, i) = A_local*c((i-1)*block_size + unchanged_index_sub{1, i});
    end
    changed_index{1, block_num} = selected_indices(selected_indices>=((block_num - 1)*block_size+1));
    changed_index_sub{1, block_num} = changed_index{1, block_num} - (block_num-1)*block_size;
    changed_index_num{1, block_num} = length(changed_index{1, block_num});
    A_local = A(:, ((block_num - 1)*block_size+1):end);
    curr_size = size(A_local, 2);
    unchanged_index_sub{1, block_num} = setdiff(1:curr_size, changed_index_sub{1, block_num});
    A_local(:, changed_index_sub{1, block_num}) = [];
	AAt_store{1, block_num} = A_local*A_local';
    Ay_sub(:, block_num) = A_local*c((block_num-1)*block_size + unchanged_index_sub{1, block_num});
    
    % build x only for selected indices;
    x_global = zeros(numelements, 1);
    
    
    rac_preparing_time = toc(rac_preparing_time_start);

    while tol_temp>tol

        %submodeltime for c_dual_vector
        sub_model_prepare_time_start = tic;
        % this requires A_i^T y_k
        index_perm = randperm(numelements);
        selected_indices_perm = selected_indices(index_perm);
        crt_start = 1;
        crt_end = changed_index_num{1};
        crt_perm = cell(1, block_num);
        crt_index = cell(1, block_num);
        for i = 1: block_num - 1
            crt_perm{1, i} = selected_indices_perm(crt_start:crt_end);
            crt_index{1, i} = index_perm(crt_start:crt_end);
            crt_start = crt_end + 1;
            crt_end = crt_end + changed_index_num{i+1};
        end
        crt_perm{1, block_num} = selected_indices_perm(crt_start:crt_end);
        crt_index{1, block_num} = index_perm(crt_start:crt_end);

        x_visited_block = 1;
        % update_order = randperm(block_num, block_num);
        update_order = 1:block_num;
        sub_model_prepare_time = toc(sub_model_prepare_time_start) + sub_model_prepare_time;
        
        %update primal
        while x_visited_block <= block_num
  
            x_crt_block =  update_order(x_visited_block);    
            sub_model_time_start = tic;
            
            %random select block num from x_unvisited [1,dim]  
            %model buildup  
            sub_select_time_start = tic;
                        
            curr_x_global_index = crt_perm{1, x_crt_block};
            curr_x_global = x_global( crt_index{1, x_crt_block} );
            A_sub_global = A(:, curr_x_global_index);
                
            sub_select_time = sub_select_time + toc(sub_select_time_start);
            
            sub_multiply_time_start = tic;
            A_sub_A_sub_t = AAt_store{1, x_crt_block} + A_sub_global*A_sub_global';
            H_current = eye(num_cons) + lambda*(A_sub_A_sub_t); 
            sub_multiply_time = sub_multiply_time + toc(sub_multiply_time_start);
            
            curr_c_res = Ay_sub(:, x_crt_block) + A_sub_global*c(curr_x_global_index) - (A_sub_A_sub_t)*y_k;

            %% A_subxi = A_sub*x_k_current;
            A_subxi = Ax_sub(:, x_crt_block) + A_sub_global*curr_x_global;
            diff_A_sub = Ax - A_subxi;
            c_current = lambda*A_sub_A_sub_t*diff_A_sub + curr_c_res; 
            
            %model solver 
            right_side = - c_current;
            left_side_unfactorized = H_current;
            sub_model_time = toc(sub_model_time_start) + sub_model_time;
  
            sub_solver_time_start = tic;
            results_mu = left_side_unfactorized\right_side;
            % update x global and Ax_sub;
            new_x = - (lambda*A_sub_global'*diff_A_sub + c(curr_x_global_index) - A_sub_global'*y_k) - lambda*A_sub_global'*results_mu;
            Ax_sub_new = results_mu - A_sub_global*new_x;
            sub_solver_time = toc(sub_solver_time_start) + sub_solver_time;
  
            %update Qx, Ax
            sub_update_time_start = tic;
            Ax = Ax + results_mu - Ax_sub(:, x_crt_block) - A_sub_global*x_global( crt_index{1, x_crt_block} );
            Ax_sub(:, x_crt_block) = Ax_sub_new;
            x_global( crt_index{1, x_crt_block} ) = new_x;
            
            
            % next iteration  
            x_visited_block = x_visited_block + 1;
            sub_update_time = toc(sub_update_time_start) + sub_update_time;

        end
  
        % update Dual
        sub_dual_update_time_start = tic;  
        res_k = Ax;

        % check tolerance, primal
        primal_tol = max(abs(res_k));
        tol_temp = primal_tol;
        y_k = y_k - lambda*res_k;
        num_iter = num_iter+1;
        sub_dual_update_time = toc(sub_dual_update_time_start) + sub_dual_update_time;
  
        if num_iter == max_iter
            break
        end
  
        pd_crt_time = toc(rac_time_start);
        if pd_crt_time > time_limit
            break
        end
  
    end 

    rac_time = toc(rac_time_start);
    sol.beta = y_k;
    sol.total_time = rac_time;
    sol.solver_time = sub_solver_time;
    sol.model_time = sub_model_time;
    sol.prepare_time = rac_preparing_time;
    sol.model_time_detail = [sub_model_prepare_time sub_select_time sub_multiply_time sub_update_time sub_dual_update_time];
    sol.num_iter = num_iter;
    sol.prim_tol = tol_temp;
