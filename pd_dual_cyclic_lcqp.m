function sol = pd_dual_cyclic_lcqp(c, A, block_num, lambda, tol, max_iter, time_limit)

%%  cyclic ADMM
%%  Solver for following Quadratic Optimization Problem 
%%  min 1/2*x^T x + c^T x
%%  s.t.  A x = b
%%  Input (c,A,b,block_num, dual penalty parameter lambda,tolerence of ||Ax-b||_inf) 
%%  Default set x0 = 0, y0 = 0;

%%  Written for MATLAB_R2016b
%%  Written by Mingxi Zhu, Graduate School of Business,
%%  Stanford University, Stanford, CA 94305.
%%  Febrary, 2019.
%%  mingxiz@stanford.edu
%%  Use sparse chol (note: sometimes use non-sparse chol could be faster)

    rac_time_start = tic;
    rac_preparing_time_start = tic;

    dim = size(A,2);
    num_cons = size(A,1);
    Aixi_k = zeros(num_cons,block_num);
    y_k = zeros(num_cons,1);

    tol_temp = inf;
    sub_model_time = 0;
    sub_update_time = 0;
    sub_model_prepare_time = 0;
    sub_dual_update_time = 0;
    sub_select_time = 0;
    %sub_multiply_time = 0;
    sub_solver_time = 0;
    
    num_iter = 0;

    Ax = zeros(num_cons, 1);

    block_size = floor(dim/block_num);
    block_index = cell(1, block_num);
    store_A_sub = cell(1, block_num);
    store_Aici_sub = cell(1, block_num);
    for i_block = 1:block_num-1
        block_index{i_block} = ((i_block - 1)*block_size + 1):i_block*block_size;
        store_A_sub{i_block} = A(:,block_index{i_block});
        store_Aici_sub{i_block} = store_A_sub{i_block}*c(block_index{i_block});
    end
    block_index{block_num} = ((block_num - 1)*block_size + 1):dim;
    store_A_sub{block_num} = A(:,block_index{block_num});
    store_Aici_sub{block_num} = store_A_sub{block_num}*c(block_index{block_num});

    %factorize block
    store_R = cell(1, block_num);
    store_AsubAsubT = cell(1, block_num);
    store_lambdaAsubAsubT = cell(1, block_num);
    for i_block = 1:1:block_num
        store_AsubAsubT{i_block} = store_A_sub{i_block}*store_A_sub{i_block}';
        store_lambdaAsubAsubT{i_block} = lambda*store_AsubAsubT{i_block};
        R = chol(store_lambdaAsubAsubT{i_block} + dim*eye(num_cons));
        store_R{i_block}=R;
    end

    rac_preparing_time = toc(rac_preparing_time_start);


    while tol_temp>tol

    %submodeltime for c_dual_vector
    sub_model_prepare_time_start = tic;
    x_block_index_perm = 1:block_num;
    x_visited_block = 1;
    sub_model_prepare_time = toc(sub_model_prepare_time_start) + sub_model_prepare_time;


    %update primal
        while x_visited_block <= block_num
            sub_model_time_start = tic;
            
            current_block_index = x_block_index_perm(x_visited_block);
            R_current = store_R{current_block_index};
            A_subxi = Aixi_k(:, current_block_index);
            diff_A_sub = (Ax  - A_subxi);
            right_side = -(store_Aici_sub{current_block_index}-store_AsubAsubT{current_block_index}*y_k+store_lambdaAsubAsubT{current_block_index}*diff_A_sub);
            
            sub_select_time = sub_select_time + toc(sub_model_time_start);
            
            sub_solver_time_start = tic;
            results_mu =  R_current\(R_current'\right_side);
            sub_solver_time = toc(sub_solver_time_start) + sub_solver_time;
  
            %update Ax
            sub_update_time_start = tic;
            Ax = Ax + results_mu - A_subxi;
            %update primal
            Aixi_k(:,current_block_index) = results_mu;
            x_visited_block = x_visited_block + 1;
            sub_update_time = sub_update_time + toc(sub_update_time_start);
            
            sub_model_time = toc(sub_model_time_start) + sub_model_time;

        end
  
        %update dual
        sub_dual_update_time_start = tic;   
        res_k = Ax;

        %update dual y_k, z_k
        y_k = y_k - lambda*res_k;
        tol_temp = max(abs(res_k));

        num_iter = num_iter+1;
  
        if num_iter == max_iter
            break
        end
        
        pd_crt_time = toc(rac_time_start);
        if pd_crt_time > time_limit
            break
        end
        
        sub_dual_update_time = toc(sub_dual_update_time_start) + sub_dual_update_time;

    end 

    rac_time = toc(rac_time_start);
    sol.beta = y_k;
    sol.total_time = rac_time;
    sol.solver_time = sub_solver_time;
    sol.model_time = sub_model_time;
    sol.prepare_time = rac_preparing_time;
    sol.model_time_detail = [sub_model_prepare_time sub_select_time sub_update_time sub_dual_update_time];
    sol.num_iter = num_iter;
    sol.prim_tol = tol_temp;
    