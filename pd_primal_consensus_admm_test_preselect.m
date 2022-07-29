function sol = pd_primal_consensus_admm_test_preselect(X, y, block_num, lambda, max_iter, tol, time_limit, percentage)

%%  Consensus ADMM
%%  Solver for estimation problem with n>>p 
%%  min (1/2N)* (y-X*beta)^T(y-X*beta)? data percentage sharing

%%  Written for MATLAB_R2016b
%%  Written by Mingxi Zhu, Graduate School of Business,
%%  Stanford University, Stanford, CA 94305.
%%  July, 2019.
%%  mingxiz@stanford.edu

pd_time_start = tic;
pd_preparing_time_start = tic;

% starting point beta = 0;
[n, p] = size(X);
% initialize starting point
x_k = zeros(p, block_num);
z_k = zeros(p, 1);
y_k = zeros(p, block_num);

% Preselect those who willing to provide data
numelements = round(percentage*n);
selected_indices = randperm(n, numelements);
X_g = X(selected_indices, :);
y_g = y(selected_indices, :);
changed_index = cell(1, block_num);

%%     prepare X_sub and X_subTX_sub
block_size = floor(n/block_num);
store_x_sub = cell(1, block_num);
store_xtx_sub = cell(1, block_num);
store_xty_sub = cell(1, block_num);
store_local_index_sub = cell(1, block_num);
store_local_num_perm = zeros(1 ,block_num);

for i_block = 1:block_num - 1
    changed_index{1, i_block} = selected_indices(selected_indices<=i_block*block_size & selected_indices>=((i_block - 1)*block_size+1));
    store_local_num_perm(1, i_block) = length(changed_index{1, i_block});
    X_local = X(((i_block - 1)*block_size+1):i_block*block_size,:);
    y_local = y(((i_block - 1)*block_size+1):i_block*block_size,:);
    store_local_index_temp = ((i_block - 1)*block_size+1):i_block*block_size;
    X_local(changed_index{1, i_block}-(i_block - 1)*block_size,:) = [];
    y_local(changed_index{1, i_block}-(i_block - 1)*block_size,:) = [];
    store_local_index_temp(changed_index{1, i_block}-(i_block - 1)*block_size) = [];
    store_local_index_sub{1, i_block} = store_local_index_temp;
    store_x_sub{1, i_block} = X_local;
    store_xtx_sub{1, i_block} = 1/n*store_x_sub{1, i_block}'*store_x_sub{1, i_block} + lambda*eye(p);
    store_xty_sub{1, i_block} = -1/n*X_local'*y_local;
end
changed_index{1, block_num} = selected_indices(selected_indices>=((block_num - 1)*block_size+1));
store_local_num_perm(1, block_num) = length(changed_index{1, block_num});
X_local = X(((block_num - 1)*block_size+1):end,:);
y_local = y(((block_num - 1)*block_size+1):end,:);
store_local_index_temp = ((block_num - 1)*block_size+1):n;
store_local_index_temp(changed_index{1, block_num}-(block_num - 1)*block_size) = [];
store_local_index_sub{1, block_num} = store_local_index_temp;
X_local(changed_index{1, block_num}-(block_num - 1)*block_size,:) = [];
y_local(changed_index{1, block_num}-(block_num - 1)*block_size,:) = [];
store_x_sub{1, block_num} = X_local;
store_xtx_sub{1, block_num} = 1/n*store_x_sub{1, block_num}'*store_x_sub{1, block_num} + lambda*eye(p);
store_xty_sub{1, block_num} = -1/n*X_local'*y_local;




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
store_crt_changed_index = cell(1, block_num);
store_crt_changed_x =cell(1, block_num);
store_crt_changed_y =cell(1, block_num);

while tol_temp>tol
    
    sub_model_time_1_start = tic;    
    %change block-wise randomization
    lambda_z = lambda*z_k;
    sub_model_time_1 = toc(sub_model_time_1_start) + sub_model_time_1;
    
    % permute data 
    crt_perm_rnd = randperm(numelements);
    crt_perm = selected_indices(crt_perm_rnd);
    temp_Xg = X_g(crt_perm_rnd, :);
    temp_yg = y_g(crt_perm_rnd, :);
    crt_ptr = 0;
    for i = 1:block_num
        crt_length = store_local_num_perm(1, i);
        store_crt_changed_index{1, i} = crt_perm(crt_ptr + 1: crt_ptr + crt_length);
        store_crt_changed_x{1, i} = temp_Xg(crt_ptr + 1: crt_ptr + crt_length, :);
        store_crt_changed_y{1, i} = temp_yg(crt_ptr + 1: crt_ptr + crt_length, :);
        crt_ptr = crt_ptr + crt_length;
    end    
    
    
    for i = 1: block_num 
        
        sub_model_time_2_start = tic;

        c = store_xty_sub{1, i} - 1/n*store_crt_changed_x{1, i}'*store_crt_changed_y{1, i};
        
        L = store_xtx_sub{1, i} +  1/n*(store_crt_changed_x{1, i}'*store_crt_changed_x{1, i});
        
        b = -(c + y_k(:, i) - lambda_z);
        
        sub_model_time_2 = toc(sub_model_time_2_start) + sub_model_time_2;
        
        
        sub_solver_time_start = tic;
        x_k(:, i) = L\b;
        sub_solver_time = toc(sub_solver_time_start) + sub_solver_time;
    end
        
    sub_model_time_3_start = tic;
    z_k = 1/block_num*(sum(x_k,2)+1/lambda*sum(y_k,2));
    sub_model_time_3 = toc(sub_model_time_3_start) + sub_model_time_3;
    
    sub_model_time_4_start = tic;    
    
   for i = 1: block_num
        diff = x_k(:, i)-z_k;
        tol_temp_vec(i, 1) = max(abs(diff));
        y_k(:, i) = y_k(:, i) + lambda*diff;
   end 
    
    tol1 = max(tol_temp_vec);
    tol_temp = tol1;
    
    num_iter = num_iter + 1;
    if num_iter == max_iter
       break
    end
    
    if toc(pd_time_start)>= time_limit
        break
    end
    sub_model_time_4 = toc(sub_model_time_4_start) + sub_model_time_4;
    
end

sol.beta = z_k;
sol.tol = tol_temp;
sol.total_time = toc(pd_time_start);
sol.solver_time = sub_solver_time;
sol.model_time_1 = sub_model_time_1;
sol.model_time_2 = sub_model_time_2;
sol.model_time_3 = sub_model_time_3;
sol.model_time_4 = sub_model_time_4;
sol.prepare_time = pd_preparing_time;
sol.num_iter = num_iter;
