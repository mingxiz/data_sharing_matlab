function sol = pd_primal_consensus_admm_logistic_two_block2(x, y, block_num, lambda, lambda_2, max_iter, tol, time_limit)

%%  Consensus ADMM
%%  Solver for estimation problem with n>>p
%%  min sum(log(1+exp(-yi*xi*beta))) data could not be shared across solver)
%%  the first argument of beta is intersection
%%  lambda_2: step-size for beta_i-beta

%%  Written for MATLAB_R2016b
%%  Written by Mingxi Zhu, Graduate School of Business,
%%  Stanford University, Stanford, CA 94305.
%%  July, 2021.
%%  mingxiz@stanford.edu

pd_time_start = tic;
pd_preparing_time_start = tic;

% startiing point
[n, p] = size(x);
% initialilze x
x = [ones(n, 1), x];
x = y.*x;
p = p + 1;

% initialize starting point
x_k = zeros(p, block_num);
z_k = zeros(p, 1);
y_k = zeros(p, block_num);

% prepare X_sub and chol and initialize start point on dual lambda_k;
block_size = floor(n/block_num);
order = 1:n;
store_x_sub = cell(1, block_num);
store_R_sub = cell(1, block_num);
lambda_k = cell(1, block_num);
for i_block = 1:block_num - 1
    store_x_sub{1, i_block} = x(order(((i_block - 1)*block_size + 1):i_block*block_size), :);
    store_R_sub{1, i_block} = chol( lambda*store_x_sub{1, i_block}'*store_x_sub{1, i_block} + lambda_2*eye(p) );
    lambda_k{1, i_block} = zeros(size(store_x_sub{1, i_block}, 1), 1);
end
store_x_sub{1, block_num} = x(order(((block_num - 1)*block_size + 1):n), :);
store_R_sub{1, block_num} = chol( lambda*store_x_sub{1, block_num}'*store_x_sub{1, block_num} + lambda_2*eye(p) );
lambda_k{1, block_num} = zeros(size(store_x_sub{1, block_num}, 1), 1);
t_k = lambda_k;

num_iter = 0;
tol_temp_vec = zeros(block_num, 1);
tol_temp_vec_2 = zeros(block_num, 1);
tol_temp = inf;

% prepare t
sub_model_time_t = 0;
% prepare beta_k
sub_model_time_beta_k = 0;
% prepare beta
sub_model_time_beta = 0;
% update lambda
sub_model_time_lambda = 0;

pd_preparing_time = toc(pd_preparing_time_start);

while tol_temp > tol
    
    
    
    % update t_k first
    for i = 1:block_num
        
        % update t
        sub_model_time_t_start = tic;
        a_k_i = lambda*store_x_sub{1, i}*x_k(:, i) + lambda_k{:, i};
        t_k{1, i} = admm_sub_solver_newton(a_k_i, lambda);
        sub_model_time_t = sub_model_time_t + toc(sub_model_time_t_start);
        
        % update beta_k
        sub_model_time_beta_k_start = tic;
        rhs = lambda_2*z_k + store_x_sub{1, i}'*( lambda*t_k{1, i} - lambda_k{1, i} ) -  y_k(:, i);
        x_k(:, i) = store_R_sub{1, i}\(store_R_sub{1, i}'\rhs);
        sub_model_time_beta_k = sub_model_time_beta_k + toc(sub_model_time_beta_k_start);
    end
    
    %update beta
    sub_model_time_beta_start = tic;
    z_k = 1/block_num*(sum(x_k,2)+1/lambda_2*sum(y_k,2));
    sub_model_time_beta = toc(sub_model_time_beta_start) + sub_model_time_beta;
    
	sub_model_time_lambda_start = tic;    
        
	% update dual
	for block_i = 1: block_num
        diff = x_k(:, block_i) - z_k;
        tol_temp_vec(block_i, 1) = max(abs(diff));
        y_k(:, block_i) = y_k(:, block_i) + lambda_2*diff;
        % update lambda_k (note here could use beta_t+1 or beta_k_{t+1} )
        diff_2 = store_x_sub{1, block_i}*z_k -  t_k{1, block_i};
        tol_temp_vec_2(block_i, 1) = max(abs(diff_2));
        % lambda_k{1, i} = lambda_k{1, i} + lambda*diff_2;
        % try this
        lambda_k{1, block_i} = lambda_k{:, block_i} + lambda*diff_2;
	end
    
	tol1 = max([tol_temp_vec; tol_temp_vec_2]);
	tol_temp = tol1;
 
	num_iter = num_iter + 1;
	if num_iter == max_iter
        break
    end
    
    sub_model_time_lambda = toc(sub_model_time_lambda_start) + sub_model_time_lambda;
    
	pd_crt_time = toc(pd_time_start);
	if pd_crt_time > time_limit
        break
	end
end

sol.beta = z_k;
sol.tol = tol_temp;
sol.total_time = toc(pd_time_start);
sol.sub_model_time_beta_k = sub_model_time_beta_k;
sol.sub_model_time_beta = sub_model_time_beta;
sol.sub_model_time_lambda = sub_model_time_lambda;
sol.prepare_time = pd_preparing_time;
sol.num_iter = num_iter;

    
    
        
    