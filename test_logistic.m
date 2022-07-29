function report = test_logistic

%% main file for testing logistic regression with 
%% gradient method, newton method, primal distributed ADMM method, dual DRAP ADMM method
%% n = 500, p = 20;
n_sample = 20;
store_performance_gradient = zeros(1, n_sample);
store_performance_distributed = zeros(1, n_sample);
store_performance_drap = zeros(1, n_sample);

for i_sample = 1:n_sample
    n = 500;
    p = 20;
    x = randn(n, p);
    % dominate the variance impact
    beta_s = (100*p)*randn(p+1, 1);
    a1 = 0.01; 
    z = [ones(n,1),x]*beta_s + a1*randn*ones(n, 1);
    % sigmod on y;
    p = 1./(1+exp(-z));
    x1all = x;
    y1all = p;
    y1all( find(y1all > 0.5) ) = 1;
    y1all( find(y1all <= 0.5) ) = -1;

    %% parameter setup
    rho = 1;
    lambda = 1;
    mu = 0;
    n = size(x1all, 1);
    time_limit = inf;
    max_iter = 10;
    tol_admm = 0;
    block_num = 4;
    percentage = 0.05;
    % newton method for benchmark
    A = y1all.*x1all;
    sol_newton_global = newton_log(A, y1all, time_limit, max_iter);
    newton_global_iter = sol_newton_global.num_iter;
    AL_newton = norm(sol_newton_global.beta - beta_s);

    % gradient method
    A = y1all.*x1all;
    sol_gradient = gradient_log(A, y1all, time_limit, max_iter);
    gradient_iter = sol_gradient.num_iter;
    AL_gradient = (norm(sol_gradient.beta - beta_s)-AL_newton)./AL_newton;
    store_performance_gradient(1, i_sample) = AL_gradient;

    % distributed admm
    sol_distributed = pd_primal_consensus_admm_logistic_two_block2(x1all, y1all, block_num, lambda, n, max_iter, tol_admm, time_limit);
    distributed_iter = sol_distributed.num_iter;
    AL_distributed = (norm(sol_distributed.beta - beta_s)-AL_newton)./AL_newton;
    store_performance_distributed(1, i_sample) = AL_distributed;
    
    % drc 2
    sol_drc_2 = pd_drc_admm_logistic_two_block(x1all, y1all, block_num, percentage, lambda, n, max_iter, tol_admm, time_limit);
    drc2_iter = sol_drc_2.num_iter;
    AL_drap = (norm(sol_drc_2.beta -beta_s)-AL_newton)./AL_newton;
    store_performance_drap(1, i_sample) = AL_drap;
end
report.AL_gradient_n500p20 = mean(store_performance_gradient);
report.AL_distributed_n500p20 = mean(store_performance_distributed);
report.AL_drap_n500p20 = mean(store_performance_drap);
% report result
formatSpec = 'For n = 500, p = 20, The relative ratio of absolute loss on gradient method is %8.3e \n';
fprintf(formatSpec, report.AL_gradient_n500p20)
formatSpec = 'For n = 500, p = 20, The relative ratio of absolute loss on distributed ADMM is %8.3e \n';
fprintf(formatSpec, report.AL_distributed_n500p20)
formatSpec = 'For n = 500, p = 20, The relative ratio of absolute loss on DRAP-ADMM is %8.3e \n';
fprintf(formatSpec, report.AL_drap_n500p20)


%% n = 800, p = 40;
n_sample = 20;
store_performance_gradient = zeros(1, n_sample);
store_performance_distributed = zeros(1, n_sample);
store_performance_drap = zeros(1, n_sample);

for i_sample = 1:n_sample
    n = 800;
    p = 40;
    x = randn(n, p);
    % dominate the variance impact
    beta_s = (100*p)*randn(p+1, 1);
    a1 = 0.01; 
    z = [ones(n,1),x]*beta_s + a1*randn*ones(n, 1);
    % sigmod on y;
    p = 1./(1+exp(-z));
    x1all = x;
    y1all = p;
    y1all( find(y1all > 0.5) ) = 1;
    y1all( find(y1all <= 0.5) ) = -1;

    %% parameter setup
    rho = 1;
    lambda = 1;
    mu = 0;
    n = size(x1all, 1);
    time_limit = inf;
    max_iter = 10;
    tol_admm = 0;
    block_num = 4;
    percentage = 0.05;
    % newton method for benchmark
    A = y1all.*x1all;
    sol_newton_global = newton_log(A, y1all, time_limit, max_iter);
    newton_global_iter = sol_newton_global.num_iter;
    AL_newton = norm(sol_newton_global.beta - beta_s);

    % gradient method
    A = y1all.*x1all;
    sol_gradient = gradient_log(A, y1all, time_limit, max_iter);
    gradient_iter = sol_gradient.num_iter;
    AL_gradient = (norm(sol_gradient.beta - beta_s)-AL_newton)./AL_newton;
    store_performance_gradient(1, i_sample) = AL_gradient;

    % distributed admm
    sol_distributed = pd_primal_consensus_admm_logistic_two_block2(x1all, y1all, block_num, lambda, n, max_iter, tol_admm, time_limit);
    distributed_iter = sol_distributed.num_iter;
    AL_distributed = (norm(sol_distributed.beta - beta_s)-AL_newton)./AL_newton;
    store_performance_distributed(1, i_sample) = AL_distributed;
    
    % drc 2
    sol_drc_2 = pd_drc_admm_logistic_two_block(x1all, y1all, block_num, percentage, lambda, n, max_iter, tol_admm, time_limit);
    drc2_iter = sol_drc_2.num_iter;
    AL_drap = (norm(sol_drc_2.beta -beta_s)-AL_newton)./AL_newton;
    store_performance_drap(1, i_sample) = AL_drap;
end
report.AL_gradient_n800p40 = mean(store_performance_gradient);
report.AL_distributed_n800p40 = mean(store_performance_distributed);
report.AL_drap_n800p40 = mean(store_performance_drap);
% report result
formatSpec = 'For n = 800, p = 40, The relative ratio of absolute loss on gradient method is %8.3e \n';
fprintf(formatSpec, report.AL_gradient_n800p40)
formatSpec = 'For n = 800, p = 40, The relative ratio of absolute loss on distributed ADMM is %8.3e \n';
fprintf(formatSpec, report.AL_distributed_n800p40)
formatSpec = 'For n = 800, p = 40, The relative ratio of absolute loss on DRAP-ADMM is %8.3e \n';
fprintf(formatSpec, report.AL_drap_n800p40)


%% n = 1000, p = 100;
n_sample = 20;
store_performance_gradient = zeros(1, n_sample);
store_performance_distributed = zeros(1, n_sample);
store_performance_drap = zeros(1, n_sample);

for i_sample = 1:n_sample
    n = 1000;
    p = 100;
    x = randn(n, p);
    % dominate the variance impact
    beta_s = (100*p)*randn(p+1, 1);
    a1 = 0.01; 
    z = [ones(n,1),x]*beta_s + a1*randn*ones(n, 1);
    % sigmod on y;
    p = 1./(1+exp(-z));
    x1all = x;
    y1all = p;
    y1all( find(y1all > 0.5) ) = 1;
    y1all( find(y1all <= 0.5) ) = -1;

    %% parameter setup
    rho = 1;
    lambda = 1;
    mu = 0;
    n = size(x1all, 1);
    time_limit = inf;
    max_iter = 10;
    tol_admm = 0;
    block_num = 4;
    percentage = 0.05;
    % newton method for benchmark
    A = y1all.*x1all;
    sol_newton_global = newton_log(A, y1all, time_limit, max_iter);
    newton_global_iter = sol_newton_global.num_iter;
    AL_newton = norm(sol_newton_global.beta - beta_s);

    % gradient method
    A = y1all.*x1all;
    sol_gradient = gradient_log(A, y1all, time_limit, max_iter);
    gradient_iter = sol_gradient.num_iter;
    AL_gradient = (norm(sol_gradient.beta - beta_s)-AL_newton)./AL_newton;
    store_performance_gradient(1, i_sample) = AL_gradient;

    % distributed admm
    sol_distributed = pd_primal_consensus_admm_logistic_two_block2(x1all, y1all, block_num, lambda, n, max_iter, tol_admm, time_limit);
    distributed_iter = sol_distributed.num_iter;
    AL_distributed = (norm(sol_distributed.beta - beta_s)-AL_newton)./AL_newton;
    store_performance_distributed(1, i_sample) = AL_distributed;
    
    % drc 2
    sol_drc_2 = pd_drc_admm_logistic_two_block(x1all, y1all, block_num, percentage, lambda, n, max_iter, tol_admm, time_limit);
    drc2_iter = sol_drc_2.num_iter;
    AL_drap = (norm(sol_drc_2.beta -beta_s)-AL_newton)./AL_newton;
    store_performance_drap(1, i_sample) = AL_drap;
end
report.AL_gradient_n1000p100 = mean(store_performance_gradient);
report.AL_distributed_n1000p100 = mean(store_performance_distributed);
report.AL_drap_n1000p100 = mean(store_performance_drap);
% report result
formatSpec = 'For n = 1000, p = 100, The relative ratio of absolute loss on gradient method is %8.3e \n';
fprintf(formatSpec, report.AL_gradient_n1000p100)
formatSpec = 'For n = 1000, p = 100, The relative ratio of absolute loss on distributed ADMM is %8.3e \n';
fprintf(formatSpec, report.AL_distributed_n1000p100)
formatSpec = 'For n = 1000, p = 100, The relative ratio of absolute loss on DRAP-ADMM is %8.3e \n';
fprintf(formatSpec, report.AL_drap_n1000p100)