function report = test_online


t=readtable('OnlineNewsPopularity.csv');
t = table2array(t(:, 2:end));
x = t(:, 1:59);
y = t(:, 60);
% further scale by sqrt n 
[n, p] = size(x);
x_org = x./(n);
y_org = y./(n);

% regularize the lower eigenvalue
% randperm regularization to cancel influence of fixed regularized block
order_insert = randperm(n+p,p); 
x = zeros(n + p, p);
y = zeros(n + p, 1);
x(order_insert, :) = eye(p);
x(setdiff(1:n+p, order_insert), :) = x_org;
y(setdiff(1:n+p, order_insert), :) = y_org;
% start on simulation 
block_num = 4;
xtx = x'*x;
beta_s = inv(xtx)*x'*y;
eigxtx = eig(xtx);
l_max = max(abs(eigxtx));
l_min = min(abs(eigxtx));

% fix time
lambda = 1;
tol_admm = 0;
max_iter = inf;
time_limit = 100;
percentage = 0.05;
sol_dual_admm_v2 = pd_dual_rac_lcqp_test_preselect_updated_new(y, x', block_num, percentage, lambda, tol_admm, max_iter, time_limit);
sol_dadmm = pd_primal_consensus_admm_1015(x, y, block_num, lambda, max_iter, tol_admm, time_limit);
sol_rp_admm =  pd_dual_rp_lcqp(y, x', block_num, lambda, tol_admm, max_iter, time_limit);
sol_double_admm =  pd_dual_ds_lcqp(y, x', block_num, lambda, tol_admm, max_iter, time_limit);
sol_cyclic_admm = pd_dual_cyclic_lcqp(y, x', block_num, lambda, tol_admm, max_iter, time_limit);

xtx = x'*x;
beta_s = inv(xtx)*x'*y;
report.num_iter_dadmm_fix_time = sol_dadmm.num_iter;
report.num_iter_dual_admm_fix_time = sol_dual_admm_v2.num_iter;
report.num_iter_rp_admm_fix_time = sol_rp_admm.num_iter;
report.num_iter_cyclic_admm_fix_time = sol_cyclic_admm.num_iter;
report.num_iter_ds_admm_fix_time = sol_double_admm.num_iter;

report.AL_dadmm_fix_time = norm(sol_dadmm.beta - beta_s, 2);
report.AL_dual_admm_fix_time =norm(sol_dual_admm_v2.beta - beta_s, 2);
report.AL_rp_admm_fix_time = norm(sol_rp_admm.beta - beta_s, 2);
report.AL_cyclic_admm_fix_time =norm(sol_cyclic_admm.beta - beta_s, 2);
report.AL_ds_admm_fix_time =norm(sol_double_admm.beta - beta_s, 2);

% fix number of iteration
lambda = 1;
tol_admm = 0;
max_iter = 200;
time_limit = inf;
sol_dadmm = pd_primal_consensus_admm_1015(x, y, block_num, lambda, max_iter, tol_admm, time_limit);
percentage = 0.05;
sol_dual_admm_v2 = pd_dual_rac_lcqp_test_preselect_updated_new(y, x', block_num, percentage, lambda, tol_admm, max_iter, time_limit);
sol_rp_admm =  pd_dual_rp_lcqp(y, x', block_num, lambda, tol_admm, max_iter, time_limit);
sol_double_admm =  pd_dual_ds_lcqp(y, x', block_num, lambda, tol_admm, max_iter, time_limit);
sol_cyclic_admm = pd_dual_cyclic_lcqp(y, x', block_num, lambda, tol_admm, max_iter, time_limit);

xtx = x'*x;
beta_s = inv(xtx)*x'*y;
report.time_dadmm_fix_iter = sol_dadmm.total_time;
report.time_dual_admm_fix_iter = sol_dual_admm_v2.total_time;
report.time_rp_admm_fix_iter = sol_rp_admm.total_time;
report.time_cyclic_admm_fix_iter = sol_cyclic_admm.total_time;
report.time_ds_admm_fix_iter = sol_double_admm.total_time;

report.AL_dadmm_fix_iter = norm(sol_dadmm.beta - beta_s, 2);
report.AL_dual_admm_fix_iter =norm(sol_dual_admm_v2.beta - beta_s, 2);
report.AL_rp_admm_fix_iter = norm(sol_rp_admm.beta - beta_s, 2);
report.AL_cyclic_admm_fix_iter =norm(sol_cyclic_admm.beta - beta_s, 2);
report.AL_ds_admm_fix_iter =norm(sol_double_admm.beta - beta_s, 2);

% loss_dadmm = ((x*sol_dadmm.beta-y)'*(x*sol_dadmm.beta-y)-(x*beta_s-y)'*(x*beta_s-y))/((x*beta_s-y)'*(x*beta_s-y))
% loss_dual_admm = ((x*sol_dual_admm.beta-y)'*(x*sol_dual_admm.beta-y)-(x*beta_s-y)'*(x*beta_s-y))/((x*beta_s-y)'*(x*beta_s-y))

% report result
formatSpec = 'Fixing run time equals to 100s, the absolute loss on primal distributed ADMM is %8.3e \n';
fprintf(formatSpec, report.AL_dadmm_fix_time)

formatSpec = 'Fixing run time equals to 100s, the absolute loss on DRC ADMM is %8.3e \n';
fprintf(formatSpec, report.AL_dual_admm_fix_time)

formatSpec = 'Fixing run time equals to 100s, the absolute loss on dual RP ADMM is %8.3e \n';
fprintf(formatSpec, report.AL_rp_admm_fix_time)

formatSpec = 'Fixing run time equals to 100s, the absolute loss on dual cyclic ADMM is %8.3e \n';
fprintf(formatSpec, report.AL_cyclic_admm_fix_time)

formatSpec = 'Fixing run time equals to 100s, the absolute loss on dual double sweep ADMM is %8.3e \n';
fprintf(formatSpec, report.AL_ds_admm_fix_time)



formatSpec = 'Fixing number of iterations equals to 200, the absolute loss on primal distributed ADMM is %8.3e \n';
fprintf(formatSpec, report.AL_dadmm_fix_iter)

formatSpec = 'Fixing number of iterations equals to 200, the absolute loss on DRC ADMM is %8.3e \n';
fprintf(formatSpec, report.AL_dual_admm_fix_iter)

formatSpec = 'Fixing number of iterations equals to 200, the absolute loss on dual RP ADMM is %8.3e \n';
fprintf(formatSpec, report.AL_rp_admm_fix_iter)

formatSpec = 'Fixing number of iterations equals to 200, the absolute loss on dual cyclic ADMM is %8.3e \n';
fprintf(formatSpec, report.AL_cyclic_admm_fix_iter)

formatSpec = 'Fixing number of iterations equals to 200, the absolute loss on dual double sweep ADMM is %8.3e \n';
fprintf(formatSpec, report.AL_ds_admm_fix_iter)


