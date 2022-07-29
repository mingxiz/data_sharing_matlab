function report = test_appliance

t=readtable('energydata_complete.csv');
t2 = table2array(t(:,2:29));
t = zeros(19735, 28);
for i = 1: 19735
    for j = 1: 28
        crt_cell = t2{i, j};
        t(i, j) = str2num(crt_cell);
    end
end
x = t(:,2:end);
y = t(:,1);
[n, p] = size(x);
x_org = x./sqrt(n);
y_org = y./sqrt(n);

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
sol_dual_admm_v2_t = pd_dual_rac_lcqp_test_preselect_updated_new(y, x', block_num, percentage, lambda, tol_admm, max_iter, time_limit);
sol_dadmm_t = pd_primal_consensus_admm_1015(x, y, block_num, lambda, max_iter, tol_admm, time_limit);
sol_rp_admm_t =  pd_dual_rp_lcqp(y, x', block_num, lambda, tol_admm, max_iter, time_limit);
sol_double_admm_t =  pd_dual_ds_lcqp(y, x', block_num, lambda, tol_admm, max_iter, time_limit);
sol_cyclic_admm_t = pd_dual_cyclic_lcqp(y, x', block_num, lambda, tol_admm, max_iter, time_limit);
sol_dadmm_datashare_t = pd_primal_consensus_admm_test_preselect(x, y, block_num, lambda, max_iter, tol_admm, time_limit, percentage);
sol_cylic_admm_datashare_t = pd_dual_cyclic_lcqp_test_preselect_updated_new(y, x', block_num, percentage, lambda, tol_admm, max_iter, time_limit);
sol_double_admm_datashare_t = pd_dual_doublesweep_lcqp_test_preselect_updated_new(y, x', block_num, percentage, lambda, tol_admm, max_iter, time_limit);


xtx = x'*x;
beta_s = inv(xtx)*x'*y;
report.num_iter_dadmm_fix_time = sol_dadmm_t.num_iter;
report.num_iter_dual_admm_fix_time = sol_dual_admm_v2_t.num_iter;
report.num_iter_rp_admm_fix_time = sol_rp_admm_t.num_iter;
report.num_iter_cyclic_admm_fix_time = sol_cyclic_admm_t.num_iter;
report.num_iter_ds_admm_fix_time = sol_double_admm_t.num_iter;
report.num_iter_dadmm_datashare_fix_time = sol_dadmm_datashare_t.num_iter;
report.num_iter_cyclic_datashare_fix_time = sol_cylic_admm_datashare_t.num_iter;
report.num_iter_ds_admm_data_share_fix_time = sol_double_admm_datashare_t.num_iter;

report.AL_dadmm_fix_time = norm(sol_dadmm_t.beta - beta_s, 2);
report.AL_dual_admm_fix_time =norm(sol_dual_admm_v2_t.beta - beta_s, 2);
report.AL_rp_admm_fix_time = norm(sol_rp_admm_t.beta - beta_s, 2);
report.AL_cyclic_admm_fix_time =norm(sol_cyclic_admm_t.beta - beta_s, 2);
report.AL_ds_admm_fix_time =norm(sol_double_admm_t.beta - beta_s, 2);
report.AL_dadmm_datashare_fix_time =norm(sol_dadmm_datashare_t.beta - beta_s, 2);
report.AL_cyclic_datashare_fix_time =norm(sol_cylic_admm_datashare_t.beta - beta_s, 2);
report.AL_ds_admm_data_share_fix_time =norm(sol_double_admm_datashare_t.beta - beta_s, 2);

% fix number of iteration
lambda = 1;
tol_admm = 0;
max_iter = 200;
time_limit = inf;
percentage = 0.05;
sol_dual_admm_v2_n = pd_dual_rac_lcqp_test_preselect_updated_new(y, x', block_num, percentage, lambda, tol_admm, max_iter, time_limit);
sol_dadmm_n = pd_primal_consensus_admm_1015(x, y, block_num, lambda, max_iter, tol_admm, time_limit);
sol_rp_admm_n =  pd_dual_rp_lcqp(y, x', block_num, lambda, tol_admm, max_iter, time_limit);
sol_double_admm_n =  pd_dual_ds_lcqp(y, x', block_num, lambda, tol_admm, max_iter, time_limit);
sol_cyclic_admm_n = pd_dual_cyclic_lcqp(y, x', block_num, lambda, tol_admm, max_iter, time_limit);
sol_dadmm_datashare_n = pd_primal_consensus_admm_test_preselect(x, y, block_num, lambda, max_iter, tol_admm, time_limit, percentage);
sol_cylic_admm_datashare_n = pd_dual_cyclic_lcqp_test_preselect_updated_new(y, x', block_num, percentage, lambda, tol_admm, max_iter, time_limit);
sol_double_admm_datashare_n = pd_dual_doublesweep_lcqp_test_preselect_updated_new(y, x', block_num, percentage, lambda, tol_admm, max_iter, time_limit);


xtx = x'*x;
beta_s = inv(xtx)*x'*y;
report.time_dadmm_fix_iter = sol_dadmm_n.total_time;
report.time_dual_admm_fix_iter = sol_dual_admm_v2_n.total_time;
report.time_rp_admm_fix_iter = sol_rp_admm_n.total_time;
report.time_cyclic_admm_fix_iter = sol_cyclic_admm_n.total_time;
report.time_ds_admm_fix_iter = sol_double_admm_n.total_time;
report.time_dadmm_datashare_fix_iter = sol_dadmm_datashare_n.num_iter;
report.time_cyclic_datashare_fix_iter = sol_cylic_admm_datashare_n.num_iter;
report.time_ds_admm_data_share_fix_iter = sol_double_admm_datashare_n.num_iter;


report.AL_dadmm_fix_iter = norm(sol_dadmm_n.beta - beta_s, 2);
report.AL_dual_admm_fix_iter =norm(sol_dual_admm_v2_n.beta - beta_s, 2);
report.AL_rp_admm_fix_iter = norm(sol_rp_admm_n.beta - beta_s, 2);
report.AL_cyclic_admm_fix_iter =norm(sol_cyclic_admm_n.beta - beta_s, 2);
report.AL_ds_admm_fix_iter =norm(sol_double_admm_n.beta - beta_s, 2);
report.AL_dadmm_datashare_fix_iter =norm(sol_dadmm_datashare_n.beta - beta_s, 2);
report.AL_cyclic_datashare_fix_iter =norm(sol_cylic_admm_datashare_n.beta - beta_s, 2);
report.AL_ds_admm_data_share_fix_iter =norm(sol_double_admm_datashare_n.beta - beta_s, 2);

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

formatSpec = 'Fixing run time equals to 100s, the absolute loss on primal distributed ADMM with data share is %8.3e \n';
fprintf(formatSpec, report.AL_dadmm_datashare_fix_time)

formatSpec = 'Fixing run time equals to 100s, the absolute loss on dual cyclic ADMM with data share is %8.3e \n';
fprintf(formatSpec, report.AL_cyclic_datashare_fix_time)

formatSpec = 'Fixing run time equals to 100s, the absolute loss on dual double sweep ADMM with data share is %8.3e \n';
fprintf(formatSpec, report.AL_ds_admm_data_share_fix_time)



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

formatSpec = 'Fixing number of iterations equals to 200, the absolute loss on primal distributed ADMM with data share is %8.3e \n';
fprintf(formatSpec, report.AL_dadmm_datashare_fix_iter)

formatSpec = 'Fixing number of iterations equals to 200, the absolute loss on dual cyclic ADMM with data share is %8.3e \n';
fprintf(formatSpec, report.AL_cyclic_datashare_fix_iter) 

formatSpec = 'Fixing number of iterations equals to 200, the absolute loss on dual double sweep ADMM with data share is %8.3e \n';
fprintf(formatSpec, report.AL_ds_admm_data_share_fix_iter)


