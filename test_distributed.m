%% main compare data share and put all things in to one block;

function report = test_distributed
    
    % YP
    load('YP_train.mat')
    [n, p] = size(x);
    x_org = x./sqrt(n);
    y_org = y./sqrt(n);
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
    % fix time
    lambda = 1;
    tol_admm = 0;
    max_iter = inf;
    time_limit = 100;
    percentage = 0.05;
    sol_dadmm_t = pd_primal_consensus_admm_1015(x, y, block_num, lambda, max_iter, tol_admm, time_limit);
    sol_dadmm_datashare_t2 = pd_primal_consensus_admm_g2(x, y, block_num, lambda, max_iter, tol_admm, time_limit, percentage);
    report.AL_dadmm_fix_time_YP = norm(sol_dadmm_t.beta - beta_s, 2);
    report.AL_dadmm_datashare_fix_time2_YP =norm(sol_dadmm_datashare_t2.beta - beta_s, 2);  
    % fix number of iteration
    lambda = 1;
    tol_admm = 0;
    max_iter = 200;
    time_limit = inf;
    percentage = 0.05;
    sol_dadmm_n = pd_primal_consensus_admm_1015(x, y, block_num, lambda, max_iter, tol_admm, time_limit);
    sol_dadmm_datashare_n2 = pd_primal_consensus_admm_g2(x, y, block_num, lambda, max_iter, tol_admm, time_limit, percentage);
    report.AL_dadmm_fix_iter_YP = norm(sol_dadmm_n.beta - beta_s, 2);
    report.AL_dadmm_datashare_fix_iter2_YP =norm(sol_dadmm_datashare_n2.beta - beta_s, 2);

    % wine_white
    t = readtable("winequality-white.csv");
    t = table2array(t);
    y = t(:, 12);
    x = t(:, 1:11);
    [n, p] = size(x);
    x_org = x./sqrt(n);
    y_org = y./sqrt(n);
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
    % fix time
    lambda = 1;
    tol_admm = 0;
    max_iter = inf;
    time_limit = 100;
    percentage = 0.05;
    sol_dadmm_t = pd_primal_consensus_admm_1015(x, y, block_num, lambda, max_iter, tol_admm, time_limit);
    sol_dadmm_datashare_t2 = pd_primal_consensus_admm_g2(x, y, block_num, lambda, max_iter, tol_admm, time_limit, percentage);
    report.AL_dadmm_fix_time_winewhite = norm(sol_dadmm_t.beta - beta_s, 2);
    report.AL_dadmm_datashare_fix_time2_winewhite =norm(sol_dadmm_datashare_t2.beta - beta_s, 2);  
    % fix number of iteration
    lambda = 1;
    tol_admm = 0;
    max_iter = 200;
    time_limit = inf;
    percentage = 0.05;
    sol_dadmm_n = pd_primal_consensus_admm_1015(x, y, block_num, lambda, max_iter, tol_admm, time_limit);
    sol_dadmm_datashare_n2 = pd_primal_consensus_admm_g2(x, y, block_num, lambda, max_iter, tol_admm, time_limit, percentage);
    report.AL_dadmm_fix_iter_winewhite  = norm(sol_dadmm_n.beta - beta_s, 2);
    report.AL_dadmm_datashare_fix_iter2_winewhite  =norm(sol_dadmm_datashare_n2.beta - beta_s, 2);
    
    % wine_red
    t = readtable("winequality-red.csv");
    t = table2array(t);
    y = t(:, 12);
    x = t(:, 1:11);
    [n, p] = size(x);
    x_org = x./sqrt(n);
    y_org = y./sqrt(n);
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
    % fix time
    lambda = 1;
    tol_admm = 0;
    max_iter = inf;
    time_limit = 100;
    percentage = 0.05;
    sol_dadmm_t = pd_primal_consensus_admm_1015(x, y, block_num, lambda, max_iter, tol_admm, time_limit);
    sol_dadmm_datashare_t2 = pd_primal_consensus_admm_g2(x, y, block_num, lambda, max_iter, tol_admm, time_limit, percentage);
    report.AL_dadmm_fix_time_winered= norm(sol_dadmm_t.beta - beta_s, 2);
    report.AL_dadmm_datashare_fix_time2_winered =norm(sol_dadmm_datashare_t2.beta - beta_s, 2);  
    % fix number of iteration
    lambda = 1;
    tol_admm = 0;
    max_iter = 200;
    time_limit = inf;
    percentage = 0.05;
    sol_dadmm_n = pd_primal_consensus_admm_1015(x, y, block_num, lambda, max_iter, tol_admm, time_limit);
    sol_dadmm_datashare_n2 = pd_primal_consensus_admm_g2(x, y, block_num, lambda, max_iter, tol_admm, time_limit, percentage);
    report.AL_dadmm_fix_iter_winered  = norm(sol_dadmm_n.beta - beta_s, 2);
    report.AL_dadmm_datashare_fix_iter2_winered =norm(sol_dadmm_datashare_n2.beta - beta_s, 2);
    
    % wave
    load('wave_data.mat')
    % change meter to kilometer
    [n, p] = size(x);
    x_org = x./sqrt(1000*n);
    y_org = y./sqrt(1000*n);
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
    % fix time
    lambda = 1;
    tol_admm = 0;
    max_iter = inf;
    time_limit = 100;
    percentage = 0.05;
    sol_dadmm_t = pd_primal_consensus_admm_1015(x, y, block_num, lambda, max_iter, tol_admm, time_limit);
    sol_dadmm_datashare_t2 = pd_primal_consensus_admm_g2(x, y, block_num, lambda, max_iter, tol_admm, time_limit, percentage);
    report.AL_dadmm_fix_time_wave = norm(sol_dadmm_t.beta - beta_s, 2);
    report.AL_dadmm_datashare_fix_time2_wave =norm(sol_dadmm_datashare_t2.beta - beta_s, 2);  
    % fix number of iteration
    lambda = 1;
    tol_admm = 0;
    max_iter = 200;
    time_limit = inf;
    percentage = 0.05;
    sol_dadmm_n = pd_primal_consensus_admm_1015(x, y, block_num, lambda, max_iter, tol_admm, time_limit);
    sol_dadmm_datashare_n2 = pd_primal_consensus_admm_g2(x, y, block_num, lambda, max_iter, tol_admm, time_limit, percentage);
    report.AL_dadmm_fix_iter_wave = norm(sol_dadmm_n.beta - beta_s, 2);
    report.AL_dadmm_datashare_fix_iter2_wave =norm(sol_dadmm_datashare_n2.beta - beta_s, 2);
    
    %UJI
    t = readtable('trainingData.csv');
    t = table2array(t);
    x = t(:, 1:520);
    y = t(:, 521);
    [n, p] = size(x);
    x_org = x./sqrt(n);
    y_org = y./sqrt(n);
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
    % fix time
    lambda = 1;
    tol_admm = 0;
    max_iter = inf;
    time_limit = 100;
    percentage = 0.05;
    sol_dadmm_t = pd_primal_consensus_admm_1015(x, y, block_num, lambda, max_iter, tol_admm, time_limit);
    sol_dadmm_datashare_t2 = pd_primal_consensus_admm_g2(x, y, block_num, lambda, max_iter, tol_admm, time_limit, percentage);
    report.AL_dadmm_fix_time_UJI = norm(sol_dadmm_t.beta - beta_s, 2);
    report.AL_dadmm_datashare_fix_time2_UJI =norm(sol_dadmm_datashare_t2.beta - beta_s, 2);  
    % fix number of iteration
    lambda = 1;
    tol_admm = 0;
    max_iter = 200;
    time_limit = inf;
    percentage = 0.05;
    sol_dadmm_n = pd_primal_consensus_admm_1015(x, y, block_num, lambda, max_iter, tol_admm, time_limit);
    sol_dadmm_datashare_n2 = pd_primal_consensus_admm_g2(x, y, block_num, lambda, max_iter, tol_admm, time_limit, percentage);
    report.AL_dadmm_fix_iter_UJI = norm(sol_dadmm_n.beta - beta_s, 2);
    report.AL_dadmm_datashare_fix_iter2_UJI =norm(sol_dadmm_datashare_n2.beta - beta_s, 2);
    
    %superconductivity
    t = readtable('train.csv');
    t = table2array(t);
    x = t(:, 1:81);
    y = t(:, 82);
    [n, p] = size(x);
    x_org = x./sqrt(n);
    y_org = y./sqrt(n);
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
    % fix time
    lambda = 1;
    tol_admm = 0;
    max_iter = inf;
    time_limit = 100;
    percentage = 0.05;
    sol_dadmm_t = pd_primal_consensus_admm_1015(x, y, block_num, lambda, max_iter, tol_admm, time_limit);
    sol_dadmm_datashare_t2 = pd_primal_consensus_admm_g2(x, y, block_num, lambda, max_iter, tol_admm, time_limit, percentage);
    report.AL_dadmm_fix_time_superconductivity = norm(sol_dadmm_t.beta - beta_s, 2);
    report.AL_dadmm_datashare_fix_time2_superconductivity =norm(sol_dadmm_datashare_t2.beta - beta_s, 2);  
    % fix number of iteration
    lambda = 1;
    tol_admm = 0;
    max_iter = 200;
    time_limit = inf;
    percentage = 0.05;
    sol_dadmm_n = pd_primal_consensus_admm_1015(x, y, block_num, lambda, max_iter, tol_admm, time_limit);
    sol_dadmm_datashare_n2 = pd_primal_consensus_admm_g2(x, y, block_num, lambda, max_iter, tol_admm, time_limit, percentage);
    report.AL_dadmm_fix_iter_superconductivity  = norm(sol_dadmm_n.beta - beta_s, 2);
    report.AL_dadmm_datashare_fix_iter2_superconductivity  =norm(sol_dadmm_datashare_n2.beta - beta_s, 2);
    
    %portugal
    t = readtable('ElectionData.csv');
    t = removevars(t, {'time','territoryName','Party'});
    t = table2array(t);
    x = t(:, 1:24);
    y = t(:, 25);
    [n, p] = size(x);
    x_org = x./(n);
    y_org = y./(n);
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
    % fix time
    lambda = 1;
    tol_admm = 0;
    max_iter = inf;
    time_limit = 100;
    percentage = 0.05;
    sol_dadmm_t = pd_primal_consensus_admm_1015(x, y, block_num, lambda, max_iter, tol_admm, time_limit);
    sol_dadmm_datashare_t2 = pd_primal_consensus_admm_g2(x, y, block_num, lambda, max_iter, tol_admm, time_limit, percentage);
    report.AL_dadmm_fix_time_portugal = norm(sol_dadmm_t.beta - beta_s, 2);
    report.AL_dadmm_datashare_fix_time2_portugal =norm(sol_dadmm_datashare_t2.beta - beta_s, 2);  
    % fix number of iteration
    lambda = 1;
    tol_admm = 0;
    max_iter = 200;
    time_limit = inf;
    percentage = 0.05;
    sol_dadmm_n = pd_primal_consensus_admm_1015(x, y, block_num, lambda, max_iter, tol_admm, time_limit);
    sol_dadmm_datashare_n2 = pd_primal_consensus_admm_g2(x, y, block_num, lambda, max_iter, tol_admm, time_limit, percentage);
    report.AL_dadmm_fix_iter_portugal  = norm(sol_dadmm_n.beta - beta_s, 2);
    report.AL_dadmm_datashare_fix_iter2_portugal  =norm(sol_dadmm_datashare_n2.beta - beta_s, 2);
    
    % GPU
    t = readtable('sgemm_product.csv');
    t = table2array(t);
    x = t(:,1:14);
    y = mean(t(:,1:14),2);
    [n, p] = size(x);
    x_org = x./sqrt(n);
    y_org = y./sqrt(n);
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
    % fix time
    lambda = 1;
    tol_admm = 0;
    max_iter = inf;
    time_limit = 100;
    percentage = 0.05;
    sol_dadmm_t = pd_primal_consensus_admm_1015(x, y, block_num, lambda, max_iter, tol_admm, time_limit);
    sol_dadmm_datashare_t2 = pd_primal_consensus_admm_g2(x, y, block_num, lambda, max_iter, tol_admm, time_limit, percentage);
    report.AL_dadmm_fix_time_GPU = norm(sol_dadmm_t.beta - beta_s, 2);
    report.AL_dadmm_datashare_fix_time2_GPU =norm(sol_dadmm_datashare_t2.beta - beta_s, 2);  
    % fix number of iteration
    lambda = 1;
    tol_admm = 0;
    max_iter = 200;
    time_limit = inf;
    percentage = 0.05;
    sol_dadmm_n = pd_primal_consensus_admm_1015(x, y, block_num, lambda, max_iter, tol_admm, time_limit);
    sol_dadmm_datashare_n2 = pd_primal_consensus_admm_g2(x, y, block_num, lambda, max_iter, tol_admm, time_limit, percentage);
    report.AL_dadmm_fix_iter_GPU  = norm(sol_dadmm_n.beta - beta_s, 2);
    report.AL_dadmm_datashare_fix_iter2_GPU  =norm(sol_dadmm_datashare_n2.beta - beta_s, 2);
    
    % CT
    t = readtable('slice_localization_data.csv');
    t = table2array(t);
    y = t(:, 386);
    x = t(:, 1:385);
    [n, p] = size(x);
    x_org = x./sqrt(n);
    y_org = y./sqrt(n);
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
    % fix time
    lambda = 1;
    tol_admm = 0;
    max_iter = inf;
    time_limit = 100;
    percentage = 0.05;
    sol_dadmm_t = pd_primal_consensus_admm_1015(x, y, block_num, lambda, max_iter, tol_admm, time_limit);
    sol_dadmm_datashare_t2 = pd_primal_consensus_admm_g2(x, y, block_num, lambda, max_iter, tol_admm, time_limit, percentage);
    report.AL_dadmm_fix_time_CT = norm(sol_dadmm_t.beta - beta_s, 2);
    report.AL_dadmm_datashare_fix_time2_CT =norm(sol_dadmm_datashare_t2.beta - beta_s, 2);  
    % fix number of iteration
    lambda = 1;
    tol_admm = 0;
    max_iter = 200;
    time_limit = inf;
    percentage = 0.05;
    sol_dadmm_n = pd_primal_consensus_admm_1015(x, y, block_num, lambda, max_iter, tol_admm, time_limit);
    sol_dadmm_datashare_n2 = pd_primal_consensus_admm_g2(x, y, block_num, lambda, max_iter, tol_admm, time_limit, percentage);
    report.AL_dadmm_fix_iter_CT = norm(sol_dadmm_n.beta - beta_s, 2);
    report.AL_dadmm_datashare_fix_iter2_CT =norm(sol_dadmm_datashare_n2.beta - beta_s, 2);
    
    % bike_seoul
    t=readtable('SeoulBikeData.csv');
    t = table2array(t(:, 2:11));
    x = t(:, 2:end);
    y = t(:, 1);
    %delete missing data
    ind = find(sum(isnan(x),2));
    x(ind,:)=[];
    y(ind,:)=[];
    [n, p] = size(x);
    x_org = x./sqrt(n);
    y_org = y./sqrt(n);
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
    % fix time
    lambda = 1;
    tol_admm = 0;
    max_iter = inf;
    time_limit = 100;
    percentage = 0.05;
    sol_dadmm_t = pd_primal_consensus_admm_1015(x, y, block_num, lambda, max_iter, tol_admm, time_limit);
    sol_dadmm_datashare_t2 = pd_primal_consensus_admm_g2(x, y, block_num, lambda, max_iter, tol_admm, time_limit, percentage);
    report.AL_dadmm_fix_time_bike_seoul = norm(sol_dadmm_t.beta - beta_s, 2);
    report.AL_dadmm_datashare_fix_time2_bike_seoul =norm(sol_dadmm_datashare_t2.beta - beta_s, 2);  
    % fix number of iteration
    lambda = 1;
    tol_admm = 0;
    max_iter = 200;
    time_limit = inf;
    percentage = 0.05;
    sol_dadmm_n = pd_primal_consensus_admm_1015(x, y, block_num, lambda, max_iter, tol_admm, time_limit);
    sol_dadmm_datashare_n2 = pd_primal_consensus_admm_g2(x, y, block_num, lambda, max_iter, tol_admm, time_limit, percentage);
    report.AL_dadmm_fix_iter_bike_seoul = norm(sol_dadmm_n.beta - beta_s, 2);
    report.AL_dadmm_datashare_fix_iter2_bike_seoul =norm(sol_dadmm_datashare_n2.beta - beta_s, 2);
    
    % bike_beijing 
    t=readtable('day.csv');
    t = table2array(t(:, 3:end));
    x = t(:, 1:13);
    y = t(:, 14);
    %delete missing data
    ind = find(sum(isnan(x),2));
    x(ind,:)=[];
    y(ind,:)=[];
    [n, p] = size(x);
    x_org = x./sqrt(n);
    y_org = y./sqrt(n);
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
    % fix time
    lambda = 1;
    tol_admm = 0;
    max_iter = inf;
    time_limit = 100;
    percentage = 0.05;
    sol_dadmm_t = pd_primal_consensus_admm_1015(x, y, block_num, lambda, max_iter, tol_admm, time_limit);
    sol_dadmm_datashare_t2 = pd_primal_consensus_admm_g2(x, y, block_num, lambda, max_iter, tol_admm, time_limit, percentage);
    report.AL_dadmm_fix_time_bike_beijing = norm(sol_dadmm_t.beta - beta_s, 2);
    report.AL_dadmm_datashare_fix_time2_bike_beijing =norm(sol_dadmm_datashare_t2.beta - beta_s, 2);  
    % fix number of iteration
    lambda = 1;
    tol_admm = 0;
    max_iter = 200;
    time_limit = inf;
    percentage = 0.05;
    sol_dadmm_n = pd_primal_consensus_admm_1015(x, y, block_num, lambda, max_iter, tol_admm, time_limit);
    sol_dadmm_datashare_n2 = pd_primal_consensus_admm_g2(x, y, block_num, lambda, max_iter, tol_admm, time_limit, percentage);
    report.AL_dadmm_fix_iter_bike_beijing = norm(sol_dadmm_n.beta - beta_s, 2);
    report.AL_dadmm_datashare_fix_iter2_bike_beijing =norm(sol_dadmm_datashare_n2.beta - beta_s, 2);
    
    % bias
    t = readtable('Bias_correction_ucl.csv');
    t = table2array(t(:, 3:end));
    x = t(:, 1:22);
    y = t(:, 23);
    ind = find(sum(isnan(x),2));
    x(ind,:)=[];
    y(ind,:)=[];
    [n, p] = size(x);
    x_org = x./sqrt(n);
    y_org = y./sqrt(n);
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
    % fix time
    lambda = 1;
    tol_admm = 0;
    max_iter = inf;
    time_limit = 100;
    percentage = 0.05;
    sol_dadmm_t = pd_primal_consensus_admm_1015(x, y, block_num, lambda, max_iter, tol_admm, time_limit);
    sol_dadmm_datashare_t2 = pd_primal_consensus_admm_g2(x, y, block_num, lambda, max_iter, tol_admm, time_limit, percentage);
    report.AL_dadmm_fix_time_bias = norm(sol_dadmm_t.beta - beta_s, 2);
    report.AL_dadmm_datashare_fix_time2_bias =norm(sol_dadmm_datashare_t2.beta - beta_s, 2);  
    % fix number of iteration
    lambda = 1;
    tol_admm = 0;
    max_iter = 200;
    time_limit = inf;
    percentage = 0.05;
    sol_dadmm_n = pd_primal_consensus_admm_1015(x, y, block_num, lambda, max_iter, tol_admm, time_limit);
    sol_dadmm_datashare_n2 = pd_primal_consensus_admm_g2(x, y, block_num, lambda, max_iter, tol_admm, time_limit, percentage);
    report.AL_dadmm_fix_iter_bias = norm(sol_dadmm_n.beta - beta_s, 2);
    report.AL_dadmm_datashare_fix_iter2_bias =norm(sol_dadmm_datashare_n2.beta - beta_s, 2);
    
    % appliance
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
    % fix time
    lambda = 1;
    tol_admm = 0;
    max_iter = inf;
    time_limit = 100;
    percentage = 0.05;
    sol_dadmm_t = pd_primal_consensus_admm_1015(x, y, block_num, lambda, max_iter, tol_admm, time_limit);
    sol_dadmm_datashare_t2 = pd_primal_consensus_admm_g2(x, y, block_num, lambda, max_iter, tol_admm, time_limit, percentage);
    report.AL_dadmm_fix_time_appliance = norm(sol_dadmm_t.beta - beta_s, 2);
    report.AL_dadmm_datashare_fix_time2_appliance =norm(sol_dadmm_datashare_t2.beta - beta_s, 2);  
    % fix number of iteration
    lambda = 1;
    tol_admm = 0;
    max_iter = 200;
    time_limit = inf;
    percentage = 0.05;
    sol_dadmm_n = pd_primal_consensus_admm_1015(x, y, block_num, lambda, max_iter, tol_admm, time_limit);
    sol_dadmm_datashare_n2 = pd_primal_consensus_admm_g2(x, y, block_num, lambda, max_iter, tol_admm, time_limit, percentage);
    report.AL_dadmm_fix_iter_appliance = norm(sol_dadmm_n.beta - beta_s, 2);
    report.AL_dadmm_datashare_fix_iter2_appliance =norm(sol_dadmm_datashare_n2.beta - beta_s, 2);
    
    % online
    t=readtable('OnlineNewsPopularity.csv');
    t = table2array(t(:, 2:end));
    x = t(:, 1:59);
    y = t(:, 60);
    % further scale by sqrt n 
    [n, p] = size(x);
    x_org = x./(n);
    y_org = y./(n);
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
     % fix time
    lambda = 1;
    tol_admm = 0;
    max_iter = inf;
    time_limit = 100;
    percentage = 0.05;
    sol_dadmm_t = pd_primal_consensus_admm_1015(x, y, block_num, lambda, max_iter, tol_admm, time_limit);
    sol_dadmm_datashare_t2 = pd_primal_consensus_admm_g2(x, y, block_num, lambda, max_iter, tol_admm, time_limit, percentage);
    report.AL_dadmm_fix_time_online = norm(sol_dadmm_t.beta - beta_s, 2);
    report.AL_dadmm_datashare_fix_time2_online  =norm(sol_dadmm_datashare_t2.beta - beta_s, 2);  
    % fix number of iteration
    lambda = 1;
    tol_admm = 0;
    max_iter = 200;
    time_limit = inf;
    percentage = 0.05;
    sol_dadmm_n = pd_primal_consensus_admm_1015(x, y, block_num, lambda, max_iter, tol_admm, time_limit);
    sol_dadmm_datashare_n2 = pd_primal_consensus_admm_g2(x, y, block_num, lambda, max_iter, tol_admm, time_limit, percentage);
    report.AL_dadmm_fix_iter_online = norm(sol_dadmm_n.beta - beta_s, 2);
    report.AL_dadmm_datashare_fix_iter2_online = norm(sol_dadmm_datashare_n2.beta - beta_s, 2);
    
    % summarize of average improvement
    report.average_improvement = 1/14*[(0.0410-0.0401)/0.0410 +(0.0014-0.0015)/0.0014 +(0.0071-0.0069)/0.0071 +(0.0070-0.0030)/0.0030 +(0.6060-0.5514)/0.6060 +(0.6756-0.6173)/0.6756 +((5.0813e-04)-(4.4306e-04))/(5.0813e-04) +(0.0050-0.0041)/0.0050 +(1.3007-1.1834)/1.3007 +(8.1307-6.4449)/8.1307 +((3.8732e-04)-(3.5650e-04))/(3.8732e-04) +(0.0048-0.0043)/0.0048 +(0.8349-0.4429)/0.8349+(0.4645-0.4193)/0.4645];