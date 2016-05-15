% set global parameters
max_samples = 10000;
resolution = 10;
samples = [0,linspace(0,10000,1000)];

% verifying resolution legality
if mod(max_samples,resolution)~=0
    error('resolution doesnt sit on grid with max samples');
end

% set final res containers
in_max_total = zeros((max_samples/resolution)+1,1);
out_max_total = zeros((max_samples/resolution)+1,1);

% evaluating means
for sample_num = 0:resolution:max_samples
    
    % generating samples
    M_pos = rand(sample_num,1);
    M_neg = rand(sample_num,1);

    % generating inner sum max
    in_max_total((sample_num/10)+1) = max(mean(M_pos),mean(M_neg));

    % generating outer sum max
    out_max_total((sample_num/10)+1) = mean(max(M_pos, M_neg));
end

% taking the average of both
avg_max_total = (in_max_total+out_max_total)./2;

% plotting graphs
figure,
plot(samples, in_max_total, 'b', samples, out_max_total, 'r');
plot(samples, avg_max_total, 'g');