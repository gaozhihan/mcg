
%% Demo to show the results of MCG
clear all;close all;home;

data_mat_path = '/home/data/gaozhihan/GitRepo/denoise_MIL/temp_data/data_mat';
rp_mat_path = '/home/data/gaozhihan/GitRepo/denoise_MIL/temp_data/rp_mat';
file_list = dir(data_mat_path);
num_data = (length(file_list) - 2) / 6; % do not count '.' and '..'

for idx = 0:(num_data - 1)
    load(fullfile(data_mat_path, sprintf('seq_%d.mat', num_data)), '-mat');
    seq = uint8(seq * 255);
    seq_size = size(seq);
    batch_size = seq_size(2);
    seq_len = seq_size(1);
    rp_batch_dir = fullfile(rp_mat_path, sprintf('rp_%d', num_data));
    mkdir(rp_batch_dir);
    for batch_idx = 0:(batch_size - 1)
        frame = seq(seq_len, batch_idx, 1, :, :);
        [candidates_mcg, ucm2_mcg] = im2mcg(frame, 'accurate', 1);
        region_proposal = candidates_mcg.masks;
        rp_path = fullfile(rp_batch_dir, sprintf('rp_%d_idx_%d', num_data, batch_idx));
        save(rp_path, 'region_proposal')
    end
end