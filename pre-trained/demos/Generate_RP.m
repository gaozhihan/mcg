function [ consumed_time ] = Generate_RP( idx_start, idx_end, gpu_idx )
gpuDevice(gpu_idx)
data_mat_path = '/home/data/gaozhihan/GitRepo/denoise_MIL/temp_data/data_mat';
rp_mat_path = '/home/data/gaozhihan/GitRepo/denoise_MIL/temp_data/rp_mat';

% for idx = 0:(num_data - 1)
tic;
for idx = idx_start:idx_end
    fprintf(sprintf('loading data seq_%d.mat\n', idx));
    load(fullfile(data_mat_path, sprintf('seq_%d.mat', idx)), '-mat');
    seq = uint8(seq * 255);
    seq_size = size(seq);
    batch_size = seq_size(2);
    seq_len = seq_size(1);
    rp_batch_dir = fullfile(rp_mat_path, sprintf('rp_%d', idx));
    mkdir(rp_batch_dir);
    for batch_idx = 1:batch_size
        frame = squeeze(seq(seq_len, batch_idx, 1, :, :));
        [candidates_mcg, ucm2_mcg] = im2mcg(frame, 'accurate', 1);
        region_proposal = candidates_mcg.masks;
        rp_path = fullfile(rp_batch_dir, sprintf('rp_%d_idx_%d', idx, batch_idx - 1));
        save(rp_path, 'region_proposal')
    end
end

consumed_time = toc;

end

