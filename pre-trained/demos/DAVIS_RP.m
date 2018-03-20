function [ consumed_time ] = DAVIS_RP( idx_start, idx_end, gpu_idx )

data_root_dir = '/home/data/gaozhihan/project/davis-2017/data/DAVIS/JPEGImages/480p';
rp_root_dir = '/home/data/gaozhihan/project/davis-2017/data/DAVIS/RegionProposals/480p';
data_list = dir(data_root_dir); 
data_list = data_list(3: end); % the first two elements are "." and ".."

gpuDevice(gpu_idx)
for idx = idx_start: idx_end
    disp(['generating rps for sequence ', data_list(idx).name]);
    sequence_dir = fullfile(data_root_dir, data_list(idx).name);
    rp_sequence_dir = fullfile(rp_root_dir, data_list(idx).name);
    mkdir(rp_sequence_dir);
    
    frame_list = dir(sequence_dir);
    frame_list = frame_list(3: end);
    for frame_idx = 1:length(frame_list)
        tic;
        frame_name = frame_list(frame_idx).name;
        disp(['generating rps for frame ', data_list(idx).name, ', ', frame_name]);
        frame_path = fullfile(sequence_dir, frame_name);
        
        [~, rp_name, ~] = fileparts(frame_name);
%         rp_name = [rp_name, '.mat'];
        rp_path = fullfile(rp_sequence_dir, rp_name);
        
        frame = imread(frame_path);
        [candidates_mcg, ~] = im2mcg(frame, 'accurate', 1);
        region_proposal = candidates_mcg.masks;
        save(rp_path, 'region_proposal');
        consumed_time = toc;
        disp(['consumed_time: ', num2str(consumed_time)])
    end
end
           
        