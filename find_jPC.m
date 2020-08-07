function jPCs = find_jPC(firingRates, dataMask, varargin)

% if isnumeric(varargin{1})
%     trialNum = varargin{1};
%     varargin = varargin{2:end};

    
    p = inputParser;
    addOptional(p,'trialNum',[],@isnumeric);
    addParameter(p,'num_iterations',3); %Number of times to iterate to find jPCs
    addParameter(p,'samp_rate',100); %Sample rate, Hz
    addParameter(p,'jPC_filt_cutoffs',[1,3]); %Bandpass filter cutoffs, Hz
    addParameter(p,'n_pc_for_jpc',6); %Number of PCs to use in finding jPCs
    addParameter(p,'num_phase_pts',100); %Number of equally spaced phases from 0 to 2*pi for estimating average firing rates
    p.parse(varargin{:});
    options = p.Results;
    trialNum = options.trialNum;

    if nargin<2 || isempty(dataMask)
        dataMask = all(~isnan(firingRates),3);
        trialNum = 1:size(firingRates,1);
    end
    

[b,a] = butter(1, options.jPC_filt_cutoffs/(options.samp_rate/2), 'bandpass');

n_neurons = size(firingRates,3);
n_tr = size(firingRates,1);
[alignedFiringRates, median_t] = calcjPCA.getMaskedData(firingRates, dataMask, trialNum);
median_t = ceil(median_t);
n_aligned_tr = size(alignedFiringRates,1);

for k = 1:options.num_iterations
if ~exist('jPCs', 'var') %For the first iteration, aligned firing rates are averaged as a function of time
    meanFiringRates = squeeze(nanmean(alignedFiringRates(:,1:median_t,:)));
    
end
global_meanFiringRates = nanmean(meanFiringRates,1);
time_meanFiringRates = nanmean(meanFiringRates,2);
% [pcaCoeff,~] = pca(meanFiringRates, 'Centered', false);
[pcaCoeff,~] = pca(meanFiringRates, 'Centered', true);

%%Find jPCs
 pcaGtPCAlignedFiringRates = reshape(reshape( bsxfun(@minus,alignedFiringRates, permute(global_meanFiringRates, [1,3,2])), [], n_neurons)*pcaCoeff(:, 1:options.n_pc_for_jpc), n_aligned_tr, [], options.n_pc_for_jpc);
% pcaGtPCAlignedFiringRates = reshape(reshape(alignedFiringRates, [], n_neurons)*pcaCoeff(:, 1:options.n_pc_for_jpc), n_aligned_tr, [], options.n_pc_for_jpc);
%     pcaGtPCAlignedFiringRates = bsxfun(@minus, pcaGtPCAlignedFiringRates, permute(mean(reshape(pcaGtPCAlignedFiringRates,[],size(global_meanFiringRates,3))), [1,3,2]));
Xt = reshape(pcaGtPCAlignedFiringRates(:,1:1:(end-1),1:options.n_pc_for_jpc), [], options.n_pc_for_jpc);
    Xt_plus1 = reshape(pcaGtPCAlignedFiringRates(:,2:end,1:options.n_pc_for_jpc), [], options.n_pc_for_jpc);
    nan_t_pts = any(isnan(Xt),2) | any(isnan(Xt_plus1),2);
    Xt = Xt(~nan_t_pts,:);
    Xt_plus1 = Xt_plus1(~nan_t_pts,:);
    Xdot = Xt_plus1 - Xt;
    jPCs_PCs = calcjPCA.perform_jPCA(Xt, Xdot);  %Returns pairs of jPCs in PCA space, PC1 and PC2 in jPCs_PCs(:,[1 2]) etc
    jPCs = pcaCoeff(:,1:options.n_pc_for_jpc)*jPCs_PCs;  %Get jPCs in original neural space by multiplying by PCA coefficents by jPC coefficients in the reduced PC space
%     %Test if jPC1 is an increase in firing rate for a majority of units
%     for d = 1:2:size(jPCs,2)
%         if sum(jPCs(:,d)<0) > sum(jPCs(:,d)>0)  %If a majority of units have a negative jPC1 (or 3, 5, ...) coefficient, flip the axes so jPC1 (or 3, 5) corresponds to more increses in firing rate than decreases
%             jPCs(:,[d,d+1]) = -jPCs(:,[d,d+1]);
%         end
%     end
    %Rotate so jPC1 is direction in plane is most positive for an increase in firing rate for a majority of units
%     for d = 1:2:size(jPCs,2)
%         rot_angle = atan2(sum(jPCs(:,d)),sum(jPCs(:,d+1)));  %sum(jPCs(:,d)) is like dotting with [1 1 ... 1 1]
%         jPCs(:,d)   =  cos(rot_angle)*jPCs(:,d) + sin(rot_angle)*jPCs(:,d+1);
%         jPCs(:,d+1) = -sin(rot_angle)*jPCs(:,d) + cos(rot_angle)*jPCs(:,d+1);
%     end

%     %  Calculate the average firing rate as a function of jPC1 phase
%     jPCSpikeFiringRates = reshape(reshape(bsxfun(@minus, firingRates, permute(global_meanFiringRates, [1,3,2])), [], n_neurons)*jPCs, n_tr, [], options.n_pc_for_jpc );
%     aligned_jPCSpikeFiringRates = getMaskedData(jPCSpikeFiringRates, dataMask, trialNum);
%     mean_jPCSpikeFiringRates = squeeze(nanmean(aligned_jPCSpikeFiringRates,1));
%     regress_vals = time_meanFiringRates\cat(2,mean_jPCSpikeFiringRates(:,1:2),ones(size(mean_jPCSpikeFiringRates,1),1));
%     regress_vals = regress_vals(1:2)/norm(regress_vals(1:2));
%     new_jPCs(:,1) = regress_vals(1)*jPCs(:,1) + regress_vals(2)*jPCs(:,2);
%     new_jPCs(:,2) = -regress_vals(2)*jPCs(:,1) + regress_vals(1)*jPCs(:,2);
%     jPCs(:,1:2) = new_jPCs;
   
    jPCSpikeFiringRates = reshape(reshape(bsxfun(@minus, firingRates, permute(global_meanFiringRates, [1,3,2])), [], n_neurons)*jPCs, n_tr, [], options.n_pc_for_jpc );
    tmp = jPCSpikeFiringRates;
    for tr = 1:n_tr
        tmp(tr,any(isnan(tmp(tr,:,:)),3),:) = repmat(tmp(tr,find(~any(isnan(tmp(tr,:,:)),3),1,'last'),:),  [1, sum(any(isnan(tmp(tr,:,:)),3)),1]);  %For filtfilt we make all the nan values at end equal to the last value
    end
    filt_jPCSpikeFiringRates = permute(filtfilt(b,a, permute(tmp,[2,1,3])),[2,1,3]);
    filt_jPCSpikeFiringRates(isnan(jPCSpikeFiringRates)) = NaN;
    
   
    
    %Find phase of jPC1 and jPC2, first and second values of 3rd dimension
    jPCsPhase = NaN([size(jPCSpikeFiringRates(:,:,1)),3]);
    for tr = 1:n_tr
        for dim = 1:2
        tmp = hilbert(filt_jPCSpikeFiringRates(tr,~isnan(filt_jPCSpikeFiringRates(tr,:,dim)),dim));  %Perform Hilbert transform on filtered jPC1
        jPCsPhase(tr,~isnan(filt_jPCSpikeFiringRates(tr,:,dim)),dim) = angle(tmp);    %Get phase of Hilbert transform
        end
    end
    %Save the combination in 3rd value of 3rd dimension, this is CIphi
    jPCsPhase(:,:,3) = CircStat2012a.circ_mean(cat(3,jPCsPhase(:,:,1),wrapToPi(jPCsPhase(:,:,2)+pi/2)),[],3);
    
   
    
%     aligned_jPCPhase = calcjPCA.getMaskedData(jPCsPhase, dataMask, trialNum);    %Get phase aligned with dataMask
%     
% %     tmp = calcjPCA.getMaskedData(jPCSpikeFiringRates, dataMask, trialNum);
% %     tmp2 = squeeze(nanmean(tmp,1));
% %     figure; plot(tmp2(:,1), tmp2(:,2))
% %     figure; plot(tmp2(:,1)); hold on; plot(tmp2(:,2))
%     
%     allPhases = aligned_jPCPhase(:);
%     allFiringRates = reshape( bsxfun(@minus,alignedFiringRates, permute(global_meanFiringRates, [1,3,2])) ,[],n_neurons);
% %     allFiringRates = reshape(alignedFiringRates,[],n_neurons);
%     [meanFiringRates, phase_pts] = calcjPCA.phase_moving_average(allPhases, allFiringRates, options.num_phase_pts);
% %     [~,max_i] = max(sum(meanFiringRates,2));     
% %     rot_angle = wrapToPi(phase_pts(max_i));
    
    allPhases = jPCsPhase(:,:,3);
    allPhases = allPhases(:);
    allFiringRates = reshape(firingRates, [], size(firingRates,3));
    allFiringRates = allFiringRates(~isnan(allPhases),:);
    allPhases = allPhases(~isnan(allPhases));
    [meanFiringRates, phase_pts] = calcjPCA.phase_moving_average(allPhases, allFiringRates, options.num_phase_pts);
   
end


time_meanFiringRates = mean(meanFiringRates,2);
regress_vals = time_meanFiringRates\cat(2, cos(phase_pts'), sin(phase_pts') ,ones(size(phase_pts')) );
regress_vals = regress_vals(1:2)/norm(regress_vals(1:2));
new_jPCs(:,1) = regress_vals(1)*jPCs(:,1) + regress_vals(2)*jPCs(:,2);
new_jPCs(:,2) = -regress_vals(2)*jPCs(:,1) + regress_vals(1)*jPCs(:,2);
jPCs(:,1:2) = new_jPCs; 

% figure; plot(phase_pts,mean(meanFiringRates,2))
% 
%     %  Calculate the average firing rate as a function of jPC1 phase
%     jPCSpikeFiringRates = reshape(reshape(bsxfun(@minus, firingRates, permute(global_meanFiringRates, [1,3,2])), [], n_neurons)*jPCs, n_tr, [], options.n_pc_for_jpc );
%     aligned_jPCSpikeFiringRates = getMaskedData(jPCSpikeFiringRates, dataMask, trialNum);
%     mean_jPCSpikeFiringRates = squeeze(nanmean(aligned_jPCSpikeFiringRates,1));
%     regress_vals = time_meanFiringRates\cat(2,mean_jPCSpikeFiringRates(:,1:2),ones(size(mean_jPCSpikeFiringRates,1),1));
%     regress_vals = regress_vals(1:2)/norm(regress_vals(1:2));
%     new_jPCs(:,1) = regress_vals(1)*jPCs(:,1) + regress_vals(2)*jPCs(:,2);
%     new_jPCs(:,2) = -regress_vals(2)*jPCs(:,1) + regress_vals(1)*jPCs(:,2);
%     jPCs(:,1:2) = new_jPCs;



% for k = 1:4
%         jPCSpikeFiringRates = reshape(reshape(bsxfun(@minus, firingRates, permute(global_meanFiringRates, [1,3,2])), [], n_neurons)*jPCs, n_tr, [], options.n_pc_for_jpc );
% %     jPCSpikeFiringRates = reshape(reshape(firingRates, [], n_neurons)*jPCs, n_tr, [], options.n_pc_for_jpc );
%     tmp = jPCSpikeFiringRates;
%     for tr = 1:n_tr
%         tmp(tr,any(isnan(tmp(tr,:,:)),3),:) = repmat(tmp(tr,find(~any(isnan(tmp(tr,:,:)),3),1,'last'),:),  [1, sum(any(isnan(tmp(tr,:,:)),3)),1]);  %For filtfilt we make all the nan values at end equal to the last value
%     end
%     filt_jPCSpikeFiringRates = permute(filtfilt(b,a, permute(tmp,[2,1,3])),[2,1,3]);
%     filt_jPCSpikeFiringRates(isnan(jPCSpikeFiringRates)) = NaN;
%     jPCsPhase = NaN(size(jPCSpikeFiringRates(:,:,1)));
%     for tr = 1:n_tr
%         tmp = hilbert(filt_jPCSpikeFiringRates(tr,~isnan(filt_jPCSpikeFiringRates(tr,:,1)),1));  %Perform Hilbert transform on filtered jPC1
%         jPCsPhase(tr,~isnan(filt_jPCSpikeFiringRates(tr,:,1))) = angle(tmp);                     %Get phase of Hilbert transform
%     end
%     aligned_jPCPhase = calcjPCA.getMaskedData(jPCsPhase, dataMask, trialNum);    %Get phase aligned with dataMask
%     
%     allPhases = aligned_jPCPhase(:);
%     allFiringRates = reshape( bsxfun(@minus,alignedFiringRates, permute(global_meanFiringRates, [1,3,2])) ,[],n_neurons);
% %     allFiringRates = reshape(alignedFiringRates,[],n_neurons);
%     [meanFiringRates, phase_pts] = calcjPCA.phase_moving_average(allPhases, allFiringRates, options.num_phase_pts);
% %     [~,max_i] = max(sum(meanFiringRates,2));     
% %     rot_angle = wrapToPi(phase_pts(max_i));
%     r = sum(meanFiringRates,2);   %figure; plot(phase_pts,sum(meanFiringRates,2))
% %     x = cos(phase_pts);
% %     y = sin(phase_pts);
% %     b_vals = cat(1,x,y,ones(size(x)))'\r;
% %     rot_angle = atan2(b_vals(2),b_vals(1));
% %     jPCs(:,1)   =  cos(rot_angle)*jPCs(:,1) + sin(rot_angle)*jPCs(:,2);
% %     jPCs(:,2)   = -sin(rot_angle)*jPCs(:,1) + cos(rot_angle)*jPCs(:,2);
%     [~,max_i] = max(r); 
%     num_pts = floor(options.num_phase_pts/8);
%     if max_i<=num_pts
%         curr_range = [((options.num_phase_pts-num_pts+max_i):options.num_phase_pts),1:(max_i+num_pts)];
%     elseif max_i>(options.num_phase_pts-num_pts)
%         curr_range = [((max_i-num_pts):options.num_phase_pts),1:(num_pts-(options.num_phase_pts-max_i))];
%     else
%         curr_range = (max_i-num_pts):(max_i+num_pts);
%     end
%      
%     curr_phase = linspace(-pi/4,pi/4,2*num_pts+1);
%     x = cos(curr_phase);   
%     y = sin(curr_phase);      
%     b_vals = cat(1,x,y,ones(size(x)))'\r(curr_range);
%     rot_angle = atan2(b_vals(2),b_vals(1));
%     rot_angle = phase_pts(max_i)+rot_angle;
%     jPCs(:,1)   =  cos(rot_angle)*jPCs(:,1) + sin(rot_angle)*jPCs(:,2);
%     jPCs(:,2)   = -sin(rot_angle)*jPCs(:,1) + cos(rot_angle)*jPCs(:,2);
% end


    
end


