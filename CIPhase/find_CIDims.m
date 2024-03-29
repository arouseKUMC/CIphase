%%find_CIDims
%This code identifies the condition-independent neural dimensions with the
%most rotational activity.
%It accompanies Rouse et al 2020
%It uses the jPCA algorithm and code as published in:
%Churchland et al 2012. Neural population dynamics during reaching. Nature, 487: 51-56. 

%firingRates - first argument is neural firing rates and should be trial x time x neurons
%dataMask - second argument is a data mask and is optional, it should be
%      zeros and ones with ones signifying time points within given trials for
%      initial averaging
%       it may have the same number of trials as firingRates or may select
%       trials using the trialNum argument
%trialNum - third argument is also optional, trial number (row of firingRates) that should be extracted for
%   a given row of dataMask 
%See getMaskedData for more information abou the functionality of dataMask and trialNum
%Options, specifice as string + value pairs:
%n_pc_for_jpc, Number Dimensions to use from PCA of neural activity to
%   submit to jPCA, Default = 6
%jPC_filt_cutoffs, high and low pass fiter edges, Default = [0.5,5]
%samp_rate, sampling rate of the neural firing rates
%num_iterations, number of times to realign the condition-independent
%   information, my observation is it doesn't change much with 3 or more, Default = 3
%num_phase_pts, number of discrete phase points from 0 to 2pi for averaging
%   condition independent activity

function CIDims = find_CIDims(firingRates, dataMask, trialNum, varargin)

% if isnumeric(varargin{1})
%     trialNum = varargin{1};
%     varargin = varargin{2:end};

    
    p = inputParser;
    addParameter(p,'num_iterations',3); %Number of times to iterate to find jPCs
    addParameter(p,'samp_rate',100); %Sample rate, Hz
    addParameter(p,'jPC_filt_cutoffs',[0.5,5]); %Bandpass filter cutoffs, Hz
    addParameter(p,'n_pc_for_jpc',6); %Number of PCs to use in finding jPCs
    addParameter(p,'num_phase_pts',100); %Number of equally spaced phases from 0 to 2*pi for estimating average firing rates
    p.parse(varargin{:});
    options = p.Results;
    
    if nargin<2 || isempty(dataMask)
        dataMask = all(~isnan(firingRates),3);
    end
    if nargin<3 || isempty(trialNum)
        trialNum = 1:size(firingRates,1);
    end
    
%Create filter
[b,a] = butter(1, options.jPC_filt_cutoffs/(options.samp_rate/2), 'bandpass');

n_neurons = size(firingRates,3);
n_tr = size(firingRates,1);
%Use getMaskedData to pull out aligned firing rates, in the example this is
%all firing rates as they are aligned with peak speed
[alignedFiringRates, median_t] = getMaskedData(firingRates, dataMask, trialNum);
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
    Xt = reshape(pcaGtPCAlignedFiringRates(:,1:1:(end-1),1:options.n_pc_for_jpc), [], options.n_pc_for_jpc);
    Xt_plus1 = reshape(pcaGtPCAlignedFiringRates(:,2:end,1:options.n_pc_for_jpc), [], options.n_pc_for_jpc);
    nan_t_pts = any(isnan(Xt),2) | any(isnan(Xt_plus1),2);
    Xt = Xt(~nan_t_pts,:);
    Xt_plus1 = Xt_plus1(~nan_t_pts,:);
    Xdot = Xt_plus1 - Xt;
    jPCs_PCs = perform_jPCA(Xt, Xdot);  %Returns pairs of jPCs in PCA space, PC1 and PC2 in jPCs_PCs(:,[1 2]) etc
    CIDims = pcaCoeff(:,1:options.n_pc_for_jpc)*jPCs_PCs;  %Get jPCs in original neural space by multiplying by PCA coefficents by jPC coefficients in the reduced PC space
    
    
    jPCSpikeFiringRates = reshape(reshape(bsxfun(@minus, firingRates, permute(global_meanFiringRates, [1,3,2])), [], n_neurons)*CIDims, n_tr, [], options.n_pc_for_jpc );
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
    jPCsPhase(:,:,3) = circ_mean(cat(3,jPCsPhase(:,:,1),wrapToPi(jPCsPhase(:,:,2)+pi/2)),3);
    
    %Average the firing rates as a function of the CIphi phase values,
    %these new condition-independent firing rates will be used in the next
    %iteration of PCA and jPCA
    allPhases = jPCsPhase(:,:,3);
    allPhases = allPhases(:);
    allFiringRates = reshape(firingRates, [], size(firingRates,3));
    allFiringRates = allFiringRates(~isnan(allPhases),:);
    allPhases = allPhases(~isnan(allPhases));
    [meanFiringRates, phase_pts] = phase_moving_average(allPhases, allFiringRates, options.num_phase_pts);
    
end

%Rotate data so CIDim(:,1) algins with the maximum firing rate when the
%CIphi = 0
time_meanFiringRates = mean(meanFiringRates,2);
regress_vals = time_meanFiringRates\cat(2, cos(phase_pts'), sin(phase_pts') ,ones(size(phase_pts')) );
regress_vals = regress_vals(1:2)/norm(regress_vals(1:2));
new_CIDims(:,1) = regress_vals(1)*CIDims(:,1) + regress_vals(2)*CIDims(:,2);
new_CIDims(:,2) = -regress_vals(2)*CIDims(:,1) + regress_vals(1)*CIDims(:,2);
CIDims(:,1:2) = new_CIDims;

end


