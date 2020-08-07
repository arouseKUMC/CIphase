
function [alignedData, median_t] = getMaskedData(data, dataMask, trialNum)
n_signals = size(data,3);

if nargin<3 || isempty(trialNum)
    if max(dataMask(:)) == 1  %If logicals or 0/1 numbers
        n_tr = size(dataMask,1);
        n_samps = sum(logical(dataMask),2);
        max_t = max(n_samps);
        alignedData = nan(n_tr, max_t, n_signals);
        for tr = 1:n_tr
            alignedData(tr,1:n_samps(tr),:) = data(tr, logical(dataMask(tr,:)), :);
        end
    else
        trialNum = unique(dataMask(:));
        trialNum = trialNum(trialNum~=0);
        n_tr = length(trialNum);
        max_t = sum(dataMask(:)==mode(dataMask(dataMask~=0)));
        alignedData = nan(n_tr, max_t, n_signals);
        n_samps = zeros(n_tr,1);
        for tr = 1:n_tr
            [n_samps(tr), curr_tr] = max(sum(dataMask == trialNum(tr),2)); %Warning: only one trial can have indexes
            alignedData(tr,1:n_samps(tr),:) = data(curr_tr, dataMask(curr_tr,:) == trialNum(tr), :);
        end
    end
else
    n_tr = length(trialNum);
    n_samps = sum(logical(dataMask),2);
    max_t = max(n_samps);
    alignedData = nan(n_tr, max_t, n_signals);
    for tr = 1:n_tr
        alignedData(tr,1:n_samps(tr),:) = data(trialNum(tr), logical(dataMask(tr,:)), :);
    end
end
median_t = median(n_samps);
end