monkey = 'P';
date_string = '20170630';
data_path = file_locations('Data', '\Project_Data\20160504_COT_precision\data_extracted\monk_p\COT_SpikesCombined\');

addpath(genpath(file_locations('tools', '\neuro_4_0_0')))

load([data_path monkey '_Spikes_' date_string '-data.mat'])    

arraysToInclude = {'G', 'H', 'I', 'J', 'K', 'L'};
channelsIncluded = ismember(SpikeSettings.array_by_channel, arraysToInclude);
SpikeFiringRates = SpikeFiringRates(:,:,channelsIncluded);

samp_rate = 100; %Sample rate, Hz

min_CursorSpeed_prominence = 50;  %Minimum threshold that must be crossed to be considered a cursor speed peak (pixels/s)
min_PeakSpeed_separation = 200;  %200 ms
minRelProminence = 0.5;
minPeakSpeed = 250;

% %Index numbers of alignment points in TrialInfo.align_samples
InstructIndex = find(strcmpi('Instruct', TrialSettings.align_names));
MoveIndex = find(strcmpi('Move', TrialSettings.align_names));
ContactIndex = find(strcmpi('Contact', TrialSettings.align_names));
EndIndex = find(strcmpi('End', TrialSettings.align_names));

n_tr = length(SpikeInfo.start_index);  %Number of trials 
n_neurons = size(SpikeFiringRates,3);
unique_trial_target = unique(TrialInfo.trial_target);
n_targets = length(unique_trial_target); 
trial_length = sum(~isnan(SpikeFiringRates(:,:,1)),2);

%Find cursor speed
[b_cursor,a_cursor] = butter(1, 10/(100/2), 'low');  %10 Hz lowpass filter
filledJoystickPos = fillmissing(JoystickPos_disp, 'previous', 2);
filtJoystickPos_disp = permute(filtfilt(b_cursor,a_cursor,permute(filledJoystickPos, [2,1,3])), [2,1,3]);
filtJoystickPos_disp(isnan(JoystickPos_disp)) = NaN;
CursorSpeed = sqrt( centdiff(filtJoystickPos_disp(:,:,1)',1,5)'.^2 + centdiff(filtJoystickPos_disp(:,:,2)',1,5)'.^2 )*SpikeSettings.samp_rate ;
CursorSpeed = cat(2,NaN(n_tr, 2), CursorSpeed, NaN(n_tr, 2));  %trials x time
CursorSpeedRaw = sqrt( centdiff(filledJoystickPos(:,:,1)',1,5)'.^2 + centdiff(filledJoystickPos(:,:,2)',1,5)'.^2 )*SpikeSettings.samp_rate ;
CursorSpeedRaw = cat(2,NaN(n_tr, 2), CursorSpeedRaw, NaN(n_tr, 2));  %trials x time

%Hack for data collected in Hayden lab with no plexon analog data 
if ~isfield(TrialInfo,'radius_contact_samples_plx')
    TrialInfo.radius_contact_samples_plx = TrialInfo.radius_contact_samples;
end

warning('off', 'signal:findpeaks:largeMinPeakHeight')
for tr = 1:size(CursorSpeed,1)
  
    curr_speeds = CursorSpeed(tr,:);
%     [peakCursorSpeed{tr},maxCursorSpeedIndex{tr},peakWidth{tr},peakProminence{tr}]  = findpeaks(curr_speeds, 'MinPeakDistance', samp_rate*min_PeakSpeed_separation/1000);
    [peakCursorSpeed{tr},maxCursorSpeedIndex{tr},peakWidth{tr},peakProminence{tr}]  = findpeaks(curr_speeds, 'MinPeakHeight', minPeakSpeed);
end
warning('on', 'signal:findpeaks:largeMinPeakHeight')

relProminence = cellfun(@rdivide, peakProminence, peakCursorSpeed, 'UniformOutput',false);
% pts_to_include = cellfun(@(x,y) x>minPeakSpeed & y>minRelProminence, peakCursorSpeed, relProminence, 'UniformOutput',false);
pts_to_include = cellfun(@(y) y>minRelProminence, relProminence, 'UniformOutput',false);

trial_ids_peakVel = zeros(n_tr,1);
speedPeaksTroughs = zeros(n_tr,3);
speedPeaksTroughs_i = zeros(n_tr,3);
speedPeaksRelProminence = zeros(n_tr,1);
speedPeaksPeakWidth = zeros(n_tr,1);
counter = 0;
for tr = 1:n_tr
    curr_num_peaks = sum(pts_to_include{tr});
    trial_ids_peakVel(counter+(1:curr_num_peaks)) = tr;
    speedPeaksTroughs(counter+(1:curr_num_peaks),2) = peakCursorSpeed{tr}(pts_to_include{tr});
    speedPeaksTroughs_i(counter+(1:curr_num_peaks),2) = maxCursorSpeedIndex{tr}(pts_to_include{tr});
    speedPeaksRelProminence(counter+(1:curr_num_peaks)) = relProminence{tr}(pts_to_include{tr});
    speedPeaksPeakWidth(counter+(1:curr_num_peaks)) = peakWidth{tr}(pts_to_include{tr});
  
    curr_peak_indexes = find(pts_to_include{tr});
    for k = 1:length(curr_peak_indexes)
        before_i = find(pts_to_include{tr}(1:(curr_peak_indexes(k)-1)),1,'last');
        after_i = find(pts_to_include{tr}((curr_peak_indexes(k)+1):end),1,'first') + curr_peak_indexes(k);
        if ~isempty(before_i)
            start_index =  maxCursorSpeedIndex{tr}(before_i);
        else
            start_index = 1;
        end
        if ~isempty(after_i)
            end_index =  maxCursorSpeedIndex{tr}(after_i);
        else
            end_index = size(CursorSpeed,2);
        end
        curr_index_range = start_index:speedPeaksTroughs_i(counter+k,2);
        [speedPeaksTroughs(counter+k,1),curr_i] = min(CursorSpeed(tr, curr_index_range));
        speedPeaksTroughs_i(counter+k,1) = curr_index_range(curr_i);
        curr_index_range = speedPeaksTroughs_i(counter+k,2):end_index;
        [speedPeaksTroughs(counter+k,3),curr_i] = min(CursorSpeed(tr, curr_index_range));
        speedPeaksTroughs_i(counter+k,3) = curr_index_range(curr_i);
    end    
    counter = counter+curr_num_peaks;
end

start_samples_peakVel = -50;
end_samples_peakVel = 30;
mid_samples_peakVel = -start_samples_peakVel+1;

min_reaction_time = -20;  %Peak must be 200 ms after instruction

% %Speed peak must be 200ms after instruction and before end of trial (after
% %successful hold of target), also must be 300 ms before end of data (this
% %is just for safety and should not affect almost any trials)
% dataMask_analysis = false(size(CursorSpeed));
% for tr = 1:size(dataMask_analysis,1)
%     if ~any(isnan(TrialInfo.align_samples(tr,:)))
%         dataMask_analysis(tr,(TrialInfo.align_samples(tr,InstructIndex)-min_reaction_time):TrialInfo.align_samples(tr,EndIndex) ) = true;
%     end
% end


%Throwout trials too close to instruction or too late in the trial
throw_out_trials = (speedPeaksTroughs_i(:,2)+min_reaction_time)<TrialInfo.align_samples(trial_ids_peakVel,InstructIndex) | speedPeaksTroughs_i(:,2)>TrialInfo.align_samples(trial_ids_peakVel,EndIndex) | (speedPeaksTroughs_i(:,2)+end_samples_peakVel)>trial_length(trial_ids_peakVel);  
speedPeaksTroughs = speedPeaksTroughs(~throw_out_trials,:);
speedPeaksTroughs_i = speedPeaksTroughs_i(~throw_out_trials,:);
speedPeaksRelProminence = speedPeaksRelProminence(~throw_out_trials,:);
speedPeaksPeakWidth = speedPeaksPeakWidth(~throw_out_trials,:);
trial_ids_peakVel = trial_ids_peakVel(~throw_out_trials,:);



start_samples_Instruct = -29;
end_samples_Instruct = 11;


% dataMask = false(size(CursorSpeed));
% for tr = 1:size(dataMask,1)
%     dataMask(tr,(TrialInfo.align_samples(tr,InstructIndex)-min_reaction_time):(TrialInfo.align_samples(tr,EndIndex)-end_samples_peakVel)) = true;
% end

outsideTarg_mask = true(size(CursorSpeed));
for tr = 1:size(outsideTarg_mask,1)
    for m = 1:(sum(TrialInfo.move_samples_plx(tr,:)>0)-1)
        outsideTarg_mask(tr,TrialInfo.contact_samples_plx(tr,m):TrialInfo.move_samples_plx(tr,m+1)) = false;
    end
    if sum(TrialInfo.move_samples_plx(tr,:)>0)>0
        outsideTarg_mask(tr,TrialInfo.contact_samples_plx(tr,sum(TrialInfo.move_samples_plx(tr,:)>0)):end) = false;
    end
end


%withinTarg_peakVel is true by testing if from trough to peak speed occurs entirely within target
withinTarg_peakVel_flag = false(size(trial_ids_peakVel));
for p =  1:length(trial_ids_peakVel)
    if all( ~outsideTarg_mask(trial_ids_peakVel(p),speedPeaksTroughs_i(p,1):speedPeaksTroughs_i(p,2)) )
        withinTarg_peakVel_flag(p) = true;
    end
end

initTrial_flag = false(size(trial_ids_peakVel)); %Speed peaks that are the first of the trial, not including the too small initial speed peaks
smallInit_flag = false(size(trial_ids_peakVel)); %Speed peaks that are initial but too small and not considered true initial speed peaks
% noInitInTrial_flag = false(size(trial_ids_peakVel)); %Speed peaks that have no initial speed peak
required_initMove = 250;  %We require the intial movement to be at least 150 pixels from the center, 
for p = 1:length(trial_ids_peakVel)
   if all( speedPeaksTroughs_i(p,2) <= speedPeaksTroughs_i(trial_ids_peakVel(p)== trial_ids_peakVel,2) ) %If the current peak is the first in the trial
        counter = 0;
       max_move(p) =  max(sqrt(sum(JoystickPos_disp(trial_ids_peakVel(p),1: speedPeaksTroughs_i(p,3),:).^2,3)));  %Determine how far this current peak when, if > 
       while max_move(p+counter)<required_initMove && (p+counter+1) < length(trial_ids_peakVel) && trial_ids_peakVel(p+counter) == trial_ids_peakVel(p+counter+1)
           smallInit_flag(p+counter) = true;
           counter = counter+1;
           max_move(p+counter) =  max(sqrt(sum(JoystickPos_disp(trial_ids_peakVel(p+counter),1: speedPeaksTroughs_i(p+counter,3),:).^2,3)));
       end
       %Commented out because it's so small a number of trials it's not
       %worth worrying about
%        if speedPeaksTroughs_i(p+counter,2) < (TrialInfo.contact_samples_plx(trial_ids_peakVel(p+counter),1)+10)  %the initial peak must be before or within 100 ms of first contact with target
            initTrial_flag(p+counter) = true;
%         else
%             noInitInTrial_flag(trial_ids_peakVel==trial_ids_peakVel(p+counter)) = true;
%        end
    end
end

analysisPeaks_flag = ~smallInit_flag & ~withinTarg_peakVel_flag; %& ~noInitInTrial_flag;

speedPeaksDist = nan(length(trial_ids_peakVel),1);
speedPeaksThrowout_flag = false(length(trial_ids_peakVel),1);
for tr = 1:max(trial_ids_peakVel)
    curr_peaks = find(trial_ids_peakVel==tr & analysisPeaks_flag);
    if length(curr_peaks)>1
        for k = 2:length(curr_peaks)
            speedPeaksDist(curr_peaks(k)) = speedPeaksTroughs_i(curr_peaks(k),2)-speedPeaksTroughs_i(curr_peaks(k-1),2);
        end
            shortDistances = find(speedPeaksDist(curr_peaks) < samp_rate*min_PeakSpeed_separation/1000);
            if ~isempty(shortDistances)
                currMaxPeaks = speedPeaksTroughs(curr_peaks,2);
                while ~isempty(shortDistances)
                [~,max_i] = max(currMaxPeaks);
                if any(max_i == shortDistances)
                    speedPeaksThrowout_flag(curr_peaks(max_i-1)) = true;
                    shortDistances = shortDistances(max_i ~= shortDistances);
                elseif any((max_i-1) == shortDistances)
                    speedPeaksThrowout_flag(curr_peaks(max_i)) = true;
                     shortDistances = shortDistances(max_i ~= (shortDistances-1));
                end
                currMaxPeaks(max_i) = 0;
                end
            end
    end
end





TrialSpikeTimes = cellfun(@(x)    cellfun(@(y,z) y-z,x,num2cell(SpikeInfo.start_time'),'Uni',false)    ,SpikeTimes,'Uni', false);

TrialSpikeBins = zeros(size(CursorPos,1), size(CursorPos,2), n_neurons);
t_edges = 0:(1/SpikeSettings.samp_rate):(size(CursorPos,2)/SpikeSettings.samp_rate);

for n = 1:n_neurons
    for tr = 1:n_tr
        for t = 1:(length(t_edges)-1)
            TrialSpikeBins(tr,t,n) = sum(TrialSpikeTimes{n}{tr}>t_edges(t) & TrialSpikeTimes{n}{tr}<=t_edges(t+1));
        end
    end
end


num_SpeedPeaks = size(speedPeaksTroughs_i,1);
dataMask_peakVel = zeros(size(speedPeaksTroughs_i,1),size(SpikeFiringRates,2));   
dataMask_Instruct = zeros(size(speedPeaksTroughs_i,1),size(SpikeFiringRates,2));   
for n = 1:size(speedPeaksTroughs_i,1)
    curr_indexes = speedPeaksTroughs_i(n,2) + (start_samples_peakVel:end_samples_peakVel);
    curr_indexes = curr_indexes(curr_indexes>0);
    dataMask_peakVel(n,curr_indexes) = 1;
    if ~isnan(TrialInfo.align_samples(trial_ids_peakVel(n),InstructIndex))
    curr_indexes = TrialInfo.align_samples(trial_ids_peakVel(n),InstructIndex) + (start_samples_Instruct:end_samples_Instruct);
    dataMask_Instruct(n,curr_indexes) = 1;
    end
end

CursorSpeed_peakVel = getMaskedData(CursorSpeed, dataMask_peakVel, trial_ids_peakVel);
TrialSpikeBins_peakVel = getMaskedData(TrialSpikeBins, dataMask_peakVel, trial_ids_peakVel);
TrialSpikeBins_Instruct = getMaskedData(TrialSpikeBins, dataMask_Instruct, trial_ids_peakVel);


start_window_peakVel = -30;
end_window_preVel = start_window_peakVel + (end_samples_Instruct-start_samples_Instruct) + 1;
t_win = mid_samples_peakVel + (start_window_peakVel:end_window_preVel);

for n = 1:size(TrialSpikeBins_peakVel,3)
    [~,p_ttest_peakVel(n),ci_ttest_peakVel(:,n),stats_ttest_peakVel(n)] = ttest( mean(TrialSpikeBins_peakVel(:,t_win,n),2), mean(TrialSpikeBins_Instruct(:,:,n),2)  );
end

% signif_units =p_ttest_peakVel<0.05;
signif_units = p_ttest_peakVel<(0.05/length(p_ttest_peakVel));

SpikeFiringRates = SpikeFiringRates(:,:,signif_units);

save('..\example_data\example_CIPhase_data', 'SpikeFiringRates','CursorSpeed', 'dataMask_peakVel', 'trial_ids_peakVel', 'initTrial_flag')



