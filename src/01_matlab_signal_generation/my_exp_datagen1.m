function my_exp_datagen1(varargin)
% function exp_datagen1(path,debug)
% Exemplary function to generate realistic NI-FECG signals
% this script presents the various physiological events modelled by the
% mecg-fecg model. For each example only one of these events is used. 
% The code provides a good start to understant the model parametrization.
% This is in order to better highlight the effect of each individual event
% on the ECG morphology. This code was used in Behar et al 2014, mostly
% default settings from the generator were used.
% 
% Input:
%   path        saving path (default pwd)
%   debug       toggle debug different levels (default 5)
%   wfdb        toggle save output in WFDB format
% 
%
% Cases/events:
% - Case 1 - Baseline
% - Case 5 - Maternal and foetal similar heart rate (alternatively changes
%            in the heart rates)
%
% 
%
% More detailed help is in the <a href="https://fernandoandreotti.github.io/fecgsyn/">FECGSYN website</a>.
%
% Examples:
% exp_datagen1(pwd,5) % generate data and plots
%
% See also:
% exp_datagen2
% exp_datagen3 
% FECGSYNDB_datagen
% 
% --
% fecgsyn toolbox, version 1.2, Jan 2017
% Released under the GNU General Public License
%
% Copyright (C) 2014  Joachim Behar & Fernando Andreotti
% University of Oxford, Intelligent Patient Monitoring Group - Oxford 2014
% joachim.behar@oxfordalumni.org, fernando.andreotti@eng.ox.ac.uk
%
% 
% For more information visit: https://www.physionet.org/physiotools/ipmcode/fecgsyn/
% 
% Referencing this work
%
%   Behar Joachim, Andreotti Fernando, Zaunseder Sebastian, Li Qiao, Oster Julien, Clifford Gari D. 
%   An ECG simulator for generating maternal-foetal activity mixtures on abdominal ECG recordings. 
%   Physiological Measurement.35 1537-1550. 2014.
%
% Last updated : 10-03-2016
% 
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
% 
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <http://www.gnu.org/licenses/>.

%% == check inputs
if nargin >3, error('Too many inputs to data generation function'),end
slashchar = char('/'*isunix + '\'*(~isunix));
optargs = {[pwd slashchar] 5 true};  % default values for input arguments
newVals = cellfun(@(x) ~isempty(x), varargin);
optargs(newVals) = varargin(newVals);
[path,debug,wfdb] = optargs{:};
if ~strcmp(path(end),slashchar), path = [path slashchar];end


%% == parameters for simulations
close all; clc;
THR = 0.2; % threshold of QRS detector
mVCG = 5; % choose mother VCG (if empty then the simulator randomly choose one within the set of available VCGs)
fVCG = 4; % choose foetus VCG (ibid)
CH_CANC = 5; % channel onto which to perform MECG cancellation
POS_DEV = 0; % slight deviation from default hearts and electrodes positions 
             % (0: hard coded values, 1: random deviation and phase initialisation)

%% Additional configuration
% Change debug flag to suppress plots
debug = 0;
% Change debug flag to show ALL plots
%debug = 5;

%% Simulating data
%%% == (1) SIMPLE RUN - BASELINE
close all, clear param out
disp('---- Example (1): SIMPLE RUN - BASELINE ---');
param.fs = 1000; % sampling frequency [Hz]

if ~isempty(mVCG); param.mvcg = mVCG; end;
if ~isempty(fVCG); param.fvcg = fVCG; end;
if ~isempty(POS_DEV); param.posdev = 0; end;
    
out = run_ecg_generator(param,debug);
%out = clean_compress(out);
out = my_clean_compress(out);

% Read parameters from generated signal
my_mhr = out.param.mhr;
my_fhr = out.param.fhr;
my_fs = out.param.fs;

% Save baseline data
filename_baseline_signal_mecg = sprintf("./output_data/baseline/mecg.%4.4d.fs_%d_mhr_%d_fhr_%d.csv", 0, my_fs, my_mhr, my_fhr);
writematrix(out.mecg, filename_baseline_signal_mecg);

filename_baseline_signal_fecg = sprintf("./output_data/baseline/fecg.%4.4d.fs_%d_mhr_%d_fhr_%d.csv", 0, my_fs, my_mhr, my_fhr);
writematrix(out.fecg{1,1}, filename_baseline_signal_fecg);

filename_baseline_mqrs = sprintf("./output_data/baseline/mqrs.%4.4d.fs_%d_mhr_%d_fhr_%d.csv", 0, my_fs, my_mhr, my_fhr);
writematrix(out.mqrs, filename_baseline_mqrs);

filename_baseline_fqrs = sprintf("./output_data/baseline/fqrs.%4.4d.fs_%d_mhr_%d_fhr_%d.csv", 0, my_fs, my_mhr, my_fhr);
writematrix(out.fqrs{1,1}, filename_baseline_fqrs);

filename_baseline_nifecg = sprintf("./output_data/baseline/nifecg.%4.4d.fs_%d_mhr_%d_fhr_%d.csv", 0, my_fs, my_mhr, my_fhr);
writematrix(out.mixture, filename_baseline_nifecg);

%%% == (5) ADDING HEART RATE VARIABILITY
close all, clear param out
disp('---- Example (5): ADDING HEART RATE VARIABILITY ----');
param.fs = 1000;

% Set the number of samples to generate
n_samples = 100;

% Initialise the random generator to make them repeatable
rng(0, 'twister');

% Set the ranges for the random mother heart-rates (MHR)
% [70-90] healthy
% > 140 may damage the baby
a = 70;
b = 90;
% Create a vector of random numbers within the range
random_MHR = a + (b-a).*rand(n_samples,1);

% Print the minimum and maximum heart-rates for this run
r_range_MHR = [min(random_MHR) max(random_MHR)]

% Generate another random vector for the fetal heartrates (FHR)
% [120 160] healthy
% about twice the normal adults HR
a = 120;
b = 160;

%a = 80; % 5 weeks (beginnig)
%b = 180; % 12 weeks

% Create a vector of random numbers within the range
random_FHR = a + (b-a).*rand(n_samples,1);

% Print the minimum and maximum heart-rates for this run
r_range_var_FHR = [min(random_FHR) max(random_FHR)]

for i=1:n_samples
    
    clear param out

    % Case 5a (similar FHR/MHR rates)
    % param.fhr = 135; param.mhr = 130;
    % param.mtypeacc = 'nsr';
    % param.ftypeacc = {'nsr'};

    % Case 5b (heart rates cross-over)
    % param.macc = 40; % maternal acceleration in HR [bpm]
    % param.mtypeacc = 'tanh';
    % param.mhr = 110;
    % param.fhr = 140;

    % Case 5c (similar random FHR/MHR rates)
    % variability FHR = MHR + random([-20, 20])
    param.mhr = floor(random_MHR(i));
    param.fhr = floor(random_FHR(i));
    param.mtypeacc = 'nsr';
    param.ftypeacc = {'nsr'};
    
    if ~isempty(mVCG); param.mvcg = mVCG; end;
    if ~isempty(fVCG); param.fvcg = fVCG; end;
    if ~isempty(POS_DEV); param.posdev = 0; end;
    
    out = run_ecg_generator(param,debug);
    %out = clean_compress(out);
    out = my_clean_compress(out);

    % Read parameters from generated signal
    my_mhr = out.param.mhr;
    my_fhr = out.param.fhr;
    my_fs = out.param.fs;
    
    % Save data
    filename_hrv_signal_mecg = sprintf("./output_data/heart_rate_variable/mecg.%4.4d.fs_%d_mhr_%d_fhr_%d.csv", i, my_fs, my_mhr, my_fhr);
    writematrix(out.mecg, filename_hrv_signal_mecg);
    
    filename_hrv_signal_fecg = sprintf("./output_data/heart_rate_variable/fecg.%4.4d.fs_%d_mhr_%d_fhr_%d.csv", i, my_fs, my_mhr, my_fhr);
    writematrix(out.fecg{1,1}, filename_hrv_signal_fecg);
    
    filename_hrv_mqrs = sprintf("./output_data/heart_rate_variable/mqrs.%4.4d.fs_%d_mhr_%d_fhr_%d.csv", i, my_fs, my_mhr, my_fhr);
    writematrix(out.mqrs, filename_hrv_mqrs);
    
    filename_hrv_fqrs = sprintf("./output_data/heart_rate_variable/fqrs.%4.4d.fs_%d_mhr_%d_fhr_%d.csv", i, my_fs, my_mhr, my_fhr);
    writematrix(out.fqrs{1,1}, filename_hrv_fqrs);
    
    filename_hrv_nifecg = sprintf("./output_data/heart_rate_variable/nifecg.%4.4d.fs_%d_mhr_%d_fhr_%d.csv", i, my_fs, my_mhr, my_fhr);
    writematrix(out.mixture, filename_hrv_nifecg);

    %filename_hrv_mat = sprintf("output_data/heart_rate_variable/out%4.4d_fs_%d_mhr_%d_fhr_%d.mat", i, my_fs, my_mhr, my_fhr);
    %save(filename_hrv_mat, 'out') % save as .mat
   

end

end
