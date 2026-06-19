function Alternating_soundsource(comb_table)
% ALTERNATING_SOUNDSOURCE  MATLAB port of notebooks/Alternating_soundsource.py
%
% Per-cell sound-source representation analysis for the "soso" (alternating
% speaker) paradigm. Reproduces the Python figures and statistics:
%   - per-cell raster / PSTH / polar mosaic (PDF)
%   - peak firing rate per speaker, sorted by circular distance to PD
%   - AUC per speaker (sorted / unsorted)
%   - boxplots + Friedman + post-hoc (Nemenyi-style via multcompare)
%   - PSTH correlation matrix / histogram / per-cell boxplots
%   - mean PSTH and mean whisker-pad motion line plots
%   - combined summary mosaic
%
% INPUT
%   comb_table : MATLAB table in the long format (one row per Condition per
%                cell). Baseline rows hold the HD tuning fields
%                (HDRateSmooth, HDAngle, ...); soso rows hold the per-speaker
%                raster / motion data. Rows are matched on (Animal_Id, Cell_Id).
%                If called with no argument, expects a variable named
%                comb_table in the base workspace.
%
% Requires: Statistics and Machine Learning Toolbox (friedman, multcompare).
%
% NOTE ON COLUMN NAMES
%   The per-speaker raster/whisk variables could not be named unambiguously
%   from the table display (they appear as numbered sub-columns). Set the
%   correct names in the CONFIG block below. If a configured name is missing,
%   the script prints all available variable names and stops so you can fix it.

% ====================================================================
% CONFIG — map logical names to actual table variable names
% ====================================================================
cfg.var_condition    = 'Condition';
cfg.var_animal       = 'Animal_Id';
cfg.var_cell         = 'Cell_Id';
cfg.var_hd_rate      = 'HDRateSmooth';   % 1x37 HD tuning (Baseline row)
cfg.var_hd_angle     = 'HDAngle';        % preferred direction [deg] (Baseline row)
cfg.var_hd_peakfr    = 'HDpeakFR';
cfg.var_hd_fr        = 'HDFR';
cfg.var_hd_inx       = 'HDInx';
cfg.var_pvalr        = 'pValR';
cfg.var_nstims       = 'nstims';         % stim count (soso row)

% per-speaker data (soso row), stored as N x 4 cell columns ordered {a,w,e,r}.
% The accessor getSpeakerData() resolves: an N x 4 cell (col k = speaker),
% a struct with fields a/w/e/r, a 1xK cell, or separate <base>_a..<base>_r cols.
cfg.var_raster_times = 'RasterTimes';    % N x 4 cell, spike times [ms] per speaker
cfg.var_raster_rows  = 'RasterRows';     % N x 4 cell, trial-row index per spike
cfg.var_whisk_psth   = 'whisk_psth';     % N x 4 cell, whisk chunks (trials x time) per speaker
cfg.var_trigger_time = 'trigger_time';   % motion time vector [s] (soso row)

baseline_label = "Baseline";
soso_label     = "soso";

speakers   = {'a','w','e','r'};
speaker_position = [93, 178, 272, 356];   % deg, order matches `speakers`
distance_label = {'Closest','Second Closest','Second Farthest','Farthest'};

% analysis constants (match Python)
half_window = 1500;        % ms
time_bin    = 0.012;       % s
nbins       = round(half_window / (time_bin * 1000));
min_nstims  = 40;          % filter: keep cells with n_stims > 40
ybox_frac   = 0.30;        % boxplot y-axis padding (centred / condensed boxes)

DIRECTIONS = linspace(0, 360, 37);

% ====================================================================
% JOIN long -> per-cell (match Baseline + soso rows)
% ====================================================================
vn = comb_table.Properties.VariableNames;
requireVar(vn, {cfg.var_condition, cfg.var_animal, cfg.var_cell});

cond   = string(comb_table.(cfg.var_condition));
animal = string(comb_table.(cfg.var_animal));
cellid = comb_table.(cfg.var_cell);

isBase = cond == baseline_label;
isSoso = cond == soso_label;

% unique cells that have a soso row
sosoIdx = find(isSoso);
nC0 = numel(sosoIdx);

cells = struct('animal',{},'cellid',{},'baseRow',{},'sosoRow',{});
for k = 1:nC0
    r = sosoIdx(k);
    a = animal(r); c = cellid(r);
    bMatch = find(isBase & animal == a & cellid == c, 1, 'first');
    if isempty(bMatch)
        warning('No Baseline match for Animal %s Cell %s — skipped.', a, string(c));
        continue
    end
    cells(end+1).animal = a;            %#ok<AGROW>
    cells(end).cellid   = c;
    cells(end).baseRow  = bMatch;
    cells(end).sosoRow  = r;
end

% n_stims filter
nstimsAll = comb_table.(cfg.var_nstims);
keep = false(1, numel(cells));
for i = 1:numel(cells)
    keep(i) = nstimsAll(cells(i).sosoRow) > min_nstims;
end
cells = cells(keep);
nCells = numel(cells);
fprintf('%d cells after n_stims > %d filter.\n', nCells, min_nstims);
if nCells == 0
    error('No cells left after filtering.');
end

% ====================================================================
% raster bins
% ====================================================================
raster_edges = linspace(-half_window, half_window, nbins*2 + 1);
bins_plot    = raster_edges(1:end-1) + diff(raster_edges)/2;   % bin centres
resp_window  = find(bins_plot > 1 & bins_plot < 300);

% ====================================================================
% extract per-cell data into arrays
% ====================================================================
% preferred direction & info
preferred = zeros(nCells,1);
HDpeakFR  = nan(nCells,1);
HDFR      = nan(nCells,1);
HDInx     = nan(nCells,1);
pValR     = nan(nCells,1);
HDrate    = cell(nCells,1);   % 1x37 tuning curves

for i = 1:nCells
    b = cells(i).baseRow;
    preferred(i) = getScalar(comb_table, b, cfg.var_hd_angle);
    HDpeakFR(i)  = getScalar(comb_table, b, cfg.var_hd_peakfr);
    HDFR(i)      = getScalar(comb_table, b, cfg.var_hd_fr);
    HDInx(i)     = getScalar(comb_table, b, cfg.var_hd_inx);
    pValR(i)     = getScalar(comb_table, b, cfg.var_pvalr);
    HDrate{i}    = rowvec(getCellOrArray(comb_table, b, cfg.var_hd_rate));
end

% per-speaker raster + whisk
rasterTimes = cell(nCells,4);
rasterRows  = cell(nCells,4);
whiskAvg    = cell(nCells,4);
trigger_time = [];

for i = 1:nCells
    s = cells(i).sosoRow;
    rt = getSpeakerData(comb_table, s, cfg.var_raster_times, speakers, vn);
    rr = getSpeakerData(comb_table, s, cfg.var_raster_rows,  speakers, vn);
    wp = getSpeakerData(comb_table, s, cfg.var_whisk_psth,   speakers, vn);
    for k = 1:4
        rasterTimes{i,k} = rowvec(rt{k});
        rasterRows{i,k}  = rowvec(rr{k});
        chunk = wp{k};                       % trials x time
        if isempty(chunk)
            whiskAvg{i,k} = [];
        else
            whiskAvg{i,k} = mean(chunk, 1, 'omitnan');   % mean over trials -> 1 x time
        end
    end
    if isempty(trigger_time)
        trigger_time = rowvec(getCellOrArray(comb_table, s, cfg.var_trigger_time));
    end
end

% ====================================================================
% build PSTH firing-rate array  new_rate : nCells x nBins x 4
% ====================================================================
new_rate = zeros(nCells, numel(raster_edges)-1, 4);
for i = 1:nCells
    for k = 1:4
        t = rasterTimes{i,k};
        r = rasterRows{i,k};
        if ~isempty(t)
            counts = histcounts(t, raster_edges);
            new_rate(i,:,k) = counts / (max(r) * time_bin);
        end
    end
end

% whisk_avg -> nCells x nT x 4, baseline-subtracted (median over trigger_time<=0)
nT = numel(trigger_time);
whisk_avg = nan(nCells, nT, 4);
for i = 1:nCells
    for k = 1:4
        w = whiskAvg{i,k};
        m = min(numel(w), nT);
        if m > 0
            whisk_avg(i,1:m,k) = w(1:m);
        end
    end
end
baseline_subtract_whisk = nan(size(whisk_avg));
pre = trigger_time <= 0;
for k = 1:4
    bm = median(whisk_avg(:,pre,k), 2, 'omitnan');
    baseline_subtract_whisk(:,:,k) = whisk_avg(:,:,k) - bm;
end

% ====================================================================
% FIGURE: per-cell mosaic PDF  (raster / PSTH / whisk line / polar)
% ====================================================================
individuals_flag = true;
if individuals_flag
    pdf_path = fullfile(pwd, 'matlab_soso_indiv_cells.pdf');
    if exist(pdf_path, 'file'); delete(pdf_path); end
    tmin = -100; tmax = 500;
    for i = 1:nCells
        fig = figure('Visible','off','Position',[100 100 1500 1200]);
        tl = tiledlayout(fig, 4, 4, 'TileSpacing','compact','Padding','compact');
        for enum = 1:4
            key = speakers{enum};
            % raster
            ax_r = nexttile(tl, (enum-1)*4 + 1);
            scatter(ax_r, rasterTimes{i,enum}, rasterRows{i,enum}, 6, '.');
            xlim(ax_r, [tmin tmax]);
            ylabel(ax_r, sprintf('Speaker %d°', speaker_position(enum)), ...
                'FontSize', 10, 'FontWeight','bold');
            despine(ax_r);
            % PSTH
            ax_p = nexttile(tl, (enum-1)*4 + 2);
            bar(ax_p, bins_plot, reshape(new_rate(i,:,enum),1,[]), 1, 'EdgeColor','none');
            xlim(ax_p, [tmin tmax]); ylabel(ax_p, 'Firing Rate [Hz]'); despine(ax_p);
            % whisk line
            ax_l = nexttile(tl, (enum-1)*4 + 3);
            plot(ax_l, trigger_time, squeeze(baseline_subtract_whisk(i,:,enum)));
            xlim(ax_l, [tmin tmax]/1000); ylim(ax_l, [-0.2 0.8]); despine(ax_l);
        end
        % polar (top-right tile) + info (tile 8)
        ax_pol = polaraxes(tl); ax_pol.Layout.Tile = 4;
        polarplot(ax_pol, deg2rad(DIRECTIONS), HDrate{i});
        hold(ax_pol,'on');
        polarscatter(ax_pol, deg2rad(speaker_position), ones(1,4), 40, 'filled');
        thetaticks(ax_pol, sort(speaker_position));
        ax_info = nexttile(tl, 8); axis(ax_info,'off');
        infotext = sprintf(['Animal: %s\nCell:   %s\nPD:     %.2f\n' ...
            'PeakFR: %.2f\nMeanFR: %.2f\nHDInx:  %.2f\npValR:  %.2g\n' ...
            'binsize:%.2f ms'], cells(i).animal, string(cells(i).cellid), ...
            preferred(i), HDpeakFR(i), HDFR(i), HDInx(i), pValR(i), 1000*time_bin);
        text(ax_info, 0, 1, infotext, 'VerticalAlignment','top', 'FontName','FixedWidth');
        exportgraphics(fig, pdf_path, 'Append', true);
        close(fig);
    end
    fprintf('Per-cell mosaic written to %s\n', pdf_path);
end

% ====================================================================
% PEAK FR sorted by circular distance to PD
% ====================================================================
speaker_angles = speaker_position;
diffd  = preferred - speaker_angles;          % nCells x 4 (broadcast)
wrapped = mod(diffd + 180, 360) - 180;
dist_deg = abs(wrapped);
[dist_sorted, idx_sorted] = sort(dist_deg, 2);

resp_peak_fr = squeeze(max(new_rate(:,resp_window,:), [], 2));   % nCells x 4
resp_peak_sorted = takeAlong(resp_peak_fr, idx_sorted);

xlab = distance_label;

% scatter: peak FR vs distance, one panel per rank
figure('Position',[100 100 1400 300]);
for i = 1:4
    subplot(1,4,i);
    scatter(resp_peak_sorted(:,i), dist_sorted(:,i), 12, 'filled', ...
        'MarkerFaceAlpha', 0.7);
    xlabel(xlab{i});
    if i == 1; ylabel('Circular distance to speaker (deg)'); end
end

% boxplot: max FR (unsorted, canonical speaker order)
figure('Position',[100 100 1000 500]);
ax = axes; %#ok<LAXES>
boxplot(ax, resp_peak_fr, speakers, 'Symbol','');
hold(ax,'on');
plotPairedLines(ax, resp_peak_fr);
padYLim(ax, resp_peak_fr(:), ybox_frac);
xlabel(ax,'Speaker'); ylabel(ax,'Max Psth'); title(ax,'max psth');

[p_unsorted, ~, ~] = friedman(resp_peak_fr, 1, 'off');
fprintf('%s\nMax FR not sorted\nFriedman p = %.4f\n%s\n', repmat('-',1,70), p_unsorted, repmat('-',1,70));

% boxplot: max FR sorted by distance
figure('Position',[100 100 1200 500]);
ax = axes; %#ok<LAXES>
boxplot(ax, resp_peak_sorted, distance_label, 'Symbol','');
hold(ax,'on');
plotPairedLines(ax, resp_peak_sorted);
padYLim(ax, resp_peak_sorted(:), ybox_frac);
ylabel(ax,'Fr [Hz]'); xlabel(ax,'Speaker position relative to PD');
title(ax,'Max FR sorted by speaker distance');

[p_fr, ~, stats_fr] = friedman(resp_peak_sorted, 1, 'off');
fprintf('%s\nMax FR sorted\nFriedman p = %.4f\n', repmat('-',1,70), p_fr);
printPosthoc(stats_fr, distance_label);

% ====================================================================
% AUC (integral over response bins)
% ====================================================================
auc_resp = squeeze(sum(new_rate(:,resp_window,:) * time_bin, 2));   % nCells x 4

% unsorted
figure('Position',[100 100 1000 500]);
ax = axes; %#ok<LAXES>
boxplot(ax, auc_resp, speakers);
hold(ax,'on'); plotPairedLines(ax, auc_resp);
padYLim(ax, auc_resp(:), ybox_frac);
title(ax,'auc unsorted'); ylabel(ax,'AUC [spikes]');
[p_auc, ~, stats_auc] = friedman(auc_resp, 1, 'off');
fprintf('%s\nAUC unsorted\nFriedman p = %.4f\n', repmat('-',1,70), p_auc);
printPosthoc(stats_auc, speakers);

% sorted
auc_resp_sorted = takeAlong(auc_resp, idx_sorted);
figure('Position',[100 100 1200 500]);
ax = axes; %#ok<LAXES>
boxplot(ax, auc_resp_sorted, distance_label);
hold(ax,'on'); plotPairedLines(ax, auc_resp_sorted);
padYLim(ax, auc_resp_sorted(:), ybox_frac);
title(ax,'auc sorted'); ylabel(ax,'AUC [spikes]');
[p_aucs, ~, stats_aucs] = friedman(auc_resp_sorted, 1, 'off');
fprintf('%s\nAUC sorted\nFriedman p = %.4f\n', repmat('-',1,70), p_aucs);
printPosthoc(stats_aucs, distance_label);

% ====================================================================
% PSTH correlations per cell  (distance-sorted, baseline-subtracted)
% ====================================================================
new_rate_sort = takeAlong3(new_rate, idx_sorted);     % nCells x nBins x 4
baseline_time_idx = bins_plot > -1000 & bins_plot < 0;
baseline_mean = mean(new_rate_sort(:,baseline_time_idx,:), 2);
new_rate_bs = new_rate_sort - baseline_mean;          % no smoothing (matches Python)

avg_whisker_sort = takeAlong3(whisk_avg, idx_sorted);
baseline_whisk_idx = trigger_time > -1000 & trigger_time < 0;
baseline_mean_whisk = mean(avg_whisker_sort(:,baseline_whisk_idx,:), 2, 'omitnan');
avg_whisker_bs = avg_whisker_sort - baseline_mean_whisk;

response_data = new_rate_bs(:, resp_window, :);
n_speakers = 4;
corr_all = zeros(nCells, n_speakers, n_speakers);
for c = 1:nCells
    M = squeeze(response_data(c,:,:));     % time x 4
    corr_all(c,:,:) = corrcoef(M);
end
mean_corr_matrix = squeeze(mean(corr_all, 1));

% heatmap
figure('Position',[100 100 500 500]);
imagesc(mean_corr_matrix, [0 1]); colormap(viridisish()); colorbar;
set(gca,'XTick',1:4,'XTickLabel',xlab,'YTick',1:4,'YTickLabel',xlab);
xtickangle(45); title('Mean PSTH correlation across cells');

% unique pairwise correlations
triu_mask = triu(true(n_speakers), 1);
[ii, jj] = find(triu_mask);
corr_pairs = zeros(nCells, numel(ii));
for c = 1:nCells
    cm = squeeze(corr_all(c,:,:));
    corr_pairs(c,:) = cm(triu_mask);
end
corr_values = corr_pairs(:);
mean_corr_per_cell = mean(corr_pairs, 2);

% histogram of all pairwise correlations
figure('Position',[100 100 500 400]);
histogram(corr_values, 20); xlim([0 1]);
xlabel('correlations (r)'); ylabel('Count');
title('Distribution of PSTH correlations');

% per-cell mean correlation scatter
figure('Position',[100 100 500 400]);
scatter(1:numel(mean_corr_per_cell), mean_corr_per_cell, 'filled');
xlabel('Cell'); ylabel('Mean correlation'); title('Correlation per cell');
padYLim(gca, mean_corr_per_cell, ybox_frac);

% combined: heatmap | distribution | per-cell boxplot
figure('Position',[100 100 1500 400]);
subplot(1,3,1);
imagesc(mean_corr_matrix,[0 1]); colormap(viridisish()); colorbar;
set(gca,'XTick',1:4,'XTickLabel',distance_label,'YTick',1:4,'YTickLabel',distance_label);
xtickangle(45); title('Mean correlation matrix');
subplot(1,3,2);
histogram(corr_values, 'Normalization','pdf', 'FaceColor',[0.27 0.51 0.71], ...
    'FaceAlpha',0.7, 'EdgeColor','k'); hold on;
mv = mean(corr_values);
xline(mv,'--r','LineWidth',2,'Label',sprintf('Mean = %.2f', mv));
[f,xi] = ksdensity(corr_values, linspace(0,1,500));
plot(xi, f, 'Color',[0 0 0.55], 'LineWidth',2);
xlim([0 1]); xlabel('Correlation (r)'); ylabel('Probability density');
title('Correlation distribution'); despine(gca);
subplot(1,3,3);
boxplot(mean_corr_per_cell, 'Symbol',''); hold on;
xj = 1 + (rand(numel(mean_corr_per_cell),1)-0.5)*0.16;
scatter(xj, mean_corr_per_cell, 40, 'k', 'filled', 'MarkerFaceAlpha',0.7);
set(gca,'XTick',[]); title('Mean correlation per cell');
padYLim(gca, mean_corr_per_cell, ybox_frac); despine(gca);

fprintf('Mean correlation: %.3f ± %.3f\n%s\n', mean(corr_values), std(corr_values), repmat('-',1,70));

% ====================================================================
% MEAN PSTH and MEAN whisker-pad motion
% ====================================================================
mean_psth = squeeze(mean(new_rate_bs, 1));
sd_psth   = squeeze(std(new_rate_bs, 0, 1));
plot_resp = (resp_window(1)-50):(resp_window(end)+100);
plot_resp(plot_resp < 1 | plot_resp > numel(bins_plot)) = [];

mean_whisk    = squeeze(mean(avg_whisker_bs, 1, 'omitnan'));
sd_whisk      = squeeze(std(avg_whisker_bs, 0, 1, 'omitnan'));
plot_resp_whisk = trigger_time <= 0.5 & trigger_time >= -0.1;

figure('Position',[100 100 1500 600]);
ax1 = subplot(1,2,1); hold(ax1,'on');
x = bins_plot(plot_resp) / 1000;
for k = 1:4
    y = mean_psth(plot_resp,k)'; sem = sd_psth(plot_resp,k)';
    fillBand(ax1, x, y, sem);
    plot(ax1, x, y, 'DisplayName', distance_label{k}, 'LineWidth',1.2);
end
xlabel(ax1,'Time (s)'); ylabel(ax1,'Baseline-subtracted firing rate');
legend(ax1,'show'); despine(ax1);

ax2 = subplot(1,2,2); hold(ax2,'on');
xw = trigger_time(plot_resp_whisk);
for k = 1:4
    y = mean_whisk(plot_resp_whisk,k)'; sem = sd_whisk(plot_resp_whisk,k)';
    fillBand(ax2, xw, y, sem);
    plot(ax2, xw, y, 'DisplayName', distance_label{k}, 'LineWidth',1.2);
end
xlabel(ax2,'Time (s)'); ylabel(ax2,'Baseline-subtracted Whisker pad');
legend(ax2,'show'); despine(ax2);

% ====================================================================
% pairwise correlation by rank separation (boxplot + per-cell scatter)
% ====================================================================
pair_labels = arrayfun(@(a,b) sprintf('%s – %s', distance_label{a}, distance_label{b}), ...
    ii, jj, 'UniformOutput', false);
pair_rank_dist = jj - ii;
[~, sidx] = sort(pair_rank_dist);
corr_pairs_s = corr_pairs(:, sidx);
pair_labels  = pair_labels(sidx);

figure('Position',[100 100 800 500]); ax = axes; hold(ax,'on'); %#ok<LAXES>
boxplot(ax, corr_pairs_s, pair_labels, 'Symbol','', 'Widths',0.5);
nP = size(corr_pairs_s,2);
cmap = lines(nCells);
for c = 1:nCells
    jit = (rand(1,nP)-0.5)*0.3;
    scatter(ax, (1:nP)+jit, corr_pairs_s(c,:), 50, cmap(c,:), 'filled', ...
        'MarkerEdgeColor','k', 'MarkerFaceAlpha',0.9);
end
xtickangle(ax,65); ylabel(ax,'PSTH correlation (r)'); xlabel(ax,'Speaker pair');
padYLim(ax, corr_pairs_s(:), ybox_frac); despine(ax);

[p_corr, ~, stats_corr] = friedman(corr_pairs_s, 1, 'off');
fprintf('Friedman p = %.3f (pairwise correlations)\n', p_corr);
printPosthoc(stats_corr, pair_labels);

% ====================================================================
% COMBINED summary mosaic   (5x4 grid)
%   col1: polar (row1) + 4 PSTHs (rows2-5)
%   col2: 4 whisk lines (rows2-5)
%   col3-4: mean PSTH, mean whisk, peak-FR box, AUC box (spanned)
% ====================================================================
figure('Position',[100 100 1260 980]);
tl = tiledlayout(5, 4, 'TileSpacing','compact','Padding','compact');
target = min(4, nCells);   % Python used cell index 3 (0-based) -> 4th cell
xwin = [-0.1 0.5];

% polar — tile 1
ax_pol = polaraxes(tl); ax_pol.Layout.Tile = 1;
polarplot(ax_pol, deg2rad(DIRECTIONS), HDrate{target}); hold(ax_pol,'on');
polarscatter(ax_pol, deg2rad(speaker_position), ones(1,4), 40, 'filled');
thetaticks(ax_pol, sort(speaker_position));

% PSTHs col1 rows2-5 -> tiles 5,9,13,17
psth_mask = bins_plot/1000 >= xwin(1) & bins_plot/1000 <= xwin(2);
psth_max = max(new_rate(target, psth_mask, :), [], 'all');
psth_ylim = [0, max(psth_max*1.15, 1)];
psth_tiles = [5 9 13 17];
for k = 1:4
    ax = nexttile(tl, psth_tiles(k));
    bar(ax, bins_plot/1000, reshape(new_rate(target,:,k),1,[]), 1, 'EdgeColor','none');
    xlim(ax, xwin); ylim(ax, psth_ylim);
    ylabel(ax, sprintf('%d°', speaker_position(k))); despine(ax);
end

% whisk col2 rows2-5 -> tiles 6,10,14,18
wmask = trigger_time >= xwin(1) & trigger_time <= xwin(2);
wiv = squeeze(baseline_subtract_whisk(target, wmask, :));
w_pad = max((max(wiv(:))-min(wiv(:)))*0.1, 0.02);
whisk_ylim = [min(wiv(:))-w_pad, max(wiv(:))+w_pad];
whisk_tiles = [6 10 14 18];
for k = 1:4
    ax = nexttile(tl, whisk_tiles(k));
    plot(ax, trigger_time, squeeze(baseline_subtract_whisk(target,:,k)));
    xlim(ax, xwin); ylim(ax, whisk_ylim); despine(ax);
end

% mean PSTH — tiles 3-4 (row1)
ax = nexttile(tl, 3); ax.Layout.TileSpan = [1 2]; hold(ax,'on');
for k = 1:4
    y = mean_psth(plot_resp,k)'; sem = sd_psth(plot_resp,k)';
    fillBand(ax, bins_plot(plot_resp)/1000, y, sem);
    plot(ax, bins_plot(plot_resp)/1000, y, 'LineWidth',1.1);
end
xlim(ax, xwin); xlabel(ax,'Time (s)'); ylabel(ax,'Firing Rate [Hz]'); despine(ax);

% mean whisk — tiles 7-8 (row2)
ax = nexttile(tl, 7); ax.Layout.TileSpan = [1 2]; hold(ax,'on');
for k = 1:4
    y = mean_whisk(plot_resp_whisk,k)'; sem = sd_whisk(plot_resp_whisk,k)';
    fillBand(ax, trigger_time(plot_resp_whisk), y, sem);
    plot(ax, trigger_time(plot_resp_whisk), y, 'LineWidth',1.1);
end
xlim(ax, xwin); ylim(ax, whisk_ylim);
xlabel(ax,'Time (s)'); ylabel(ax,'Whisker Pad motion'); despine(ax);

% peak-FR box — tiles 11-12 (row3)
ax = nexttile(tl, 11); ax.Layout.TileSpan = [1 2];
boxplot(ax, resp_peak_sorted, {'Closest','2nd Closest','2nd Farthest','Farthest'}, 'Symbol','');
padYLim(ax, resp_peak_sorted(:), ybox_frac); xtickangle(ax,20);
ylabel(ax,'Peak FR [Hz]'); xlabel(ax,'Speaker distance to PD');

% AUC box — tiles 15-16 (row4)
ax = nexttile(tl, 15); ax.Layout.TileSpan = [1 2];
boxplot(ax, auc_resp_sorted, {'Closest','2nd Closest','2nd Farthest','Farthest'}, 'Symbol','');
padYLim(ax, auc_resp_sorted(:), ybox_frac); xtickangle(ax,20);
ylabel(ax,'AUC [spikes]'); xlabel(ax,'Speaker distance to PD');

end % main function

% ====================================================================
% LOCAL FUNCTIONS
% ====================================================================
function requireVar(vn, names)
for i = 1:numel(names)
    if ~ismember(names{i}, vn)
        fprintf('Available variable names:\n');
        fprintf('  %s\n', vn{:});
        error('Required table variable "%s" not found (see list above; fix CONFIG).', names{i});
    end
end
end

function out = getScalar(tbl, ridx, varname)
col = tbl.(varname);
if iscell(col); v = col{ridx}; else; v = col(ridx,:); end
out = double(v(1));
end

function out = getCellOrArray(tbl, ridx, varname)
col = tbl.(varname);
if iscell(col); out = col{ridx}; else; out = col(ridx,:); end
end

function out = getSpeakerData(tbl, ridx, base, speakers, vn)
% Resolve per-speaker data to a 1x4 cell ordered as `speakers`.
% Supports: N x K cell column (col k = speaker), struct(.a/.w/.e/.r),
% a 1xK cell, or separate <base>_<letter> columns.
nS = numel(speakers);
out = cell(1, nS);
if ismember(base, vn)
    colvar = tbl.(base);
    % primary form: N x K cell, access {ridx, k}
    if iscell(colvar) && size(colvar,2) >= nS
        for k = 1:nS; out{k} = colvar{ridx, k}; end
        return
    end
    v = tbl.(base)(ridx,:);
    if iscell(v) && isscalar(v); v = v{1}; end
    if isstruct(v)
        for k = 1:nS; out{k} = v.(speakers{k}); end
        return
    elseif iscell(v) && numel(v) >= nS
        for k = 1:nS; out{k} = v{k}; end
        return
    end
end
% fallback: separate columns base_a, base_w, ...
sep_ok = true;
for k = 1:numel(speakers)
    col = sprintf('%s_%s', base, speakers{k});
    if ~ismember(col, vn); sep_ok = false; break; end
end
if sep_ok
    for k = 1:numel(speakers)
        col = sprintf('%s_%s', base, speakers{k});
        x = tbl.(col)(ridx);
        if iscell(x); x = x{1}; end
        out{k} = x;
    end
    return
end
fprintf('Available variable names:\n'); fprintf('  %s\n', vn{:});
error(['Could not resolve per-speaker variable "%s". Expected an N x 4 cell, ' ...
    'a struct with fields a/w/e/r, a 1xK cell, or columns %s_a..%s_r. Fix CONFIG.'], base, base, base);
end

function v = rowvec(x)
v = double(x(:)');
end

function out = takeAlong(M, idx)
% gather M(i, idx(i,:)) row-wise.  M, idx : nRows x nCols
out = zeros(size(M));
for i = 1:size(M,1)
    out(i,:) = M(i, idx(i,:));
end
end

function out = takeAlong3(A, idx)
% reorder 3rd dim of A (nRows x nT x nCols) per-row by idx (nRows x nCols)
out = zeros(size(A));
for i = 1:size(A,1)
    out(i,:,:) = A(i,:,idx(i,:));
end
end

function padYLim(ax, data, frac)
data = data(isfinite(data));
if isempty(data); return; end
lo = min(data); hi = max(data); span = hi - lo;
if span == 0; span = max(abs(hi),1); end
ylim(ax, [lo - frac*span, hi + frac*span]);
end

function plotPairedLines(ax, M)
% faint lines connecting each cell's values across columns
x = 1:size(M,2);
for i = 1:size(M,1)
    plot(ax, x, M(i,:), '-', 'Color',[0.4 0.4 0.4 0.4], 'LineWidth',0.6);
end
end

function fillBand(ax, x, y, sem)
xb = [x, fliplr(x)];
yb = [y - sem, fliplr(y + sem)];
fill(ax, xb, yb, [0.2 0.2 0.8], 'FaceAlpha',0.12, 'EdgeColor','none', ...
    'HandleVisibility','off');
end

function despine(ax)
ax.Box = 'off';
end

function printPosthoc(stats, labels)
% Nemenyi-style pairwise comparison via multcompare on Friedman ranks.
try
    c = multcompare(stats, 'CType','tukey-kramer', 'Display','off');
catch
    fprintf('  (multcompare unavailable — skipping post-hoc)\n');
    return
end
fprintf('Pairwise (multcompare, Tukey-Kramer on ranks):\n');
for r = 1:size(c,1)
    fprintf('  %s vs %s: p = %.3f\n', labels{c(r,1)}, labels{c(r,2)}, c(r,6));
end
fprintf('%s\n', repmat('-',1,70));
end

function cmap = viridisish()
% small viridis-like colormap (avoids toolbox dependency)
cmap = [ ...
 0.267 0.005 0.329; 0.283 0.141 0.458; 0.254 0.265 0.530; ...
 0.207 0.372 0.553; 0.164 0.471 0.558; 0.128 0.567 0.551; ...
 0.135 0.659 0.518; 0.267 0.749 0.441; 0.478 0.821 0.318; ...
 0.741 0.873 0.150; 0.993 0.906 0.144];
end
