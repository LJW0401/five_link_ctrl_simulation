% 计算不同腿长下的 LQR 增益矩阵 K(2x6)，导出 lqr_config.json。
% 通过顶部 FIT_MODE 在两种导出格式间切换（均与仿真侧 calc_lqr_k.py 兼容）：
%   'poly'   —— 对 12 个分量分别做 3 阶多项式拟合，导出 poly_order + K_poly_coef
%   'linear' —— 直接导出逐点采样 L0_values + K_table，由仿真侧做线性插值
% 散点（逐点 LQR）始终绘制；拟合曲线按模式叠加多项式曲线或分段折线。
% 依赖：get_k_length.m（单个腿长的 LQR 求解）
tic

%% ===== 0. 导出模式开关 =====
FIT_MODE = 'poly';        % 'poly' = 3 阶多项式拟合；'linear' = 线性插值查表
poly_order = 3;          % 多项式阶数（仅 poly 模式使用）

%% ===== 1. 逐腿长求 LQR 增益，采样到矩阵 =====
% 腿长采样区间与 calc_lqr_k.py 的 DEFAULT_L0_RANGE 一致：[0.10, 0.40]，30 点
leg = linspace(0.10, 0.40, 30);   % 腿长采样点 L0
n_leg = numel(leg);
K_samp = zeros(n_leg, 12);   % 每行 = 该腿长下 reshape 成 1x12 的增益（按 K' 行优先展开）
K_cells = cell(1, n_leg);    % 每个元素 = 该腿长下的 2x6 增益矩阵（linear 模式导出用）

for idx = 1:n_leg
    [k, params] = get_k_length(leg(idx)); % k:2x6；params:导出用参数（与腿长无关）
    K_samp(idx, :) = reshape(k.', 1, 12); % 行优先：[K11..K16 K21..K26]
    K_cells{idx}   = k;
    fprintf('leg_length=%.3f\n', leg(idx));
end

%% ===== 2. 对 12 个分量分别做 3 阶多项式拟合（poly 模式 + 绘图用）=====
K_coef = zeros(12, poly_order + 1);      % 每行 = 一个分量的多项式系数
for c = 1:12
    K_coef(c, :) = polyfit(leg, K_samp(:, c).', poly_order);
end

toc

%% ===== 3. 绘图：散点（逐点 LQR 采样）+ 拟合曲线 =====
% 散点用原始采样腿长 leg；poly 模式叠加多项式拟合曲线，linear 模式叠加分段折线
xf = linspace(min(leg), max(leg), 400);
K_name = {'K_{11}','K_{12}','K_{13}','K_{14}','K_{15}','K_{16}', ...
          'K_{21}','K_{22}','K_{23}','K_{24}','K_{25}','K_{26}'};

figure('Name','LQR 增益表拟合','Color','w','Position',[100 100 1400 600]);
for c = 1:12
    subplot(2, 6, c); hold on; grid on;

    y_samp = K_samp(:, c).';            % 采样散点值
    scatter(leg, y_samp, 18, 'b', 'filled');   % 散点：逐点 LQR 结果

    if strcmp(FIT_MODE, 'poly')
        y_fit = polyval(K_coef(c, :), xf);     % 多项式拟合曲线
        y_hat = polyval(K_coef(c, :), leg);    % 采样腿长处拟合值，用于算 R^2
        plot(xf, y_fit, 'r-', 'LineWidth', 1.4);

        ss_res = sum((y_samp - y_hat).^2);
        ss_tot = sum((y_samp - mean(y_samp)).^2);
        R2 = 1 - ss_res / ss_tot;
        title(sprintf('%s  R^2=%.4f', K_name{c}, R2));
        fit_legend = '多项式拟合';
    else
        plot(leg, y_samp, 'r-', 'LineWidth', 1.2);  % 分段折线 = 线性插值
        title(K_name{c});
        fit_legend = '线性插值';
    end

    xlabel('L_0 (m)'); ylabel(K_name{c});
    if c == 1
        legend({'LQR 采样点', fit_legend}, 'Location','best');
    end
    set(gca, 'GridLineStyle', ':', 'GridColor', 'k', 'GridAlpha', 0.4);
    hold off;
end
if strcmp(FIT_MODE, 'poly')
    sgtitle(sprintf('LQR 增益表 K_{ij}(L_0)：散点 + %d 阶多项式拟合', poly_order));
else
    sgtitle('LQR 增益表 K_{ij}(L_0)：散点 + 线性插值');
end

saveas(gcf, 'K_table_fit.png');
fprintf('拟合图已保存到 K_table_fit.png\n');

%% ===== 4. 导出 lqr_config.json（与仿真侧 calc_lqr_k.py 同格式）=====
% robot_params / Q / R 直接取自 get_k_length.m（params），不在此重复硬编码；
% leg_params 是 5-bar 连杆几何，get_k_length 的模型未涉及，仍按仿真侧取值给出
config.robot_params = params.robot_params;
config.leg_params   = struct('l1', 0.21, 'l2', 0.24, 'l3', 0.24, ...
                             'l4', 0.21, 'l5', 0.0);
config.Q            = params.Q;
config.R            = params.R;
config.L0_range     = struct('min', min(leg), 'max', max(leg), 'n_points', n_leg);

if strcmp(FIT_MODE, 'poly')
    % 多项式格式：K_poly_coef[i][j] = 第 c=(i-1)*6+j 个分量的系数（高次在前，与 polyval 一致）
    K_poly_coef = cell(1, 2);
    for i = 1:2
        row_cell = cell(1, 6);
        for j = 1:6
            row_cell{j} = K_coef((i - 1) * 6 + j, :);
        end
        K_poly_coef{i} = row_cell;
    end
    config.poly_order  = poly_order;
    config.K_poly_coef = K_poly_coef;

    %% 打印各分量多项式系数（C 数组形式，仅 poly 模式）
    for c = 1:12
        fprintf('fp32 a%d%d[%d] = {%.4f,%.4f,%.4f,%.4f};\n', ...
            floor((c - 1) / 6) + 1, mod(c - 1, 6) + 1, poly_order + 1, ...
            K_coef(c, 1), K_coef(c, 2), K_coef(c, 3), K_coef(c, 4));
    end
else
    % 线性插值格式：导出逐点采样腿长与对应 K 矩阵，由仿真侧线性插值
    config.L0_values = leg;          % 1 x n
    config.K_table   = K_cells;      % n 个 2x6 矩阵 -> JSON 中 [n][2][6]
end

fid = fopen('lqr_config.json', 'w');
fwrite(fid, jsonencode(config, 'PrettyPrint', true));
fclose(fid);
fprintf('lqr_config.json 已导出（模式：%s）\n', FIT_MODE);
toc
