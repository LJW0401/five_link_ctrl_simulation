% 计算不同腿长下的 LQR 增益矩阵 K(2x6)，对其 12 个分量分别做 3 阶多项式拟合，
% 得到每个分量随腿长 L0 变化的多项式系数；并叠加散点（逐点 LQR）+ 拟合曲线出图。
% 依赖：get_k_length.m（单个腿长的 LQR 求解）
tic

%% ===== 1. 逐腿长求 LQR 增益，采样到矩阵 =====
leg = 0.1:0.01:0.4;          % 腿长采样点 L0
n_leg = numel(leg);
K_samp = zeros(n_leg, 12);   % 每行 = 该腿长下 reshape 成 1x12 的增益（按 K' 行优先展开）

for idx = 1:n_leg
    k = get_k_length(leg(idx));          % 2x6
    K_samp(idx, :) = reshape(k.', 1, 12); % 行优先：[K11..K16 K21..K26]
    fprintf('leg_length=%.3f\n', leg(idx));
end

%% ===== 2. 对 12 个分量分别做 3 阶多项式拟合 =====
poly_order = 3;
K_coef = zeros(12, poly_order + 1);      % 每行 = 一个分量的多项式系数
for c = 1:12
    K_coef(c, :) = polyfit(leg, K_samp(:, c).', poly_order);
end

toc

%% ===== 3. 绘图：散点（逐点 LQR 采样）+ 拟合曲线（多项式）=====
% 散点用原始采样腿长 leg；拟合曲线用加密网格 xf 使曲线平滑
xf = linspace(min(leg), max(leg), 400);
% 子图标题：第 1 行 K1*（驱动轮力矩增益），第 2 行 K2*（髋关节力矩增益）
K_name = {'K_{11}','K_{12}','K_{13}','K_{14}','K_{15}','K_{16}', ...
          'K_{21}','K_{22}','K_{23}','K_{24}','K_{25}','K_{26}'};

figure('Name','LQR 增益表拟合','Color','w','Position',[100 100 1400 600]);
for c = 1:12
    subplot(2, 6, c); hold on; grid on;

    y_samp = K_samp(:, c).';            % 采样散点值
    y_fit  = polyval(K_coef(c, :), xf); % 拟合曲线
    y_hat  = polyval(K_coef(c, :), leg);% 采样腿长处的拟合值，用于算 R^2

    scatter(leg, y_samp, 18, 'b', 'filled');   % 散点：逐点 LQR 结果
    plot(xf, y_fit, 'r-', 'LineWidth', 1.4);   % 实线：多项式拟合

    % 拟合优度 R^2
    ss_res = sum((y_samp - y_hat).^2);
    ss_tot = sum((y_samp - mean(y_samp)).^2);
    R2 = 1 - ss_res / ss_tot;

    title(sprintf('%s  R^2=%.4f', K_name{c}, R2));
    xlabel('L_0 (m)'); ylabel(K_name{c});
    if c == 1
        legend({'LQR 采样点','多项式拟合'}, 'Location','best');
    end
    set(gca, 'GridLineStyle', ':', 'GridColor', 'k', 'GridAlpha', 0.4);
    hold off;
end
sgtitle(sprintf('LQR 增益表 K_{ij}(L_0)：散点 + %d 阶多项式拟合', poly_order));

saveas(gcf, 'K_table_fit.png');
fprintf('拟合图已保存到 K_table_fit.png\n');

%% ===== 4. 打印各分量多项式系数（C 数组形式）=====
for c = 1:12
    fprintf('fp32 a%d%d[%d] = {%.4f,%.4f,%.4f,%.4f};\n', ...
        floor((c - 1) / 6) + 1, mod(c - 1, 6) + 1, poly_order + 1, ...
        K_coef(c, 1), K_coef(c, 2), K_coef(c, 3), K_coef(c, 4));
end
toc
