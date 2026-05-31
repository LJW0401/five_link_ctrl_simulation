% 定义一个函数来显示多项式方程
function display_polynomial(coefficients, name)
    equation = sprintf('%s = ', name);
    n = length(coefficients);
    for i = 1:n
        if coefficients(i) ~= 0
            if i == n
                term = sprintf('%.4f', coefficients(i));
            else
                term = sprintf('%.4f*t%d', coefficients(i), n-i);
            end
            if i > 1 && coefficients(i) > 0
                term = [' + ', term];
            end
            equation = [equation, term];
        end
    end
    fprintf('%s;\n', equation);
end