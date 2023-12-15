
function dualBarPlot(values1, values2, categories,titles,valuename1,valuename2,color1,color2)
    % Determine the number of categories
    numCategories = length(categories);
    
    % Set the bar width
    barWidth = 0.4;  % Adjust as needed
    
    % Calculate the x-coordinates for the bars
    x1 = 1:numCategories;
    x2 = x1 + barWidth;
    
    % Create a side-by-side bar plot
    figure;
    bar(x1, values1, barWidth, color1, 'DisplayName', valuename1);
    hold on;
    bar(x2, values2, barWidth, color2, 'DisplayName', valuename2);
    
    % Add labels and title
    xlabel('Posits');
    ylabel('Values');
    title(titles);
    
    % Add category labels to the x-axis
    xticks(x1 + barWidth / 2);
    xticklabels(categories);
    
    % Add a legend
    legend('show','Location', 'southwestoutside');
    
    % Display the grid
    grid on;
    hold off;
end
