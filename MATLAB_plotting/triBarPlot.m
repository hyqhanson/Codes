function triBarPlot(values1, values2, values3, categories, titles, valuename1, valuename2, valuename3, color1, color2, color3)
    % Determine the number of categories
    numCategories = length(categories);
    
    % Set the bar width
    barWidth = 0.2;  % Adjust as needed
    
    % Calculate the x-coordinates for the bars
    x1 = 1:numCategories;
    x2 = x1 + barWidth;
    x3 = x1 + 2 * barWidth;  % Adjusted for the third bar
    
    % Create a side-by-side bar plot
    figure;
    bar(x1, values1, barWidth, 'FaceColor', color1, 'DisplayName', valuename1);
    hold on;
    bar(x2, values2, barWidth, 'FaceColor', color2, 'DisplayName', valuename2);
    hold on;
    bar(x3, values3, barWidth, 'FaceColor', color3, 'DisplayName', valuename3);
    
    % Add labels and title
    xlabel('Posits');
    ylabel('Values');
    title(titles);
    
    % Add category labels to the x-axis
    xticks(x1 + barWidth);
    xticklabels(categories);
    
    % Add a legend
    legend('show', 'Location', 'southwestoutside');
    
    % Display the grid
    grid on;
    hold off;
end
