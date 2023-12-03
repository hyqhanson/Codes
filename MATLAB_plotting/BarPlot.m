
function BarPlot(values1, categories,titles,valuename)
    % Determine the number of categories
    numCategories = length(categories);
    
    barWidth = 0.4;

    % Calculate the x-coordinates for the bars
    x1 = 1:numCategories;
    
    % Create a side-by-side bar plot
    figure;
    bar(x1, values1, barWidth, 'b', 'DisplayName', valuename);
    
    % Add labels and title
    xlabel('Posits');
    ylabel('Values');
    title(titles);
    
    % Add category labels to the x-axis
    xticks(x1 + barWidth / 2);
    xticklabels(categories);
    
    % Add a legend
    legend('show');
    
    % Display the grid
    grid on;
    hold off;
end