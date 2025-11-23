% set figure
function fun_setfig(titlestr, xalbelstr,ylabelstr)
    if 3 == nargin
        title(titlestr);xlabel(xalbelstr);ylabel(ylabelstr);
    end
    if 2 == nargin
        xlabel(xalbelstr);ylabel(ylabelstr);
    end
    if 1 == nargin
        xlabel(xalbelstr);
    end
    grid on

    %调整 XLABLE和YLABLE不会被切掉
    % set(gcf,'Position',[100 100 260 220]);
    % set(gca,'Position',[.13 .17 .80 .74]);

    % font size
    figure_FontSize=12;
    %     set(get(gca,'XLabel'),'FontSize',figure_FontSize,'Vertical','top');
    %     set(get(gca,'YLabel'),'FontSize',figure_FontSize,'Vertical','middle');
    %     set(findobj('FontSize',12),'FontSize',figure_FontSize);
    set(gca,'FontSize',figure_FontSize);

    % line width
    set(findobj(get(gca,'Children'),'LineWidth',0.5),'LineWidth',2,'Markersize', 10);

end