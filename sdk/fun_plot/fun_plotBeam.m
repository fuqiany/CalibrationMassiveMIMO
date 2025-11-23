function fun_plotBeam(h,d,lambda,startdegree,enddegree,stepnum)
%Input:
%     h---channel vector : Nr*1
%     d---antenna interval
%     lambda---wavelength
%     startpi/endpi---xlabel start/end degree(бу), e.g. 0, 180
%     stepnum---stepnum, e.g. 1024
Nrh = length(h);
fsteer = @(theata)exp(-1j*2*pi*cos(theata)*d*(0:1:Nrh-1)'/lambda)/sqrt(Nrh);
startdegree = startdegree*pi/180;
enddegree = enddegree*pi/180;
theatam = startdegree:pi/stepnum:enddegree;
for n = 1:length(theatam)
    thetan = theatam(n);
    beampower(n) = abs(h'*fsteer(thetan));
end





beampower = beampower./max(beampower);
beampowerdB = 20*log10(beampower);

plot(theatam*180/pi, beampowerdB); hold on
xlabel('angle(бу)');ylabel('power in dB');grid on;
% plot(theatam, beampower); hold on
% xlabel('radius');ylabel('power in dB');grid on;
end