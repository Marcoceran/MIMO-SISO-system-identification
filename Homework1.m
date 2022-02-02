clc, close all, clear all

Nx = 1500;                      % length of process x
Niter = 1;                    % number of iteration of Montecarlo simulation
SNRdB = [-20, -10, 0, 10, 20, 30];
SNRlin = 10.^(SNRdB/20);        % divide by 20 instead of 10 because we multiply the root power quantities
p_val = [(1:+2:101)];             % those are the values of filter length p that we are using
Nsnr = length(SNRdB);           % this is the number of different values of SNR that we are taking into account

%% Part a

%% H1(z)

% Niter = 1;
% w0 = randn([Nx 1]);
% w = zeros([Nx 1]);
% x = randn([Nx 1]);
% hw = 0.95.^(0:Nx-1);
% cwwinv = inv(toeplitz(hw));
% y_noiseless = zeros(Nx, 1);
% y_noiseless(1) = x(1);
% for i = 2:Nx
%     y_noiseless(i) = x(i) + 0.9*y_noiseless(i-1);
% end
% MSE = zeros(length(p_val), length(SNRlin), Niter);
% h = 0.9.^(0:2*max(p_val)-1)';
% for kk = 1:Niter
%     for i = 1:length(p_val)
%         w0 = randn([Nx 1]);
%         w(1) = w0(1);
%         for ii = 2:Nx
%             w(ii) = w0(ii) + 0.95*w(ii-1);
%         end
%         w0 = w./sqrt(mean(w.^2));   % normalize noise
%         for k = 1:length(SNRdB)
%             w = sqrt(max(conv(h, flip(h))))*w0./SNRlin(k);
%             y_noise = y_noiseless + w;
%             X = zeros(Nx, p_val(i));
%             for j = 1:p_val(i)
%                 X(j:(Nx-p_val(i)-1+j), j) = x(1:Nx-p_val(i));
%             end
%             h_est = inv(X'*cwwinv*X)*X'*cwwinv*y_noise;
%             MSE(i, k, kk) = mean(([h_est; zeros(length(h)-length(h_est), 1)] - h).^2);
%         end
%     end
%     100*kk/Niter
% end
% meanMSE = mean(MSE, 3);
% 
% 
% 
% hold on
% for i = 1:length(SNRdB)
%     semilogy(p_val, meanMSE(:, i));
%     legend('SNR = -20dB', 'SNR = -10dB', 'SNR = 0dB', 'SNR = 10dB', 'SNR = 20dB', 'SNR = 30dB')
%     xlabel('Estimated filter length p'); ylabel('Mean Square Error of h');
%     set(gca, 'yscale', 'log');
% end
% 
% 
% hold on; stem(h(1:length(h_est))); stem(h_est);
% legend('$h_1[n]$','$\hat{h}_1[n]$', 'interpreter', 'latex');
% set(gca, 'FontSize', 14)
% xlabel('n'); ylabel('h[n]');
% 
% 
% hold on; pzplot(tf([1],[1 -0.9])); pzplot(tf([h_est'],[1], -1));
% xlim([-1 1]); ylim([-1 1]);
% legend('True sys. singularities','Estimated sys. singularities');
% set(gca, 'FontSize', 8);
% 
% 
% % Power Spectral Densities computation
% XH = fftshift(fft([h; zeros(length(x)-length(h), 1)]))/Nx;
% X = fftshift(fft([1; zeros(length(x)-1, 1)]))/Nx;
% W =  fftshift(fft(hw*sqrt(max(conv(h, flip(h))))/SNRlin(k)))/Nx;
% psdXH = real(XH).^2 + imag(XH).^2;
% psdW =  real(W).^2 + imag(W).^2;
% fh = (-length(XH)/2: length(XH)/2-1) * (2*pi/length(XH));
% %plots
% hold on; plot(fh, psdXH); plot(fh, abs(X).^2);
% legend('$|H_1(z) \cdot X(z)|^2$', '$|W(z)|^2$', 'interpreter', 'latex');
% xlabel('\omega'); ylabel('Power Spectral Density', 'interpreter', 'latex'); xlim([-pi, pi]);
% set(gca, 'FontSize', 14, 'yScale', 'log')
% 
% 
% 
% % Wiener Filter implementation
% xpower = mean(x.^2);
% wpower = mean(w.^2);
% H = fftshift(fft([h; zeros(length(y_noiseless)-length(h), 1)]))/length(y_noiseless);
% Hconj = conj(H);
% Hw = fftshift(fft(hw(1:length(H))))'/length(H);
% psdH = abs(H).^2;
% psdHw = abs(Hw).^2;
% H_wiener = (xpower.*Hconj)./(xpower*psdH + wpower*psdHw);
% % plots
% hold on; plot(fh, psdXH); plot(fh, abs(H_wiener).^2, 'linewidth', 1); plot(fh, psdW);
% legend('$|H_1(z) \cdot X(z)|^2$', 'Wiener filter PSD', 'Noise PSD', 'interpreter', 'latex');
% xlabel('\omega'); ylabel('Power Spectral Density', 'interpreter', 'latex'); xlim([-pi, pi]);
% set(gca, 'FontSize', 11, 'yScale', 'log')



%% H2(z)


% p_val = [(1:24), (25:+2:61), (66:+5:101)];
% hw = 0.95.^(0:Nx-1);
% cwwinv = inv(toeplitz(hw));
% w0 = randn([Nx 1]);
% w = zeros([Nx 1]);
% x = randn([Nx 1]);
% y_noiseless = zeros(Nx, 1);
% h_est = zeros(2*max(p_val), 1);
% h = h_est;
% MSE = zeros(length(p_val), length(SNRlin), Niter);
% h_est(3) = 1;
% for ii = 3:length(h)
%     h(ii) = h_est(ii) + 0.8*sqrt(2)*h(ii-1) - 0.64*h(ii-2);
% end
% h(1:end-2) = h(3:end);
% for kk = 1:Niter
%     for i = 23:23
%         for k = 1:length(SNRdB)
%             w0 = randn([Nx 1]);
%             w(1) = w0(1);
%             for ii = 2:Nx
%                 w(ii) = w0(ii) + 0.95*w(ii-1);
%             end
%             w0 = w./sqrt(mean(w.^2));   % normalize noise
%             x = randn([Nx 1]);
%             y_noiseless = zeros(Nx, 1);
%             y_noiseless(1:2) = x(1:2);
%             for ii = 3:Nx
%                 y_noiseless(ii) = x(ii) + 0.8*sqrt(2)*y_noiseless(ii-1) - 0.64*y_noiseless(ii-2);
%             end
%             w = sqrt(max(conv(h, flip(h))))*w0./SNRlin(k);
%             y = y_noiseless + w;
%             X = zeros(Nx, p_val(i));
%             for j = 1:p_val(i)
%                 X(j:(Nx-p_val(i)-1+j), j) = x(1:Nx-p_val(i));
%             end
%             h_est = pinv(X'*cwwinv*X)*X'*cwwinv*y;
%             MSE(i, k, kk) = mean((h-[h_est(1:length(h_est)); zeros(length(h)-length(h_est), 1)]).^2);
%         end
%     end
%     100*kk/Niter
% end
% meanMSE = mean(MSE, 3);
% 
% 
% hold on
% for i = 1:length(SNRdB)
%     semilogy(p_val, meanMSE(:, i));
%     legend('SNR = -20dB', 'SNR = -10dB', 'SNR = 0dB', 'SNR = 10dB', 'SNR = 20dB', 'SNR = 30dB')
%     xlabel('Estimated filter length p'); ylabel('Mean Square Error');
%     set(gca, 'yscale', 'log');
% end
% 
% 
% 
% hold on; stem(h(1:length(h_est))); stem(h_est);
% legend('$h_2[n]$','$\hat{h}_2[n]$', 'interpreter', 'latex');
% set(gca, 'FontSize', 14)
% xlabel('n'); ylabel('h[n]');
% 
% 
% hold on; pzplot(tf([1],[1 -0.8*sqrt(2) 0.64])); % now we are plotting the poles
% pzplot(tf([h_est'],[1], -1));        % and zeroes of the true and of
% xlim([-1.1 1.1]); ylim([-1.1 1.1]);             % the estimated systems
% legend('True system singularities', 'Estimated system singularities');
% set(gca, 'FontSize', 9);
% 
% 
% XH = fftshift(fft([h ; zeros(length(x)-length(h), 1)]))/length(y_noiseless);
% W =  fftshift(fft(hw*sqrt(max(conv(h, flip(h))))/SNRlin(k)))/length(w);
% psdXH = (real(XH).^2 + imag(XH).^2);
% psdW =  (real(W).^2 + imag(W).^2);
% fh = (-length(XH)/2: length(XH)/2-1) * (2*pi/length(XH));
% % plots
% hold on; plot(fh, psdXH); plot(fh, psdW);
% legend('$|H_2(z) \cdot X(z)|^2$', '$|W(z)|^2$', 'interpreter', 'latex');
% xlabel('\omega'); ylabel('Power Spectral Density', 'interpreter', 'latex'); xlim([-pi, pi]);
% set(gca, 'FontSize', 14, 'yScale', 'log')
% 
% 
% 
% % Wiener Filter implementation
% xpower = mean(x.^2);
% wpower = mean(w.^2);
% H = fftshift(fft([h; zeros(length(y_noiseless)-length(h), 1)]))/length(y_noiseless);
% Hconj = conj(H);
% Hw = fftshift(fft(hw(1:length(H))))'/length(H);
% psdH = abs(H).^2;
% psdHw = abs(Hw).^2;
% H_wiener = (xpower.*Hconj)./(xpower*psdH + wpower*psdHw);
% % plots
% hold on; plot(fh, psdXH); plot(fh, abs(H_wiener).^2, 'linewidth', 1); plot(fh, psdW);
% legend('$|H_2(z) \cdot X(z)|^2$', 'Wiener filter PSD', 'Noise PSD', 'interpreter', 'latex');
% xlabel('\omega'); ylabel('Power Spectral Density', 'interpreter', 'latex'); xlim([-pi, pi]);
% set(gca, 'FontSize', 11, 'yScale', 'log')



%% H3(z)

% Niter = 1;
% p_val = [(1:24), (25:+2:81)];
% hw = 0.95.^(0:Nx-1);
% cwwinv = inv(toeplitz(hw));
% w0 = randn([Nx 1]);
% x = randn([Nx 1]);
% y_noiseless = zeros(Nx, 1);
% h_est = zeros(2*max(p_val), 1);
% MSE = zeros(length(p_val), length(SNRlin), Niter);
% h = h_est;
% h_est(length(h)-2) = 1;
% for ii = length(h)-2:-1:1
%     h(ii) = h_est(ii) + 0.8*sqrt(2)*h(ii+1) - 0.64*h(ii+2);
% end
% h(3:end) = h(1:end-2);
% for kk = 1:Niter
%     for i = 23:23
%         for k = 1:length(SNRdB)
%             w0 = randn([Nx 1]);
%             w(1) = w0(1);
%             for ii = 2:Nx
%                 w(ii) = w0(ii) + 0.95*w(ii-1);
%             end
%             w0 = w./sqrt(mean(w.^2));   % normalize noise
%             x = randn([Nx 1]);
%             y_noiseless = zeros(Nx, 1);
%             y_noiseless(end-1:end) = x(end-1:end);
%             for ii = Nx-2:-1:1
%                 y_noiseless(ii) = x(ii) + 0.8*sqrt(2)*y_noiseless(ii+1) - 0.64*y_noiseless(ii+2);
%             end
%             w = sqrt(max(conv(h, flip(h))))*w0./SNRlin(k);
%             y = y_noiseless + w';
%             X = zeros(Nx, p_val(i));
%             for j = p_val(i):-1:1
%                 X(j:(Nx-p_val(i)+j), p_val(i)+1-j) = x(p_val(i):Nx);
%             end
%             h_est = flip(pinv(X'*cwwinv*X)*X'*cwwinv*y);
%             MSE(i, k, kk) = mean((h-[zeros(length(h)-length(h_est), 1); h_est]).^2);
%         end
%     end
% 100*kk/Niter
% end
% meanMSE = mean(MSE, 3);
% 
% 
% hold on
% for i = 1:length(SNRdB)
%     semilogy(p_val, meanMSE(:, i));
%     legend('SNR = -20dB', 'SNR = -10dB', 'SNR = 0dB', 'SNR = 10dB', 'SNR = 20dB', 'SNR = 30dB')
%     xlabel('Estimated filter length p'); ylabel('Mean Square Error');
%     set(gca, 'yscale', 'log');
% end
% 
% 
% hold on; stem(-length(h_est)+1:0, h(end+1 - length(h_est):end)); stem(-length(h_est)+1:0, h_est);
% legend('$h_3[n]$','$\hat{h}_3[n]$', 'interpreter', 'latex');
% set(gca, 'FontSize', 14); xlabel('n'); ylabel('h[n]');
% 
% 
% hold on; pzplot(tf([1],[0.64 -0.8*sqrt(2) 1]));  % now we are plotting the poles
% pzplot(tf([h_est'],[1], -1));                       % and zeroes of the true and of
% xlim([-1.5 1.5]); ylim([-1.5 1.6]);             % the estimated systems
% legend('True system singularities', 'Estimated system singularities');
% set(gca, 'FontSize', 9);
% 
% 
% %psd
% XH = fftshift(fft([h; zeros(length(x)-length(h), 1)]))/length(y_noiseless);
% W =  fftshift(fft(hw*sqrt(max(conv(h, flip(h))))/SNRlin(k)))/length(w);
% psdXH = (real(XH).^2 + imag(XH).^2);
% psdW =  (real(W).^2 + imag(W).^2);
% fh = (-length(XH)/2: length(XH)/2-1)*(2*pi/length(XH));
% %plots
% hold on; plot(fh, psdXH); plot(fh, psdW);
% legend('$|H_3(z) \cdot X(z)|^2$', '$|W(z)|^2$', 'interpreter', 'latex');
% xlabel('\omega'); ylabel('Power Spectral Density', 'interpreter', 'latex'); xlim([-pi, pi]);
% set(gca, 'FontSize', 14, 'yScale', 'log')
% 
% 
% 
% 
% % Wiener Filter implementation
% xpower = mean(x.^2);
% wpower = mean(w.^2);
% H = fftshift(fft(flip([zeros(length(y_noiseless)-length(h), 1); h])))/length(y_noiseless);
% Hconj = conj(H);
% Hw = fftshift(fft(hw(1:length(H))))'/length(H);
% psdH = abs(H).^2;
% psdHw = abs(Hw).^2;
% H_wiener = (xpower.*Hconj)./(xpower*psdH + wpower*psdHw);
% % plots
% hold on; plot(fh, psdXH); plot(fh, abs(H_wiener).^2, 'linewidth', 1); plot(fh, psdW);
% legend('$|H_3(z) \cdot X(z)|^2$', 'Wiener filter PSD', 'Noise PSD', 'interpreter', 'latex');
% xlabel('\omega'); ylabel('Power Spectral Density', 'interpreter', 'latex'); xlim([-pi, pi]);
% set(gca, 'FontSize', 11, 'yScale', 'log')



%% H4(z)
 
Niter = 1;
p_val = (1:+2:151);
Nx = 1500;
MSE = zeros(length(p_val), length(SNRlin), Niter);

hw = 0.95.^(0:Nx+max(p_val)-1);
cwwinv = inv(toeplitz(hw));
w0 = randn([Nx 1]);
x = randn([Nx 1]);
y_noiseless = zeros(Nx, 1);

h_est = zeros(2*max(p_val+1), 1);
h = h_est;
h_est(length(h_est)/2-2) = 1;
for ii = length(h)/2-1:-1:1
    h(ii) = h_est(ii) + 0.8*sqrt(2)*h(ii+1) - 0.64*h(ii+2);
end
h(3:length(h)/2) = h(1:length(h)/2-2);
h_est = h;
for ii = 3:length(h)
    h(ii) = h_est(ii) + 0.8*sqrt(2)*h(ii-1) - 0.64*h(ii-2);
end

for kk = 1:Niter
    w0 = randn([Nx+max(p_val) 1]);     % w0 is the unitary power noise
    for i = 14:14
        w0 = randn([Nx+max(p_val) 1]);
        w(1) = w0(1);
        for ii = 2:Nx+max(p_val)
            w(ii) = w0(ii) + 0.95*w(ii-1);
        end
        w0 = w./sqrt(mean(w.^2));   % normalize noise power to 1
        for k = 1
            x = randn([Nx+max(p_val) 1]);
            y_noiseless = zeros(length(x), 1);
            dumm_y = y_noiseless;
            dumm_y(end-1:end) = x(end-1:end);
            for ii = 3:length(x)
                dumm_y(ii) = x(ii) + 0.8*sqrt(2)*dumm_y(ii-1) - 0.64*dumm_y(ii-2);
            end
            y_noiseless(1:2) = dumm_y(1:2);
            for ii = length(x)-2:-1:1
                y_noiseless(ii) = dumm_y(ii) + 0.8*sqrt(2)*(y_noiseless(ii+1)) - 0.64*y_noiseless(ii+2);
            end
            w = sqrt(max(conv(h, flip(h))))*w0./SNRlin(k);
            y = y_noiseless + w';
            y = y(1:end-(p_val(i)-1)/2);
            x = x(1+(p_val(i)-1)/2:end);
            X = zeros(length(x), p_val(i));
            for j = 1:p_val(i)
                X(j:(length(x)-p_val(i)-1+j), j) = x(1:length(x)-p_val(i));
            end
            h_est = inv(X'*cwwinv(1:length(x), 1:length(x))*X)*X'*cwwinv(1:length(x), 1:length(x))*y(1:length(x));
            MSE(i, k, kk) = mean((h - [zeros((length(h)-1-length(h_est))/2, 1); h_est; zeros((length(h)+1-length(h_est))/2, 1)]).^2);
        end
    end
100*kk/Niter
end
meanMSE = mean(MSE, 3);


hold on
for i = 1:length(SNRdB)
    semilogy(p_val, meanMSE(:, i));
    legend('SNR = -20dB', 'SNR = -10dB', 'SNR = 0dB', 'SNR = 10dB', 'SNR = 20dB', 'SNR = 30dB')
    xlabel('Estimated filter length p'); ylabel('Mean Square Error');
    set(gca, 'yscale', 'log');
end


hold on; stem(-(length(h_est)-1)/2:(length(h_est)-1)/2, h_est);
stem(-(length(h_est)-1)/2:(length(h_est)-1)/2, h(length(h)/2-(length(h_est)-1)/2:length(h)/2+(length(h_est)-1)/2));
xlim([-length(h_est)/2+1, length(h_est)/2]);
legend('$h_4[n]$','$\hat{h}_4[n]$', 'interpreter', 'latex');
set(gca, 'FontSize', 14); xlabel('n'); ylabel('h[n]');


hold on; pzplot(tf([1],[0.64 -0.8*sqrt(2) 1])*tf([1],[1 -0.8*sqrt(2) 0.64]));  % now we are plotting the poles
pzplot(tf([h_est'],[1]));                       % and zeroes of the true and of
xlim([-1.5 1.5]); ylim([-1.5 1.5]);             % the estimated systems
legend('True system singularities', 'Estimated system singularities');
set(gca, 'FontSize', 9);


XH = fftshift(fft([h; zeros(length(x) - length(h), 1)]))/length(x);
W =  fftshift(fft(hw(1:length(x))*sqrt(max(conv(h, flip(h))))/SNRlin(k)))/length(x);
psdXH = (real(XH).^2 + imag(XH).^2);
psdW =  (real(W).^2 + imag(W).^2);
fh = (-length(XH)/2: length(XH)/2-1) * (2*pi/length(XH));
%plots
hold on; plot(fh, psdXH); plot(fh, psdW);
legend('$|H_4(z) \cdot X(z)|^2$', '$|W(z)|^2$', 'interpreter', 'latex');
xlabel('\omega'); ylabel('Power Spectral Density', 'interpreter', 'latex'); xlim([-pi, pi]);
set(gca, 'FontSize', 14, 'yScale', 'log')


% Wiener Filter implementation
xpower = mean(x.^2);
wpower = mean(w.^2);
H = (fftshift(fft([zeros(round((length(x)-length(h))/2), 1); h; zeros(length(x)-length(h)-round((length(x)-length(h))/2), 1)]))/length(x));
Hconj = conj(H);
Hw = fftshift(fft(hw(1:length(H))))'/length(H);
psdH = abs(H).^2;
psdHw = abs(Hw).^2;
H_wiener = (xpower.*Hconj)./(xpower*psdH + wpower*psdHw);
% plots
hold on; plot(fh, psdXH); plot(fh, abs(H_wiener).^2, 'linewidth', 1); plot(fh, psdW);
legend('$|H_4(z) \cdot X(z)|^2$', 'Wiener filter PSD', 'Noise PSD', 'interpreter', 'latex');
xlabel('\omega'); ylabel('Power Spectral Density', 'interpreter', 'latex'); xlim([-pi, pi]);
set(gca, 'FontSize', 11, 'yScale', 'log')



%% Part b
%% B1)

% Nx = 750;
% p_val = (2:+2:100);
% hw = 0.95.^(0:(2*Nx+2*max(p_val)));
% cwwinv = inv(toeplitz(hw));
% Niter = 5;
% 
% load('HW1.mat');   % imports the Fs and x variables
% 
% h1 = zeros(max(p_val), 1);
% h2 = h1;
% h1(3) = 1;
% for ii = 3:length(h2)
%     h2(ii) = h1(ii) + 0.8*sqrt(2)*h2(ii-1) - 0.64*h2(ii-2);
% end
% h2(1:end-2) = h2(3:end);        % this is h12 and h21
% h1 = (0.9).^(0:max(p_val)-1)';   % this is h11 and h22
% 
% MSE1 = zeros(length(p_val), length(SNRlin), Niter);
% MSE2 = zeros(length(p_val), length(SNRlin), Niter);
% y = zeros(size(x));
% y1 = zeros(length(x), 1);
% y2 = y1;
% y1(1) = x(1, 1);
% y2(1) = x(1, 2);
% for i = 2:length(y1)
%     y1(i) = x(i, 1) + 0.9*y1(i-1);
%     y2(i) = x(i, 2) + 0.9*y2(i-1);
% end
% y = y + [y1, y2];
% y1 = zeros(length(x), 1);
% y2 = y1;
% y1(1:2) = x(1:2, 1);
% y2(1:2) = x(1:2, 2);
% for i = 3:length(y1)
%     y1(i) = x(i, 2) + 0.8*sqrt(2)*y1(i-1)- 0.64*y1(i-2);
%     y2(i) = x(i, 1) + 0.8*sqrt(2)*y2(i-1) - 0.64*y2(i-2);
% end
% y = y + [y1, y2];
% 
% y0 = y;
% for i = 1:Niter
%     w0 = randn(size(x));
%     w = w0;
%     for ii = 2:length(w0)
%         w0(ii, :) = w(ii, :) + 0.95*w0(ii-1, :);
%     end
%     w0(:, 1) = w0(:, 1)/sqrt(mean(w0(:, 1).^2));
%     w0(:, 2) = w0(:, 2)/sqrt(mean(w0(:, 2).^2));
%     for k = 1:length(SNRlin)
%         y = y0 + w0*sqrt(max(conv(h1, flip(h1))))/SNRlin(k);
%         for j = 1:length(p_val)     % here we should put length(p_val)
%             X = zeros(2*Nx + 2*p_val(j) - 2, 2*p_val(j));
%             for ii = 1:p_val(j)
%                 X(ii:Nx+ii-1, ii) = x(1:Nx, 1);     
%                 X(p_val(j)+Nx-1+ii:p_val(j)+2*Nx-2+ii, ii) = x(1:Nx, 2);
%                 X(ii:Nx+ii-1, p_val(j) + ii) = x(1:Nx, 2);
%                 X(p_val(j)+Nx-1+ii:p_val(j)+2*Nx-2+ii, p_val(j) + ii) = x(1:Nx, 1);
%             end
%             h_est = pinv(X'*cwwinv(1:2*Nx + 2*p_val(j) - 2, 1:2*Nx + 2*p_val(j) - 2)*X)*X'*cwwinv(1:2*Nx + 2*p_val(j) - 2, 1:2*Nx + 2*p_val(j) - 2)*[y(1:Nx+p_val(j)-1, 1);y(1:Nx+p_val(j)-1, 2)];
%             h1_est = h_est(1:p_val(j));
%             h2_est = h_est(p_val(j)+1:end);
%             MSE1(j , k, i) = mean((h1_est - h1(1:length(h1_est))).^2);
%             MSE2(j , k, i) = mean((h2_est - h2(1:length(h2_est))).^2);
%         end
%     end
%     100*i/Niter
% end
% 
% meanMSE1 = mean(MSE1, 3);
% meanMSE2 = mean(MSE2, 3);
% meanMSE = meanMSE1/2 + meanMSE2/2;
% 
% 
% hold on
% for i = 1:length(SNRdB)
%     semilogy(p_val, meanMSE(:, i));
%     legend('SNR = -20dB', 'SNR = -10dB', 'SNR = 0dB', 'SNR = 10dB', 'SNR = 20dB', 'SNR = 30dB')
%     xlabel('Estimated filter length p'); ylabel('Mean Square Error of h');
%     set(gca, 'yscale', 'log');
% end
% 
% 
% hold on; stem(h1(1:length(h1_est))); stem(h1_est);
% legend('$h_1[n]$','$\hat{h}_1[n]$', 'interpreter', 'latex');
% set(gca, 'FontSize', 14)
% xlabel('n'); ylabel('h[n]');
% 
% 
% hold on; stem(h2(1:length(h2_est))); stem(h2_est);
% legend('$h_2[n]$','$\hat{h}_2[n]$', 'interpreter', 'latex');
% set(gca, 'FontSize', 14)
% xlabel('n'); ylabel('h[n]');


% B2)

% 
% Nx = 500;
% p_val = 7;
% SNRdB = -30:+2:40;
% SNRlin = 10.^(SNRdB/20);
% Niter = 1;
% hw = 0.95.^(0:(2*Nx+2*p_val-3));
% cwwinv = inv([toeplitz(hw(1:length(hw)/2)), zeros(length(hw)/2); zeros(length(hw)/2), toeplitz(hw(1:length(hw)/2))]);
% 
% load('HW1.mat');   % imports the Fs and x variables
% 
% alpha = 0.6;                % alpha parameter
% MSE1 = zeros(length(SNRlin), Niter);
% MSE2 = MSE1;
% h1 = [exp(-1) exp(-0.75) exp(-0.5) exp(-0.25) exp(-0.5) exp(-0.75) exp(-1)]';
% h2 = h1*alpha;
% y = zeros(length(x)+length(h1)-1, 2);
% y(:, 1) = conv(h1, x(:, 1)) + conv(h2, x(:, 2));
% y(:, 2) = conv(h2, x(:, 1)) + conv(h1, x(:, 2));
% y0 = y;                     % noiseless y
% 
% 
% lambda_f1 = zeros(Niter, length(SNRdB));
% for i = 1:Niter
%     for k = 1:length(SNRdB)
%         w0 = randn(size(y0));
%         w = w0;
%         for ii = 2:length(w0)
%             w0(ii, :) = w(ii, :) + 0.95*w0(ii-1, :);
%         end
%         w0 = w0./sqrt(mean(mean(w0.^2)));
%         y = y0 + w0*sqrt(max(conv(h1, flip(h1))))/SNRlin(k);
%         X = zeros(2*Nx + 2*p_val - 2, 2*p_val);
%         for ii = 1:p_val
%             X(ii:Nx+ii-1, ii) = x(1:Nx, 1);
%             X(p_val+Nx-1+ii:p_val+2*Nx-2+ii, ii) = x(1:Nx, 2);
%             X(ii:Nx+ii-1, p_val + ii) = x(1:Nx, 2);
%             X(p_val+Nx-1+ii:p_val+2*Nx-2+ii, p_val + ii) = x(1:Nx, 1);
%         end
%         h_est = pinv(X'*cwwinv*X)*X'*cwwinv*[y(1:Nx+p_val-1, 1);y(1:Nx+p_val-1, 2)];
%         % h_est = pinv(X'*X)*X'*[y(1:Nx+p_val-1, 1);y(1:Nx+p_val-1, 2)];
%         h1_est = h_est(1:length(h_est)/2);
%         h2_est = h_est(length(h_est)/2+1:end);
%         MSE1(k, i) = mean((h1 - h1_est).^2);
%         MSE2(k, i) = mean((h2 - h2_est).^2);
%     end
%     i*100/Niter
% end
% meanMSE1 = mean(MSE1, 2);
% meanMSE2 = mean(MSE2, 2);
% MSE = (MSE1 + MSE2);
% meanMSE = (meanMSE1 + meanMSE2);
% 
% figure(1)
% hold on;
% for ii = 1:Niter
%     plot(SNRdB, MSE(:, ii), '.');
%     set(gca, 'yscale', 'log');
% end
% xlabel('SNR [dB]'); ylabel('Mean Square Error of h');
% plot(SNRdB, max(conv(h1, flip(h1)))*(SNRlin.^-2)*mean(diag(inv(X'*cwwinv*X))), '-.');
% set(gca, 'yscale', 'log'); xlim([SNRdB(1)-1, SNRdB(end)+1]);
% 
% figure(2)
% hold on
% plot(SNRdB, meanMSE)
% xlabel('SNR [dB]'); ylabel('Mean Square Error of h');
% plot(SNRdB, max(conv(h1, flip(h1)))*(SNRlin.^-2)*mean(diag(inv(X'*cwwinv*X))), '-.');
% legend('Mean Square Error of h', 'Cramer Rao Bound')
% set(gca, 'yscale', 'log'); xlim([SNRdB(1)-1, SNRdB(end)+1])
% 
% figure(3)
% hold on; stem((-3:3), h1); stem((-3:3), h1_est(1:7));
% stem((-3:3), h2); stem((-3:3), h2_est(1:7));
% title('SNR = 20dB')
% legend('$h_1[n]$','$\hat{h}_1[n]$', '$h_2[n]$','$\hat{h}_2[n]$', 'interpreter', 'latex');
% set(gca, 'FontSize', 14)
% xlabel('n'); ylabel('h[n]');
