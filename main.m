% Create folders if they don't exist
if ~exist('images', 'dir')
    mkdir('images');
end

if ~exist('output', 'dir')
    mkdir('output');
end

clc; clear; close all;

%% 2.1 Transmitter

%% Parameters
M = 16; % 16-QAM
k = log2(M);
num_bits = k * 1e5;
FsT = 2; % Oversampling factor
alpha = 0.1; % RRC roll-off factor
Nd = 10; % RRC duration in symbols
Rs = 40e6; % Symbol rate
Fs = Rs * FsT; % Sampling frequency
snr_db = 15;
h = [-0.3878 - 0.5538i, -0.4723 + 0.7558i, 0.0278 - 0.5302i, 0.2233 - 0.3868i, 0.0221 + 0.3155i];

%% Bit generation
bits = randi([0 1], num_bits, 1);

%% Gray Encoding & Symbol Mapping
symbols = bi2de(reshape(bits, [], k), 'left-msb');
symbols_gray = bitxor(symbols, floor(symbols/2));
modulated = qammod(symbols_gray, M, 'gray', 'UnitAveragePower', true);

% Plot constellation
disp('Plotting Transmitted Constellation...');
figure; plot(real(modulated), imag(modulated), '.'); grid on;
title('Transmitted Constellation'); xlabel('In-phase'); ylabel('Quadrature');

%% Upsampling and Pulse Shaping
upsampled = upsample(modulated, FsT);
rrc = rcosdesign(alpha, Nd, FsT, 'sqrt');
tx_filtered = conv(upsampled, rrc, 'full');
rrc_delay = (length(rrc)-1)/2;
tx_filtered = tx_filtered(rrc_delay+1:end-rrc_delay);

% Eye Diagram (Real Part)
disp('Plotting Transmitted Eye Diagram...');
eyediagram(real(tx_filtered), 2*FsT);
title('Eye Diagram - Transmitted Signal');

% Frequency response of filter
figure;
freqz(rrc, 1, 1024, Fs); title('RRC Filter Frequency Response');

% Spectrum of transmitted signal
figure;
N = length(tx_filtered);
f = linspace(-Fs/2, Fs/2, N);
Xf = fftshift(abs(fft(tx_filtered)));
plot(f/1e6, Xf); title('Spectrum of Transmitted Signal'); xlabel('Frequency (MHz)'); ylabel('Amplitude');

%% 2.2 Transmission over Channel

%% Channel + Noise
rx = conv(tx_filtered, h, 'full');
chan_delay = find(abs(h) == max(abs(h)), 1) - 1;
rx = rx(chan_delay+1:end);

% Add noise
rx = rx(1:length(tx_filtered)); % Match lengths
rx_noisy = awgn(rx, snr_db, 'measured');

% Channel Frequency Response
figure;
freqz(h, 1, 1024, Fs); title('Channel Frequency Response');

% Spectrum after channel + noise
figure;
Yf = fftshift(abs(fft(rx_noisy)));
plot(f/1e6, Yf); title('Spectrum after Channel and Noise'); xlabel('Frequency (MHz)'); ylabel('Amplitude');

%% 2.3 Receiver

%% Matched Filter
rx_filtered = conv(rx_noisy, rrc, 'full');
rx_filtered = rx_filtered(rrc_delay+1:end-rrc_delay);

% Eye Diagram at Receiver
disp('Plotting Received Eye Diagram...');
figure;
eyediagram(real(rx_filtered), 2*FsT);
title('Eye Diagram - Received Signal');

%% Downsampling
rx_down = rx_filtered(1:FsT:end);

% Constellation after channel and noise
figure; plot(rx_down, '.'); title('Constellation after Channel and Noise');

%% LMS Equalization
train_len = 1000;
mu = 0.001; eq_order = 20;
eq_weights = zeros(eq_order, 1);
eq_weights(floor(eq_order/2)) = 1;
e_input = buffer(rx_down, eq_order, eq_order-1, 'nodelay').';
desired = modulated(1:size(e_input,1));
error = zeros(length(desired), 1);

for n = 1:length(desired)
    y = eq_weights' * e_input(n, :)';
    error(n) = desired(n) - y;
    eq_weights = eq_weights + mu * error(n) * conj(e_input(n, :)');
end

% LMS Error
figure;
plot(abs(error)); title('LMS Error'); xlabel('Iteration'); ylabel('Error');

% Equalize full signal safely
rx_down_padded = [rx_down; zeros(length(eq_weights), 1)];
eq_out = conv(rx_down_padded, eq_weights, 'valid');
eq_out = eq_out(1:length(modulated));

% Check for NaNs
if any(isnan(eq_out))
    error('eq_out contains NaNs! LMS may have diverged.');
end

% Frequency response of equalizer
figure; freqz(eq_weights, 1, 1024, Fs);
title('Equalizer Frequency Response');

%% Constellation after equalization
figure;
plot(rx_down, '.'); hold on;
plot(eq_out, 'x'); hold on;
plot(modulated(1:length(eq_out)), 'o');
title('Equalized Constellation'); legend('Unequalized', 'Equalized', 'Original');

%% Symbol Detection and BER (No ECC yet)
rx_symbols = qamdemod(eq_out, M, 'gray', 'UnitAveragePower', true);
rx_bits = de2bi(rx_symbols, k, 'left-msb');
rx_bits = rx_bits(:);
rx_bits = rx_bits(1:length(bits));
ber = sum(bits ~= rx_bits) / length(bits);

fprintf('BER without ECC: %e\n', ber);



%% 2.4 Error Control Coding

%% ECC - Hamming(7,4)
% Generate G and H matrices
G = [eye(4) [1 0 1; 1 1 1; 1 1 0; 0 1 1]];
% Correct construction of parity-check matrix H for Hamming(7,4)
P = [1 0 1; 1 1 1; 1 1 0; 0 1 1];
% Correct: H should be a 3x7 matrix -> [P'; I3] gives 3x4 and 3x3, we must use horizontal concatenation
H = [P.' eye(3)];
