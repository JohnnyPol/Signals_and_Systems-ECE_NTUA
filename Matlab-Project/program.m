% A.M : 03121089
% First-Last name: Ioannis Polychronopoulos

clc
close all
% ------------------------------------------------------------------------%
% Exercise 1

%-(a)-

% Reading and saving the audio file to the vector audio
[audio,fs_audio] = audioread('Ioannis Polychronopoulos.wav');

% Listening to the audio
sound(audio,fs_audio);
pause(2);

%-(b)-

% Initializing a vector which is going to represent time from 0-2 seconds
time = linspace(0,2,length(audio));

% Plotting the audio signal
figure
plot(time,audio);
grid on
xlabel('Time (s)');
ylabel('Amplitude');
title('Audio Signal of my Name');
pause(2);

% Selecting an interval in which the signal appears to be periodic and has length 50ms
time_start = 1;
time_finish = 1.05;
start = fs_audio*time_start;
finish = fs_audio*time_finish;
cut_signal = audio(start:finish);

% Time interval for cut signal
interval = linspace(time_start,time_finish,length(cut_signal));

% Plotting the cut version of our signal
figure
plot(interval,cut_signal,'-r');
grid on
axis([time_start time_finish -0.2 0.2]);
xlabel('Time (s)');
ylabel('Amplitude');
title('Cut Audio Signal');
pause(2);

% Listening to the cut audio signal
sound(cut_signal,fs_audio);

%-(c)-

% Normalizing the audio signal
normalized_signal = normalize(audio,'range',[-1 1]);

% Finding the energy of the signal with convolution
energy = conv(normalized_signal.^2,hamming(200));

% Plotting the energy of the signal and the signal itself
figure

% Normalizing the already normalized signal to change the scales properly
normalized_signal = normalize(audio, 'range',[min(energy),max(energy)]);
plot(time,normalized_signal,'-g');
grid on
hold on
energy_time = linspace(0,2,length(energy));
plot(energy_time, energy);
hold off
legend('Audio Signal','Energy Signal');
xlabel('Time (s)');
title('Signal and Energy');
pause(2);

%-(d)-

% Applying the Discrete Fourier Transform (DFT) at our cut signal
DFT_cut_signal = fft(cut_signal,1024);

% Initializing the frequency vector
fft_frequency= linspace(0,fs_audio,length(DFT_cut_signal));

% Plotting the DFT in linear scale
figure
subplot(1,2,1);
plot(fft_frequency,abs(DFT_cut_signal));
grid on
xlabel('Frequency (Hz)');
ylabel('Amplitude');
title('DFT in linear scale');

% Plotting the DFT in logarithmic scale
subplot(1,2,2);
plot(fft_frequency, 20*log10(abs(DFT_cut_signal)),'r');
grid on
xlabel('Frequency (Hz)');
ylabel('Amplitude');
title('DFT in logarithmic scale');
pause(2);

%-------------------------------------------------------------------------%

% Exercise 2

%-(a)-

% Reading and saving the image
leaf = imread('leaf3.png');

% Displaying the image
figure
imshow(leaf);
pause(2);

%-(b)-

hold on

% With the find function find all the positions of the 1's
[row, column] = find(leaf);

% Using the bwtraceboundary function to trace the boundary
contour = bwtraceboundary(leaf,[row(1),column(1)],'N');
plot(contour(:,2),contour(:,1),'g','Linewidth',2);
pause(2);

hold off

% Ploting the x and y values
figure
x = contour(:,2);
y = contour(:,1);
plot(x);
grid on
hold on
plot(y);
hold off
title('x-Boundary and y-Boundary');
legend('x[n]','y[n]')
pause(2);

%-(c)-

z = x + 1j*y;

% Calculating the coefficients of the discrete fourier series with fft()
Z = fft(z);
figure
plot(abs(Z),'Linewidth',3);
grid on
title('Spectrum of the Coefficients');
xlabel('Frequency (Hz)');
pause(2);

%-(d)-

N = length(z);
n = 1:N;

for M = [10,50,200]

    Zm = zeros(1,N);
    for k = 0:M
        Zm = Zm + (Z(k+1)*exp((2j*pi*k*n)/N))/N;
    end


    % Now we isolate the x and the y value out of the Zm variable
    Yz = round(abs(real(Zm)));
    Xz = round(abs(imag(Zm)));

    % Reconstructing the 317x350 image
    reconstructed_image = zeros(317,350);
    for index = 1:length(Xz)
        reconstructed_image(Xz(index)+1,Yz(index)+1) = 1;
    end

    % Showing the images
    drawnow;
    figure(8)
    imshow(reconstructed_image);
    hold on
    pause(2);
end

hold off

%-(e)-

for M = [10,50,200]

    Zm = zeros(1,N);
    for k = 0:M/2
        Zm = Zm + (Z(k+1)*exp((2j*pi*(k)*n)/N))/N;
    end

    for k = N-M/2: N-1
        Zm = Zm + (Z(k+1)*exp((2j*pi*k*n)/N))/N;
    end

    % Now we isolate the x and the y value out of the Zm variable
    Yz = round(abs(real(Zm)))';
    Xz = round(abs(imag(Zm)))';

    % Reconstructing the 350x317 image

    reconstructed_image = zeros(317,350);
    for index = 1:length(Xz)
        reconstructed_image(Xz(index)+1,Yz(index)+1) = 1;
    end

    % Showing the images
    drawnow;
    figure(9)
    imshow(reconstructed_image);
    hold on
    pause(2);
end

%-(st)-

% Reading and saving the image
rect = imread('rect.png');
rect = im2bw(rect);

% Displaying the image
figure
imshow(rect);
pause(2);

hold on

% With the find function find all the positions of the 1's
[row, column] = find(rect);

% Using the bwtraceboundary function to trace the boundary
contour = bwtraceboundary(rect,[row(1),column(1)],'N');
plot(contour(:,2),contour(:,1),'g','Linewidth',2);
pause(2);

% Ploting the x and y values
figure
x = contour(:,2);
y = contour(:,1);
plot(x);
grid on
hold on
plot(y);
hold off
title('x-Boundary and y-Boundary');
legend('x[n]','y[n]')
pause(2);

z = x + 1j*y;

% Calculating the coefficients of the discrete fourier series with fft()
Z = fft(z);
figure
plot(abs(Z),'Linewidth',3);
grid on
title('Spectrum of the Coefficients');
xlabel('Frequency (Hz)');
pause(2);

N = length(z);
n = 1:N;

for M = [10,50,200]
    Zm = zeros(1,N);

    for k = 0:M
        Zm = Zm + (Z(k+1)*exp((2j*pi*k*n)/N))/N;
    end


    % Now we isolate the x and the y value out of the Zm variable
    Yz = abs(round(real(Zm)));
    Xz = abs(round(imag(Zm)));

    % Reconstructing the 256x256 image
    reconstructed_image = zeros(300,400);
    for index = 1:length(Xz)
        reconstructed_image(Xz(index)+1,Yz(index)+1) = 1;
    end

    % Showing the images
    figure(13)
    drawnow;
    imshow(reconstructed_image);
    hold on
    pause(2);
end

hold off

for M=[10,50,200]
    Zm = zeros(1,N);

    for k = 0:M/2
        Zm = Zm + (Z(k+1)*exp((2j*pi*k*n)/N))/N;
    end


    for k = N-M/2: N-1
        Zm = Zm + (Z(k+1)*exp((2j*pi*k*n)/N))/N;
    end

    % Now we isolate the x and the y value out of the Zm variable
    Yz = round(abs(real(Zm)));
    Xz = round(abs(imag(Zm)));

    % Reconstructing the 256x256 image
    reconstructed_image = zeros(300,400);
    for index = 1:length(Xz)
        reconstructed_image(Xz(index)+1,Yz(index)+1) = 1;
    end

    % Showing the images
    figure(14)

    drawnow;
    imshow(reconstructed_image);
    hold on
    pause(2);
end

hold off

%-------------------------------------------------------------------------%

% Exercise 3.1

%-(a)-

% Storing the poles and the zeroes into vectors
poles = [0.51 + 1j*0.68, 0.51 - 1j*0.68];
zeroes = [0.8, -1];

% Plotting the diagram poles-zeroes
figure
% zplane requires column vectors so we use the transpose function
zplane(transpose(zeroes),transpose(poles));
title('Poles-Zeroes Diagram');
pause(2);

% Calculating the vector coefficients a and b
K = 0.15;
[num, den] = zp2tf(transpose(zeroes),transpose(poles),K);
% Numerator: Coefficients of the output variable Y(z)
% Denominator: Coefficients of the input variable X(z)

%-(b)-

% Plotting the Amplitute and Phase Response of the filter
figure
freqz(num,den);
title('Filter''s Magnitude and Phase Response')
pause(2);

%-(c)-

% Plotting the impulse response of the system
figure
impz(num,den);
grid on
title('System''s Impulse Response');
pause(2);

% Plotting the step response of the system
figure
stepz(num,den);
grid on
title('System''s Step Response');
pause(2);

%-(d)-

poles = [
        [0.57 + 1j*0.76,0.57 - 1j*0.76]
        [0.6 + 1j*0.8, 0.6 - 1j*0.8]
        [0.63 + 1j*0.84,0.63 - 1j*0.84]
        ];
zeroes = zeroes';

% For each pair of poles plot the desired values
for i=1:3
    [num, den] = zp2tf(zeroes,[poles(i,1),poles(i,2)]',K);
    figure
    zplane(zeroes,[poles(i,1),poles(i,2)]');
    title("Poles-Zeroes Diagram of pole number "+i);
    pause(2);

    if i==1 || i==2
        figure
        stepz(num,den);
        grid on
        title("System's Step Response with pole number "+i);
        pause(2);
        figure
        freqz(num,den);
        title("Filter''s Magnitude and Phase Response with pole number "+i);
        pause(2);
    else
        figure
        stepz(num,den);
         title("System's Step Response with pole number "+i);
        grid on
        pause(2);
    end
end

%-(e)-

poles = [0.51 + 1j*0.68, 0.51 - 1j*0.68]';
[num, den] = zp2tf(zeroes,poles,K);

% gensig syntax:
%(type of signal, period of signal, duration of signal, distance of successive steps)
x = gensig("sine",20/3,100,1)+ gensig("sine",20/7,100,1);

% Filtered Output Signal
y = filter(num,den,x);

% Plotting the input and output signals
figure
plot(0:1:100,x,'LineWidth',1);
hold on
plot(0:1:100,y,'LineWidth',2.5,'Color',[1 0.513 0]);
plot(0:1:100,gensig("sine",20/3,100,1),"Color",[0 0 0])
hold off
xlabel('Time (s)');
ylabel('Amplitude');
legend('Input: x[n]','Output: y[n]', 'sin(0.3πn)');
title("Input and Output Signals");
pause(2);

%-(st)-

poles = [0.68 + 1j*0.51, 0.68 + 1j*0.51, 0.68 - 1j*0.51, 0.68 - 1j*0.51]';

% Plotting the diagram poles-zeroes
figure
zplane(zeroes,poles);
title('Poles-Zeroes Diagram');
pause(2);

% Calculating the vector coefficients a and b
[num, den] = zp2tf(zeroes,poles,K);

% Plotting the Amplitute and Phase Response of the filter
figure
freqz(num,den);
title('Filter''s Magnitude and Phase Response')
pause(2);

%-------------------------------------------------------------------------%

% Exercise 3.2

%-(a)-

% Reading and saving the audio file to the variable viola_series
[viola_series, fs_viola_series] = audioread('viola_series.wav');
info_series = audioinfo('viola_series.wav');

% Initializing a vector which is going to represent time from 0-size seconds
time_viola_series = linspace(0,info_series.Duration, length(viola_series));

% plotting the audio signal
figure
plot(time_viola_series,viola_series);
grid on;
xlabel('Time (s)');
ylabel('Amplitude');
axis([0 info_series.Duration -0.04 0.04]);
title('Audio Signal of Viola');
pause(2);

% Listening to the audio
sound(viola_series,fs_viola_series);
pause(9);

%-(b)-

% Applying the Discrete Fourier Transform (DFT) at our cut signal
DFT_viola = fft(viola_series);
Freq = linspace(0,fs_viola_series,length(DFT_viola));

% Plotting the DFT in linear scale
figure
plot(Freq,abs(DFT_viola));
grid on
xlabel('Frequency (Hz)');
ylabel('Magnitude');
title('DFT in linear scale');
pause(2);

%-(c)-

% Load the signal
[viola_note, fs_viola_note] = audioread('viola_note.wav');

DFT_viola_note = fft(viola_note);
viola_Frequency = linspace(0,fs_viola_note,length(viola_note));

% Plotting the DFT
figure
plot(viola_Frequency,abs(DFT_viola_note));
grid on
xlabel('Frequency (kHz)');
ylabel('Magnitude');
title('DFT in linear scale');
pause(2);

% Computing the index that corresponds to the fundamentalfrequency in the viola_Frequency vector
[~,I] = max(abs(DFT_viola_note));

% Find the fundamental frequency while knowing the index
fundamental_frequency = viola_Frequency(I);

% Display the results
disp(['Fundamental frequency: ',num2str(fundamental_frequency),' Hz']);
pause(2);

%-(d)-

K = [0.000000003,0.00000003];
index = 1;
% For the second and third harmonic
for harmonics = [2*fundamental_frequency, 3*fundamental_frequency]

    info_note = audioinfo('viola_note.wav');

    bandwidth = 20; % width of the passband around the second harmonic

    % define the range of the second harmonic
    fmin = harmonics - bandwidth/2; % lower cutoff frequency
    fmax = harmonics + bandwidth/2; % upper cutoff frequency

    % Design the poles and zeroes of the filter
    Wn = [fmin fmax]/(fs_viola_note/2); % Normalized cutoff frequencies
    [z,p,~] = butter(3,Wn,'bandpass');

    % Convert the poles and zeroes to the numerator and denominator polynomials
    [num,den] = zp2tf(z,p,K(index));
    index = index + 1;
    % Apply the filter to the signal
    y_harmonic = filter(num,den,viola_note);

    % Apply the DFT
    y_harmonic_DFT = fft(y_harmonic);


    figure
    freqz(num,den);
    if  harmonics/fundamental_frequency == 2
        title('Filter''s Magnitude and Phase Response at the 2nd Harmonic')
    else
        title('Filter''s Magnitude and Phase Response at the 3rd Harmonic')
    end
    pause(2);

    figure
    plot(viola_Frequency, abs(y_harmonic_DFT));
    grid on
    xlabel('Frequency (kHZ)');
    ylabel('Amplitude');
    if  harmonics/fundamental_frequency == 2
        title('2nd Harmonic');
    else
        title('3rd Harmonic');
    end
    pause(2);

    time_viola_note = linspace(0,info_note.Duration, length(viola_note));
    figure
    plot(time_viola_note,y_harmonic);
    if  harmonics/fundamental_frequency == 2
        title("Viola Note Allowing Only the 2nd Harmonic");
    else
         title("Viola Note Allowing Only the 3rd Harmonic");
    end
    xlabel('Time (s)');
    pause(2);

end

%-------------------------------------------------------------------------%

% Exercise 4

%-(a)-

% Load the audio file
[mixture, fs_mixture] = audioread('mixture.wav');
info_mixture = audioinfo("mixture.wav");

% Listening to the audio
sound(mixture);
pause(7.5);

% Taking the DFT of the signal
mixture_DFT = fft(mixture);
mixture_frequency = linspace(0,fs_mixture,length(mixture));

% Plotting the fft
figure
plot(mixture_frequency,abs(mixture_DFT));
xlabel('Frequency (Hz)')
ylabel('Amplitute');
title("Mixture Signal Spectrum");
pause(2);

%-(b)-

FirstNoteHarmonics = zeros(1,5);
SecondNoteHarmonics = zeros(1,5);
FirstNote_FundamentalFrequency = 350;
SecondNote_FundamentalFrequency = 440;

% The expected first five harmonics of the two notes are the following
for i=1:5
    FirstNoteHarmonics(i) = FirstNote_FundamentalFrequency * i;
    SecondNoteHarmonics(i) = SecondNote_FundamentalFrequency * i;
end

% Now we normalize all the above frequencies
Normalized_First_Note_Harmonics = 2 * pi .* FirstNoteHarmonics / fs_mixture;
Normalized_Second_Note_Harmonics = 2 * pi .* SecondNoteHarmonics / fs_mixture;

% Displaying the computations
disp("First five harmonics for the first note:");
disp(FirstNoteHarmonics);
disp("First five harmonics for the second note:");
disp(SecondNoteHarmonics);
disp("Normalized frequencies for the first note:");
disp(Normalized_First_Note_Harmonics);
disp("Normalized frequencies for the first note:");
disp(Normalized_Second_Note_Harmonics);
pause(2);

%-(c)-

bandwidth = 20; % width of the passband around the harmonic

% Initialize the filtered signals
y1_filtered = zeros(size(mixture));
y2_filtered = zeros(size(mixture));

K = 0.00000006;

% Iterate over the range of harmonic frequencies
for i = 1:5
    % current harmonic frequency
    fmin1 = FirstNoteHarmonics(i) - bandwidth/2;
    fmax1 = FirstNoteHarmonics(i) + bandwidth/2;
    fmin2 = SecondNoteHarmonics(i) - bandwidth/2;
    fmax2 = SecondNoteHarmonics(i) + bandwidth/2;

    % Design the band-pass filters
    Wn1 = [fmin1 fmax1]/(fs_mixture/2);
    [z1,p1,~] = butter(3,Wn1,'bandpass');
    Wn2 = [fmin2 fmax2]/(fs_mixture/2);
    [z2,p2,~] = butter(3,Wn2,'bandpass');

    [num1,den1] = zp2tf(z1,p1,K);
    [num2,den2] = zp2tf(z2,p2,K);
    figure
    freqz(num1,den1);
    title("First signal harmonic number "+i);
    pause(2);
    figure
    freqz(num2,den2);
    title("Second signal harmonic number "+i);
    pause(2);
    % Apply the filter to the signal
    y1_filtered = y1_filtered + filter(num1,den1,mixture);
    y2_filtered = y2_filtered + filter(num2,den2,mixture);
end

%-(d)-

% Taking the Fast Fourier Transform
y1_fourier = fft(y1_filtered);
y2_fourier = fft(y2_filtered);
y1_frequency = linspace(0,fs_mixture,length(y1_filtered));
y2_frequency = linspace(0,fs_mixture,length(y2_filtered));

% Plotting the DFT of the first note
figure
plot(y1_frequency,abs(y1_fourier));
title('First Note');
xlabel('Frequency (Hz)');
ylabel('Amplitute');
pause(2);

% Plotting the DFT of the second note
figure
plot(y2_frequency, abs(y2_fourier));
title('Second Note');
xlabel('Frequency (Hz)');
ylabel('Amplitute');
pause(2);

y_fourier = y1_fourier + y2_fourier;
y_filtered = ifft(y_fourier);
y_frequency = linspace(0,fs_mixture,length(y_filtered));

% Plotting the DFT the sum which is the same as the fft of the mixture audio
figure
plot(y_frequency,abs(y_fourier));
title('First and Second Note');
xlabel('Frequency (Hz)');
ylabel('Amplitute');
pause(2);

% Plotting the reconstructed and the original signal
time_mixture = linspace(0,info_mixture.Duration,length(mixture));
figure
subplot(2,1,1);
plot(time_mixture,mixture,'-r');
xlabel('Time (s)');
ylabel('Amplitude');
title('Audio Signal');
ylim([-1 1])
subplot(2,1,2);
plot(time_mixture,y_filtered);
xlabel('Time (s)');
ylabel('Amplitude');
title('Composition of First and Second Note');
ylim([-1 1])
pause(2);

% Listening to the reconstructed audio
sound(y_filtered); % Sounds close to the original audio
pause(7);

% Load the source signals
[flute,fs_flute] = audioread("flute_acoustic_002-069-025.wav");
[reed, fs_reed] = audioread("reed_acoustic_037-065-075.wav");

% Create the proper time intervals
info_flute = audioinfo("flute_acoustic_002-069-025.wav");
info_reed = audioinfo("reed_acoustic_037-065-075.wav");

time_flute = linspace(0,info_flute.Duration,length(flute));
time_reed = linspace(0,info_reed.Duration,length(reed));

% Plotting the reed and the first note
figure
subplot(2,1,1);
plot(time_reed,reed,'-r');
title("Reed Accoustic Signal");
xlabel("Time (s)");
ylabel("Magnitute");
subplot(2,1,2);
plot(time_reed,y1_filtered);
title("First Note");
xlabel("Time (s)");
ylabel("Magnitute");
pause(2);

% Plotting the flute and the second note
figure
subplot(2,1,1);
plot(time_flute,flute,'-r');
title("Flute Accoustic Signal");
xlabel("Time (s)");
ylabel("Magnitute");
subplot(2,1,2);
plot(time_flute,y2_filtered);
title("Second Note");
xlabel("Time (s)");
ylabel("Magnitute");
pause(2);

% Creating a time interval with length 0.5 seconds
startup_time = 0.7;
start_flute = startup_time*fs_flute;
end_flute = (startup_time+0.5)*fs_flute;
cut_signal_flute = flute(start_flute:end_flute);
cut_signal_y2_filtered = y2_filtered(start_flute:end_flute);
interval_flute = linspace(startup_time,startup_time+0.5,length(cut_signal_flute));

% Plotting the cut for the flute and the cut second note
figure
subplot(2,1,1);
plot(interval_flute,cut_signal_flute,'-r');
xlabel('Time (s)');
ylabel('Amplitude');
axis([startup_time startup_time+0.5 -0.5 0.5])
title('Fluto Cut Audio Signal');
subplot(2,1,2);
plot(interval_flute,cut_signal_y2_filtered);
axis([startup_time startup_time+0.5 -0.5 0.5])
xlabel('Time (s)');
ylabel('Amplitude');
title('Cut Second Note');
pause(2);

% Creating a time interval with length 0.3 seconds
startup_time = 1.5;
start_reed = startup_time*fs_reed;
end_reed = (startup_time+0.3)*fs_reed;
cut_signal_reed = reed(start_reed:end_reed);
cut_signal_y1_filtered = y1_filtered(start_reed:end_reed);
interval_reed = linspace(startup_time,startup_time+0.3,length(cut_signal_reed));


% Plotting the cut for the reed and the cut first note
figure
subplot(2,1,1);
plot(interval_reed,cut_signal_reed,'-r');
xlabel('Time (s)');
ylabel('Amplitude');
axis([startup_time startup_time+0.3 -0.5 0.5])
title('Reed Cut Audio Signal');
subplot(2,1,2);
plot(interval_reed,cut_signal_y1_filtered);
axis([startup_time startup_time+0.3 -0.5 0.5])
xlabel('Time (s)');
ylabel('Amplitude');
title('Cut First Note');
pause(2);

%-(e)-

% Load the audio file
[mixture2, fs_mixture] = audioread('mixture2.wav');
info_mixture = audioinfo("mixture2.wav");

% Listening to the audio
sound(mixture2);
pause(7.5);

% Taking the DFT of the signal
mixture_DFT = fft(mixture2);
mixture_frequency = linspace(0,fs_mixture,length(mixture_DFT));

% Plotting the fft
figure
plot(mixture_frequency,abs(mixture_DFT));
xlabel('Frequency (Hz)')
ylabel('Amplitute');
title("Mixture Signal Spectrum");
pause(2);

% Finding the harmonics
FirstNoteHarmonics = zeros(1,5);
SecondNoteHarmonics = zeros(1,5);
FirstNote_FundamentalFrequency = 440;
SecondNote_FundamentalFrequency = 220;

% The expected first five harmonics of the two notes are the following
for i=1:5
    FirstNoteHarmonics(i) = FirstNote_FundamentalFrequency * i;
    SecondNoteHarmonics(i) = SecondNote_FundamentalFrequency * i;
end

% Now we normalize all the above frequencies
Normalized_First_Note_Harmonics = 2 * pi .* FirstNoteHarmonics / fs_mixture;
Normalized_Second_Note_Harmonics = 2 * pi .* SecondNoteHarmonics / fs_mixture;

% Displaying the computations
disp("First five harmonics for the first note:");
disp(FirstNoteHarmonics);
disp("First five harmonics for the second note:");
disp(SecondNoteHarmonics);
disp("Normalized frequencies for the first note:");
disp(Normalized_First_Note_Harmonics);
disp("Normalized frequencies for the first note:");
disp(Normalized_Second_Note_Harmonics);
pause(2);

% Width of the passband around the harmonic
bandwidth = 20;

% Initialize the filtered signals
y1_filtered = zeros(size(mixture2));
y2_filtered = zeros(size(mixture2));

K = 0.00000006;

% Iterate over the range of harmonic frequencies
for i = 1:5
    % current harmonic frequency
    fmin1 = FirstNoteHarmonics(i) - bandwidth/2;
    fmax1 = FirstNoteHarmonics(i) + bandwidth/2;
    fmin2 = SecondNoteHarmonics(i) - bandwidth/2;
    fmax2 = SecondNoteHarmonics(i) + bandwidth/2;

    % Design the band-pass filters
    Wn1 = [fmin1 fmax1]/(fs_mixture/2);
    [z1,p1,~] = butter(3,Wn1,'bandpass');
    Wn2 = [fmin2 fmax2]/(fs_mixture/2);
    [z2,p2,~] = butter(3,Wn2,'bandpass');

    [num1,den1] = zp2tf(z1,p1,K);
    [num2,den2] = zp2tf(z2,p2,K);
    figure
    freqz(num1,den1);
    title("First signal harmonic number "+i);
    pause(2);
    figure
    freqz(num2,den2);
    title("Second signal harmonic number "+i);
    pause(2);
    % Apply the filter to the signal
    y1_filtered = y1_filtered + filter(num1,den1,mixture2);
    y2_filtered = y2_filtered + filter(num2,den2,mixture2);
end


% Taking the Fast Fourier Transform
y1_fourier = fft(y1_filtered);
y2_fourier = fft(y2_filtered);
y1_frequency = linspace(0,fs_mixture,length(y1_filtered));
y2_frequency = linspace(0,fs_mixture,length(y2_filtered));

% Plotting the DFT of the first note
figure
plot(y1_frequency,abs(y1_fourier));
title('First Note');
xlabel('Frequency (Hz)');
ylabel('Amplitute');
pause(2);

% Plotting the DFT of the second note
figure
plot(y2_frequency, abs(y2_fourier));
title('Second Note');
xlabel('Frequency (Hz)');
ylabel('Amplitute');
pause(2);

y_fourier = y1_fourier + y2_fourier;
y_filtered = ifft(y_fourier);
y_frequency = linspace(0,fs_mixture,length(y_filtered));

% Plotting the DFT the sum which is the same as the fft of the mixture audio
figure
plot(y_frequency,abs(y_fourier));
title('First and Second Note');
xlabel('Frequency (Hz)');
ylabel('Amplitute');
pause(2);

% Plotting the reconstructed and the original signal
time_mixture = linspace(0,info_mixture.Duration,length(mixture2));
figure
subplot(2,1,1);
plot(time_mixture,mixture2,'-r');
xlabel('Time (s)');
ylabel('Amplitude');
ylim([-2.1 2.1])
title('Audio Signal');
subplot(2,1,2);
plot(time_mixture,y_filtered);
xlabel('Time (s)');
ylabel('Amplitude');
ylim([-2.1 2.1])
title('Composition of First and Second Note');
pause(2);

% Load the source signals
[flute,fs_flute] = audioread("flute_acoustic_002-069-025.wav");
[reed, fs_reed] = audioread("reed_acoustic_037-057-127.wav");

% Listening to the reconstructed audio
sound(y_filtered);
pause(7);

% Create the proper time intervals
info_flute = audioinfo("flute_acoustic_002-069-025.wav");
info_reed = audioinfo("reed_acoustic_037-057-127.wav");

time_flute = linspace(0,info_flute.Duration,length(flute));
time_reed = linspace(0,info_reed.Duration,length(reed));

% Plotting the reed and the first note
figure
subplot(2,1,1);
plot(time_reed,reed,'-r');
title("Reed Accoustic Signal");
xlabel("Time (s)");
ylabel("Magnitute");
subplot(2,1,2);
plot(time_reed,y1_filtered);
title("First Note");
xlabel("Time (s)");
ylabel("Magnitute");
pause(2);

% Plotting the flute and the second note
figure
subplot(2,1,1);
plot(time_flute,flute,'-r');
title("Flute Accoustic Signal");
xlabel("Time (s)");
ylabel("Magnitute");
subplot(2,1,2);
plot(time_flute,y2_filtered);
title("Second Note");
xlabel("Time (s)");
ylabel("Magnitute");
pause(2);

% Creating a time interval with length 0.5 seconds
startup_time = 0.7;
start_flute = startup_time*fs_flute;
end_flute = (startup_time+0.5)*fs_flute;
cut_signal_flute = flute(start_flute:end_flute);
cut_signal_y2_filtered = y2_filtered(start_flute:end_flute);
interval_flute = linspace(startup_time,startup_time+0.5,length(cut_signal_flute));

% Plotting the cut for the flute and the cut second note
figure
subplot(2,1,1);
plot(interval_flute,cut_signal_flute,'-r');
xlabel('Time (s)');
ylabel('Amplitude');
axis([startup_time startup_time+0.5 -0.7 0.7])
title('Fluto Cut Audio Signal');
subplot(2,1,2);
plot(interval_flute,cut_signal_y2_filtered);
axis([startup_time startup_time+0.5 -0.7 0.7])
xlabel('Time (s)');
ylabel('Amplitude');
title('Cut First Note');
pause(2);

% Creating a time interval with length 0.3 seconds
startup_time = 1.5;
start_reed = startup_time*fs_reed;
end_reed = (startup_time+0.3)*fs_reed;
cut_signal_reed = reed(start_reed:end_reed);
cut_signal_y1_filtered = y1_filtered(start_reed:end_reed);
interval_reed = linspace(startup_time,startup_time+0.3,length(cut_signal_reed));


% Plotting the cut for the reed and the cut first note
figure
subplot(2,1,1);
plot(interval_reed,cut_signal_reed,'-r');
xlabel('Time (s)');
ylabel('Amplitude');
axis([startup_time startup_time+0.3 -1 1])
title('Reed Cut Audio Signal');
subplot(2,1,2);
plot(interval_reed,cut_signal_y1_filtered);
axis([startup_time startup_time+0.3 -1 1])
xlabel('Time (s)');
ylabel('Amplitude');
title('Cut Second Note');
pause(2);


% Plotting the original signal, the first note and the second note
figure
subplot(4,1,1);
plot(time_mixture,mixture2);
xlabel('Time (s)');
ylabel('Amplitude');
title('Audio Signal');
subplot(4,1,2);
plot(time_mixture,y1_filtered,'-r');
title("First Note");
xlabel("Time (s)");
ylabel("Magnitute");
subplot(4,1,3);
plot(time_mixture,y2_filtered,'-g');
title("Second Note");
xlabel("Time (s)");
ylabel("Magnitute");
subplot(4,1,4);
plot(time_mixture,y_filtered,"Color",'#0F6292');
title("First and Second audio");
xlabel("Time (s)");
ylabel("Magnitute");
pause(2);















