import sounddevice as sd
import soundfile as sf
from pydub import AudioSegment
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
import librosa

# Zad 1
s, fs = sf.read('Recording.wav', dtype='float32')
amount_of_samples = len(s[:, 0])
print("Odtwarzam...")
# sd.play(s, fs)
status = sd.wait()
s = s[:, 0]/np.linalg.norm(s[:, 0])

# Parametry dźwieku
audio = AudioSegment.from_wav("Recording.wav")
print("Parametry dźwięku:")
print(f"Kanały: {audio.channels}")
print(f"Bitów w próbce: {audio.sample_width}")  # 1 - 8 bitów, 2 - 16 bitów
print(f"Częstotliwość: {audio.frame_rate} Hz")
print(f"Długość nagrania (ms): {len(audio)}")


# Wyznaczenie czasu
time = np.arange(0, float(amount_of_samples)/audio.frame_rate, 1/audio.frame_rate)
fig, axd = plt.subplot_mosaic([['A plot', 'A plot', 'A plot', 'A plot'],
                               ['B plot', 'C plot', 'D plot', 'E plot'],
                               ['F plot', 'F plot', 'F plot', 'F plot']],
                              figsize=(12, 8), constrained_layout=True)
axd['A plot'].plot(time*1000, s, linewidth=0.3)


# Zad 2
def energia(si):
    """Funkcja do obliczenia funkcji energii E

    Args:
        si (list): wektor dźwięku (okno)
    Returns:
        float: wartość funkcji energii E
    """
    ej = 0
    for i in si:
        ej += i**2
    return ej


def z(zi):
    """Funkcja do obliczenia funkcji przejść przez zer Z

    Args:
        zi (list): wektor dźwięku (okno)
    Returns:
        float: wartość funkcji przejść przez zer Z
    """
    zj = 0
    for i in range(len(zi) - 1):
        if zi[i] * zi[i+1] >= 0:
            zj += 0
        elif zi[i] * zi[i+1] < 0:
            zj += 1
    return zj


# Print iterations progress
def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


ej, zj = [], []
printProgressBar(0, len(s), prefix='Liczę statystyki dla ramki 10ms',
                 suffix='ukończone', length=50)
for i in range(len(s)):
    ej.append(energia(s[i:i+10]))
    zj.append(z(s[i:i+10]))
    printProgressBar(i+1, len(s), prefix='Liczę statystyki dla ramki 10ms', suffix='ukończone', length=50)

# Normalizacja wektorów energii i zer
ej = ej / np.linalg.norm(ej)
zj = zj / np.linalg.norm(zj)

axd['A plot'].plot(time*1000, zj, linewidth=0.3, color='green')
axd['A plot'].plot(time*1000, ej, linewidth=0.3, color='red')
axd['A plot'].set_title("Sygnał")
axd['A plot'].legend(['Sygnał', 'Przejścia przez zera', 'Energia'])
axd['A plot'].set_xlabel("Czas [ms]")
axd['A plot'].set_ylabel("Wartości sygnału")


fig2, axd2 = plt.subplot_mosaic([['5ms'],
                                 ['20ms'],
                                 ['50ms']],
                                figsize=(8, 8), constrained_layout=True)

# ramka -> 5ms
ramka = 5
ej, zj = [], []
printProgressBar(0, len(s), prefix=f'Liczę statystyki dla ramki {ramka}ms',
                 suffix='ukończone', length=50)
for i in range(len(s)):
    ej.append(energia(s[i:i+ramka]))
    zj.append(z(s[i:i+ramka]))
    printProgressBar(i+1, len(s), prefix=f'Liczę statystyki dla ramki {ramka}ms', suffix='ukończone', length=50)
# Normalizacja wektorów energii i zer
ej = ej / np.linalg.norm(ej)
zj = zj / np.linalg.norm(zj)
# Wykresy
axd2['5ms'].plot(time*1000, s)
axd2['5ms'].plot(time*1000, zj, color='green')
axd2['5ms'].plot(time*1000, ej, color='red')
axd2['5ms'].set_xlabel("Czas [ms]")
axd2['5ms'].set_ylabel("Wartości sygnału")
axd2['5ms'].legend(['Sygnał', 'Przejścia przez zera', 'Energia'])
axd2['5ms'].set_title(f"Sygnał, ramka {ramka}ms")


# ramka -> 20ms
ramka = 20
ej, zj = [], []
printProgressBar(0, len(s), prefix=f'Liczę statystyki dla ramki {ramka}ms', suffix='ukończone', length=50)
for i in range(len(s)):
    ej.append(energia(s[i:i+ramka]))
    zj.append(z(s[i:i+ramka]))
    printProgressBar(i+1, len(s), prefix=f'Liczę statystyki dla ramki {ramka}ms', suffix='ukończone', length=50)
# Normalizacja wektorów energii i zer
ej = ej / np.linalg.norm(ej)
zj = zj / np.linalg.norm(zj)
# Wykresy
axd2['20ms'].plot(time*1000, s)
axd2['20ms'].plot(time*1000, zj, color='green')
axd2['20ms'].plot(time*1000, ej, color='red')
axd2['20ms'].set_xlabel("Czas [ms]")
axd2['20ms'].set_ylabel("Wartości sygnału")
axd2['20ms'].legend(['Sygnał', 'Przejścia przez zera', 'Energia'])
axd2['20ms'].set_title(f"Sygnał, ramka {ramka}ms")


# ramka -> 50ms
ramka = 50
ej, zj = [], []
printProgressBar(0, len(s), prefix=f'Liczę statystyki dla ramki {ramka}ms', suffix='ukończone', length=50)
for i in range(len(s)):
    ej.append(energia(s[i:i+ramka]))
    zj.append(z(s[i:i+ramka]))
    printProgressBar(i+1, len(s), prefix=f'Liczę statystyki dla ramki {ramka}ms', suffix='ukończone', length=50)
# Normalizacja wektorów energii i zer
ej = ej / np.linalg.norm(ej)
zj = zj / np.linalg.norm(zj)
# Wykresy
axd2['50ms'].plot(time*1000, s)
axd2['50ms'].plot(time*1000, zj, color='green')
axd2['50ms'].plot(time*1000, ej, color='red')
axd2['50ms'].set_xlabel("Czas [ms]")
axd2['50ms'].set_ylabel("Wartości sygnału")
axd2['50ms'].legend(['Sygnał', 'Przejścia przez zera', 'Energia'])
axd2['50ms'].set_title(f"Sygnał, ramka {ramka}ms")

fig2.show()


# Nakładanie ramek
fig3, axd3 = plt.subplot_mosaic([['10ms'],
                                 ['5ms'],
                                 ['20ms'],
                                 ['50ms']],
                                figsize=(10, 10), constrained_layout=True)
# ramka -> 10ms
ramka = 10
ej, zj = [], []
printProgressBar(0, len(s), prefix=f'Liczę statystyki dla nakładających się ramek {ramka}ms',
                 suffix='ukończone', length=50)
for i in range(len(s)):
    ej.append(energia(s[int(i-ramka/2):int(i + ramka / 2)]))
    zj.append(z(s[int(i-ramka/2):int(i+ramka/2)]))
    printProgressBar(i+1, len(s), prefix=f'Liczę statystyki dla nakładających się ramek {ramka}ms',
                     suffix='ukończone', length=50)
# Normalizacja wektorów energii i zer
ej = ej / np.linalg.norm(ej)
zj = zj / np.linalg.norm(zj)
# Wykresy
axd3['10ms'].plot(time*1000, s)
axd3['10ms'].plot(time*1000, zj, color='green')
axd3['10ms'].plot(time*1000, ej, color='red')
axd3['10ms'].set_xlabel("Czas [ms]")
axd3['10ms'].set_ylabel("Wartości sygnału")
axd3['10ms'].legend(['Sygnał', 'Przejścia przez zera', 'Energia'])
axd3['10ms'].set_title(f"Sygnał, nakładające się ramki {ramka}ms")

# ramka -> 5ms
ramka = 5
ej, zj = [], []
printProgressBar(0, len(s), prefix=f'Liczę statystyki dla nakładających się ramek {ramka}ms',
                 suffix='ukończone', length=50)
for i in range(len(s)):
    ej.append(energia(s[int(i-ramka/2):int(i+ramka/2)]))
    zj.append(z(s[int(i-ramka/2):int(i+ramka/2)]))
    printProgressBar(i+1, len(s), prefix=f'Liczę statystyki dla nakładających się ramek {ramka}ms',
                     suffix='ukończone', length=50)
# Normalizacja wektorów energii i zer
ej = ej / np.linalg.norm(ej)
zj = zj / np.linalg.norm(zj)
# Wykresy
axd3['5ms'].plot(time*1000, s)
axd3['5ms'].plot(time*1000, zj, color='green')
axd3['5ms'].plot(time*1000, ej, color='red')
axd3['5ms'].set_xlabel("Czas [ms]")
axd3['5ms'].set_ylabel("Wartości sygnału")
axd3['5ms'].legend(['Sygnał', 'Przejścia przez zera', 'Energia'])
axd3['5ms'].set_title(f"Sygnał, nakładające się ramki {ramka}ms")

# ramka -> 20ms
ramka = 20
ej, zj = [], []
printProgressBar(0, len(s), prefix=f'Liczę statystyki dla nakładających się ramek {ramka}ms',
                 suffix='ukończone', length=50)
for i in range(len(s)):
    ej.append(energia(s[(i-ramka//2):(i+ramka//2)]))
    zj.append(z(s[i-ramka//2:i+ramka//2]))
    printProgressBar(i+1, len(s), prefix=f'Liczę statystyki dla nakładających się ramek {ramka}ms',
                     suffix='ukończone', length=50)
# Normalizacja wektorów energii i zer
ej = ej / np.linalg.norm(ej)
zj = zj / np.linalg.norm(zj)
# Wykresy
axd3['20ms'].plot(time*1000, s)
axd3['20ms'].plot(time*1000, zj, color='green')
axd3['20ms'].plot(time*1000, ej, color='red')
axd3['20ms'].set_xlabel("Czas [ms]")
axd3['20ms'].set_ylabel("Wartości sygnału")
axd3['20ms'].legend(['Sygnał', 'Przejścia przez zera', 'Energia'])
axd3['20ms'].set_title(f"Sygnał, nakładające się ramki {ramka}ms")

# ramka -> 50ms
ramka = 50
ej, zj = [], []
printProgressBar(0, len(s), prefix=f'Liczę statystyki dla nakładających się ramek {ramka}ms',
                 suffix='ukończone', length=50)
for i in range(len(s)):
    ej.append(energia(s[i-ramka//2:i+ramka//2]))
    zj.append(z(s[i-ramka//2:i+ramka//2]))
    printProgressBar(i+1, len(s), prefix=f'Liczę statystyki dla nakładających się ramek {ramka}ms',
                     suffix='ukończone', length=50)
# Normalizacja wektorów energii i zer
ej = ej / np.linalg.norm(ej)
zj = zj / np.linalg.norm(zj)
# Wykresy
axd3['50ms'].plot(time*1000, s)
axd3['50ms'].plot(time*1000, zj, color='green')
axd3['50ms'].plot(time*1000, ej, color='red')
axd3['50ms'].set_xlabel("Czas [ms]")
axd3['50ms'].set_ylabel("Wartości sygnału")
axd3['50ms'].legend(['Sygnał', 'Przejścia przez zera', 'Energia'])
axd3['50ms'].set_title(f"Sygnał, nakładające się ramki {ramka}ms")


# Zad 3
fragment = s[34000:36048]
axd['B plot'].plot(time[34000:36048]*1000, fragment / np.linalg.norm(fragment))
axd['B plot'].set_title("Fragment")
axd['B plot'].set_xlabel("Czas [ms]")
axd['B plot'].set_ylabel("Wartości sygnału")

fragment_zamaskowany = np.hamming(len(fragment))
axd['C plot'].plot(time[34000:36048]*1000, fragment_zamaskowany)
axd['C plot'].set_title("Okno Hamminga dla fragmentu")
axd['C plot'].set_xlabel("Czas [ms]")
axd['C plot'].set_ylabel("Wartości sygnału")

WH = fragment * fragment_zamaskowany
axd['D plot'].plot(time[34000:36048]*1000, WH)
axd['D plot'].set_title("W * H")
axd['D plot'].set_xlabel("Czas [ms]")
axd['D plot'].set_ylabel("Wartości sygnału")

yf = scipy.fftpack.fft(WH)
yf = np.log(np.abs(yf))
axd['E plot'].plot(yf, color='red')
axd['E plot'].set_title("Widmo aplitudowe fragmentu")
axd['E plot'].set_xlabel("Częstotliwość [Hz]")
axd['E plot'].set_ylabel("Wartości sygnału")

axd['F plot'].plot(yf[:1000], color='red')
axd['F plot'].set_title("Widmo aplitudowe fragmentu (0-1000 Hz)")
axd['F plot'].set_xlabel("Częstotliwość [Hz]")
axd['F plot'].set_ylabel("Wartości sygnału")
fig.show()
fig3.show()


# Zad 4
okno = s[9000:11048]
p = 20
a = librosa.lpc(okno, order=p)
a = np.lib.pad(a, (0, 2048-len(a)))
widmoLPC = np.log(np.abs(scipy.fftpack.fft(a)))
widmoLPC = widmoLPC * -1
widmoOkna = np.log(np.abs(scipy.fftpack.fft(okno)))
plt.title("Liniowe Kodowanie Predykcyjne")
plt.plot(widmoOkna)
plt.plot(widmoLPC-5.5, color='red')
plt.legend(['Widmo okna', 'Widmo LPC'])
plt.xlabel("Częstotliwość [Hz]")
plt.ylabel("Wartości sygnału")
plt.show()
