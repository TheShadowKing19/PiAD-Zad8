import sounddevice as sd
import soundfile as sf
from pydub import AudioSegment
from pydub.effects import normalize
import numpy as np
import matplotlib.pyplot as plt
import wave

# Zad 1
s, fs = sf.read('Recording.wav', dtype='float32')
amount_of_samples = len(s[:, 0])
print("Odtwarzam...")
# sd.play(s, fs)
status = sd.wait()
raw = wave.open('Recording.wav')
signal = raw.readframes(-1)
signal = np.frombuffer(signal, dtype=np.int16)
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
plt.plot(time*1000, s, linewidth=0.3)


# Zad 2
def energia(si):
    ej = 0
    for i in si:
        ej += i**2
    return ej


def z(zi):
    zj = 0
    for i in range(len(zi) - 1):
        if zi[i] * zi[i+1] >= 0:
            zj += 0
        elif zi[i] * zi[i+1] < 0:
            zj += 1
    return zj


ej, zj = [], []
print("Liczę...")
for i in range(len(s)):
    ej.append(energia(s[i:i+10]))
    zj.append(z(s[i:i+10]))

ej = ej / np.linalg.norm(ej)
zj = zj / np.linalg.norm(zj)

plt.plot(time*1000, zj, linewidth=0.3, color='green')
plt.plot(time*1000, ej, linewidth=0.3, color='red')
plt.title("Sygnał")
plt.legend(['Sygnał', 'Przejścia przez zera', 'Energia'])
plt.xlabel("Czas [ms]")
plt.ylabel("Wartości sygnału")
plt.show()

# Zad 3
fragment = s[4000:6048]
plt.plot(time[4000:6048]*1000, fragment, linewidth=0.3)
plt.show()
fragment_zamaskowany = np.hamming(len(fragment))
