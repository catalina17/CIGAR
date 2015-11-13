import librosa
import numpy
import os
import pylab


def spectrograms_for_genre(genre_name):
    """
    Extract spectrograms of audio files belonging to the given genre.

    Parameters
    ----------
    genre_name : str
        The name of the genre we wish to extract spectrograms for
    """

    # Create directory, if not already existent
    dir_path = os.path.abspath(os.path.join(os.getcwd(), '../../spectrograms/', genre_name))
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Produce a spectrogram for each audio file
    for song_index in range(0, 100):
        filepath = '000' + str(song_index) + '.wav'
        if song_index < 10:
            filepath = '0' + filepath
        filepath = genre_name + '/' + genre_name + '.' + filepath

        # Extract raw representation of signal from audio file
        audio_time_series, sampling_rate = librosa.load(
            os.path.abspath(os.path.join(os.getcwd(), '../../genres/', filepath)))

        # Produce the spectrogram with 128 frequency bins, FFT window of 1024 samples
        spectrogram = librosa.feature.melspectrogram(y=audio_time_series, sr=sampling_rate,
                                                     n_mels=128, n_fft=1024, fmax=10000)

        # Save the spectrogram image
        librosa.display.specshow(librosa.logamplitude(spectrogram, ref_power=numpy.max), fmax=10000)
        pylab.savefig(os.path.abspath(os.path.join(os.getcwd(), '../../spectrograms/',
                                                   filepath[:-3] + 'png')),
                      bbox_inches='tight', pad_inches=0)


if __name__ == '__main__':
    genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae',
              'rock']
    for genre in genres:
        spectrograms_for_genre(genre)
