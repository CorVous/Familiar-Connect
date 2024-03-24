import audioop
import numpy

def process_audio(buffer: bytes, samplerate: int, desiredSR: int, channels: int):
    audio_buffer, _ = audioop.ratecv(buffer, 2, channels, samplerate, desiredSR, None)
    if channels > 1:
        audio_buffer = audioop.tomono(audio_buffer, 2, 1, 0)
    audio_buffer = audioop.mul(audio_buffer, 2, 5)
    audio_np = numpy.frombuffer(audio_buffer, dtype=numpy.int16)
    complete_audio = audio_np.astype(numpy.float32) / 32768.0
    return complete_audio
