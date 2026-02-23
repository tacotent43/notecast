# not sure, but why not?

class AudioPreparationError(Exception):
    """Base exception for audio preparation"""
    pass

class ResamplingError(AudioPreparationError):
    """Raises when resampling fails"""
    pass

class DownmixingError(AudioPreparationError):
    """Raises when downmixing to mono fails"""
    pass
