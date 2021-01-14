import librosa
from deepspeech import Model


class DeepSpeechModel:
    def __init__(self):
        self.ds = self._load_model()

    @staticmethod
    def _load_model():
        ds = Model("deepspeech-0.9.3-models.pbmm")
        ds.enableExternalScorer("deepspeech-0.9.3-models.scorer")
        return ds

    def predict_to_string(self, audio, fs) -> str:
        return self._metadata_to_string(
            self.ds.sttWithMetadata(audio, fs).transcripts[0]
        )

    @staticmethod
    def _metadata_to_string(metadata):
        return "".join(token.text for token in metadata.tokens)


def deepspeech_predict():
    ds = Model("../deepspeech-0.9.3-models.pbmm")
    ds.enableExternalScorer("../deepspeech-0.9.3-models.scorer")
    from pathlib import Path
    from io import BytesIO
    audio, fs = librosa.load(
        BytesIO(Path("audio/8455-210777-0068.wav").read_bytes()), sr=None, dtype="int16", mono=False
    )
    return metadata_to_string(ds.sttWithMetadata(audio, fs).transcripts[0])


def metadata_to_string(metadata):
    return "".join(token.text for token in metadata.tokens)


if __name__ == "__main__":
    print(deepspeech_predict())
    # ...
