
import os
import sys
import subprocess
import multiprocessing
import re

whisper_cpp_base_path = "./whisper_cpp"
whisper_cpp_bin_position = os.path.join(whisper_cpp_base_path, "win64" if sys.platform == "win32" else "linux64", "main.exe" if sys.platform == "win32" else "main")

models_dict = {
    "tiny": "ggml-tiny.bin",
    "base": "ggml-base.bin",
    "small": "ggml-small.bin",
    "medium": "ggml-medium.bin",
    "large": "ggml-large.bin",
}

class WhisperCPP():
    threads = int(multiprocessing.cpu_count()/2)

    line_regex = re.compile(r"\[(\d\d:\d\d:\d\d.\d\d\d)\s-->\s(\d\d:\d\d:\d\d.\d\d\d)\]\s+(.*)", flags=re.MULTILINE)
    language_regex = re.compile(r"auto-detected language: (\w+) \(p = (\d+\.\d+)\)", flags=re.MULTILINE)
    def __init__(self, model_type):
        self.model_type = model_type

    def transcribe(self, audio_path, language="auto"):
        #.\main.exe -t 8 -l auto -m G:\whisper.cpp\models\ggml-medium.bin .\plateforme_16.wav
        argv = [whisper_cpp_bin_position, "-t", str(self.threads), "-l", language, "-m", os.path.join(whisper_cpp_base_path, "models", models_dict[self.model_type]), audio_path]
        result_txt = None


        output = subprocess.run(argv, capture_output=True, text=True) #better to use posix_spawn on linux but i couldn't manage to get stdout and stderr to work
        result_txt = output.stderr + output.stdout
        if sys.platform == "win32":
            result_txt = result_txt.encode('latin1').decode('utf8')


        if language == "auto":
            language = self.language_regex.findall(result_txt)[0][0]

        #expected segment = {"start": 0.0, "end": 0.0, "text": "text"}
        segments = []
        for segment in self.line_regex.findall(result_txt):
            start_txt = segment[0]
            end_txt = segment[1]
            text = segment[2]
            start_secs = float(start_txt.split(":")[0]) * 3600 + float(start_txt.split(":")[1]) * 60 + float(start_txt.split(":")[2])
            end_secs = float(end_txt.split(":")[0]) * 3600 + float(end_txt.split(":")[1]) * 60 + float(end_txt.split(":")[2])
            segments.append({"start": start_secs, "end": end_secs, "text": text})


        ret = {"language": language , "segments": segments}
        return ret
        



        


def whisper_load_model_cpp(model_type):
    return WhisperCPP(model_type)


if __name__ == "__main__":
    model = whisper_load_model_cpp("medium")



