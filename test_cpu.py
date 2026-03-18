from src.tts import TTS


tts = TTS(
    repo= "pnnbao-ump/VieNeu-TTS",
    device="cpu",
    local_model_path="models/VieNeu-TTS",
    local_files_only=False,
    use_logger=True
)

print("TTS model loaded successfully on CPU.")

audio_spec = tts.infer(
    "Xin chào, tôi là một mô hình chạy trên. Tôi có thể giúp bạn chuyển đổi văn bản thành giọng nói một cách hiệu quả, ngay cả khi không có. Hãy thử tôi với một đoạn văn bản và nghe kết quả nhé!"
    )

tts.save(audio_spec, "output_cpu.wav")

tts.close()