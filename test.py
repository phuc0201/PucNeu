from tts.cuda_balance import TTSWithCudaBalance

tts = TTSWithCudaBalance(local_model_path="models/VieNeu-TTS", local_files_only=False, use_logger=True)

voice_available = tts.list_preset_voices()

for desc, vid in voice_available:
    print(f"Voice ID: {vid}, Description: {desc}")
    voice_data = tts.get_preset_voice(vid)

    audio_spec = tts.infer("Xin chào, tôi là một mô hình TTS.", voice=voice_data)
    tts.save(audio_spec, f"output_{vid}.wav")

tts.close()
