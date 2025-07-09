from forcealign import ForceAlign

# Provide path to audio file and transcript
align = ForceAlign(audio_file='./harvard.wav')

# Run prediction and return alignment results
words = align.inference()

print("Transcript: ", align.raw_text)

# Access predicted phoneme-level alignments
for word in words:
    print(f"Word: {word.word}")
    for phoneme in word.phonemes:
        print(f"Phoneme: {phoneme.phoneme}, Start: {phoneme.time_start}s, End: {phoneme.time_end}s")