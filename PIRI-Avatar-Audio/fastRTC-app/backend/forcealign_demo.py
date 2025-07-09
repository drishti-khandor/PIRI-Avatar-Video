from forcealign import ForceAlign

# Provide path to audio file and transcript
transcript = "THE STALE SMELL OF OLD BEER LINGERS IT TAKES HEAT TO BRING OUT THE ODOUR A COLD DIP RESTORES HEALTH AND ZEST A SALT PICKLE TASTES FINE WITH HAM TAKOZAL PASTOR ARE MY FAVORITE A ZESTFUL FOOD IS THE HOT CROSS BUN"
align = ForceAlign(audio_file='./harvard.wav', transcript=transcript,)

# Run prediction and return alignment results
words = align.inference()

print("Transcript: ", align.raw_text)

print("Phoneme Alignments:", align.phoneme_alignments)

# Access predicted phoneme-level alignments

for phoneme in align.phoneme_alignments:
    print(f"Phoneme: {phoneme.phoneme}, Start: {phoneme.time_start}s, End: {phoneme.time_end + 0.1}s")
# for word in words:
#     print(f"Word: {word.word}, Start: {word.time_start}s, End: {word.time_end}s")
    
#     # print(f"Phonemes: {word.phonemes}")
    
    
#     for phoneme in word.phonemes:
#       print(f"Phoneme: {phoneme}")
#       # print(f"Phoneme: {phoneme.phoneme}, Start: {start_phoneme}s, End: {phoneme.time_end}s")