# Voice audio recordings decompression with neural style transfer
This is modification of [Multi-target Voice Conversion without Parallel Data by Adversarially Learning Disentangled Audio Representations](https://github.com/jjery2243542/voice\_conversion) for decompression problem.


# Preprocess
Model is trained on [CSTR VCTK Corpus](https://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html).
First run change_bit_rate.sh from VCTK-Corpus to compress wavs to 8kbps mp3. Then run mp3_to_wav.sh to get wav from mp3. It's possible to omit the second step if you can generate spectrograms directly from mp3.

### Feature extraction, training, testing
Similar to [base repo](https://github.com/jjery2243542/voice\_conversion) but moved to jupyter notebooks.

### WaveNet synthesis
Download [pretrained model](https://drive.google.com/file/d/1Zksy0ndlDezo9wclQNZYkGi\_6i7zi4nQ/view) (found in AutoVC repo) and move it to implementation directory.
I use code from [r9y9's wavenet\_vocoder](https://github.com/r9y9/wavenet\_vocoder) for spectrograms generation and synthesis.

## Other notices
I dont's use GAN mode because it learns about 80 hours on my RTX 2070, so it may not work.

