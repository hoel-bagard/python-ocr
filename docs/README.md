Mostly temp notepad. Will be re-organised later (maybe in a github wiki ?)

## Fonts
The original SynthText code uses the [PyGame module](https://www.pygame.org/docs/ref/freetype.html) to generate the text.\
Since it seems like Pillow has a decent [font module](https://pillow.readthedocs.io/en/stable/reference/ImageFont.html), I will try to use it instead of PyGame. First reason being that I'm more familiar with it, second being that using a game module to generate text seems a bit overkill to me.\
OpenCV also supports custom fonts to some extent (see [here](https://docs.opencv.org/3.4/d9/dfa/classcv_1_1freetype_1_1FreeType2.html)), but it seemed more limitted compared to PyGame and Pillow.
