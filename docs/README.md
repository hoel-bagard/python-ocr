Mostly temp notepad. Will be re-organised later (maybe in a github wiki ?)

## Fonts
The original SynthText code uses the [PyGame module](https://www.pygame.org/docs/ref/freetype.html) to generate the text.\
Since it seems like Pillow has a decent [font module](https://pillow.readthedocs.io/en/stable/reference/ImageFont.html), I will try to use it instead of PyGame. First reason being that I'm more familiar with it, second being that using a game module to generate text seems a bit overkill to me.\
OpenCV also supports custom fonts to some extent (see [here](https://docs.opencv.org/3.4/d9/dfa/classcv_1_1freetype_1_1FreeType2.html)), but it seemed more limitted compared to PyGame and Pillow.



## Learning the colors
From [the SynthText paper](https://arxiv.org/pdf/1604.06646.pdf):
> Once the location and orientation of text has been decided, text is assigned a colour. The colour palette for text is learned from cropped word images in [the IIIT5K word dataset](https://www.di.ens.fr/willow/pdfscurrent/mishra12a.pdf). Pixels in each cropped word images are partitioned into two sets using K-means, resulting in a colour pair, with one colour approximating the foreground (text) colour and the other the background. When rendering new text, the colour pair whose background colour matches the target image region the best (using L2-norm in the Lab colour space) is selected, and the corresponding foreground colour is used to render the text.

The dataset can be downloaded with:
```
wget http://cvit.iiit.ac.in/projects/SceneTextUnderstanding/IIIT5K-Word_V3.0.tar.gz
```
