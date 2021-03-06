# Python OCR
## Data
### Getting background data
#### Inspiration
The [SynthText in the Wild Dataset](https://www.robots.ox.ac.uk/~vgg/data/scenetext/) seems like a very nice dataset, but not for commercial usage. [The corresponding github](https://github.com/ankush-me/SynthText) might contain some useful ideas though.
Some forks of this repo might also contain valuable insights, such as [this one](https://github.com/gachiemchiep/SynthText) that adds support for Japanese or the [CurvedSynthText](https://github.com/PkuDavidGuan/CurvedSynthText) one that adds curved text (and also multithread amongst other things). 

The [ICDAR challenges](https://rrc.cvc.uab.es/?ch=2) are also notable and can be downloaded (might get used later).


#### Unsplash
The Lite Unsplash dataset is [explicitely free for commercial use](https://unsplash.com/data). It can be downloaded from [this link](https://unsplash.com/data/lite/latest), the github is [here](https://github.com/unsplash/datasets) and the documentation for the data format [here](https://github.com/unsplash/datasets/blob/master/DOCS.md). 

Example on how to get it:
```
wget https://unsplash.com/data/lite/latest -P datasets
unzip datasets/latest -d datasets/unsplash
rm datasets/latest
```

You can then use the `unsplash.py` utils script to visualize some images or to download them:
```
python utils/unsplash.py datasets/unsplash -s
python utils/unsplash.py datasets/unsplash -o datasets/unsplash_images
```
(first command simply shows the images, second one saves them to disk. They can be combined.)

### Making some labelled data
I'm following the approach from the [Synthetic Data for Text Localisation in Natural Images](https://arxiv.org/pdf/1604.06646.pdf) paper (paper that created the SynthText dataset). Due to the ultimate goal of this project not being set in the usual "real world" setting, I'm going for non DL approaches.

#### Segmentation / Contour detection
See [here](https://github.com/hoel-bagard/python-ocr/blob/master/docs/countour_detection.md) for now.

## OCR
### References
#### Blogs and the likes
- [x] This [Blog post from Dropbox](https://dropbox.tech/machine-learning/creating-a-modern-ocr-pipeline-using-computer-vision-and-deep-learning). Nice read, is a bit technical but not too much.
[This](https://www.sicara.ai/blog/ocr-text-detection-recognition) blog post about 3 papers from 2018/2019. Fairly short, nice read.
- [ ] [This](https://github.com/handong1587/handong1587.github.io/blob/master/_posts/deep_learning/2015-10-09-ocr.md) very long list of papers.
- [x] Use a GAN to augment the data like [this](https://www.reddit.com/r/deeplearning/comments/ofhq7r/textboxgan_first_gan_generating_text_boxes_for/) ?
- [x] [This japanese blog post about FOTS](https://qiita.com/jjjkkkjjj/items/bfa03d89eaf6ab0c0487)

#### Papers
- [x] [FOTS: Fast Oriented Text Spotting with a Unified Network](https://arxiv.org/pdf/1801.01671.pdf). Looks nice, probably the one to use. Kinda old and no official code, but many open source implementations.
  - [FOTS.PyTorch](https://github.com/jiangxiluning/FOTS.PyTorch)
- [ ] [Attention-based Extraction of Structured Information from Street View Imagery](https://arxiv.org/pdf/1704.03549.pdf)


### Tools for later
- [Mecab](https://pypi.org/project/mecab-python3/) to cut words from a text in Japanese.


#### Tabs that I found but do not plan to read in the near future:
- [Attention-based Extraction of Structured Information from Street View Imagery](https://arxiv.org/pdf/1704.03549.pdf) (2017)
- [Python Wand](https://docs.wand-py.org/en/0.6.7/guide/draw.html#texts)   (from [this](https://stackoverflow.com/questions/68979045/how-can-i-draw-a-curved-text-using-python-converting-text-to-curved-image) stackoverflow question)
- [List of OCR datasets](https://github.com/TianzhongSong/awesome-SynthText)
- https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/mcg/
