## Segmentation / Contour detection
### TODO list
- [x] The paper used to generate contours for the SynthText dataset can be found [here](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/papers/amfm_pami2010.pdf). There is no code for it though (maybe [here](https://github.com/jponttuset/mcg), but it's not usable.).
  - gPb implementations: [C++, was meant for OpenCV](https://github.com/HiDiYANG/gPb-GSoC), [what looks to be the main implementation](https://github.com/vrabaud/gPb), [another matlab one](https://github.com/SCrommelinck/gPb-Contour-Detection).
- [ ] The paper introducing UCM (I think) is [here](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.123.9972&rep=rep1&type=pdf).
- [x] The [COB paper](https://arxiv.org/pdf/1701.04658v2.pdf) builds upon previous methods by adding DL to them (skimmed trough it). [This](https://arxiv.org/pdf/1701.04658v2.pdf) paper seems to be a follow up (didn't read it). Code is [here](https://github.com/kmaninis/COB).
  - Code uses Caffe (even worse, a modified version of it...). Plus it's from the same people as [mcg](https://github.com/jponttuset/mcg), so I expect it to be hot garbage.
  - [This PyTorch implementation](https://github.com/lejeunel/cobnet) might be worth looking at. Not documentation or answer from the author though.
- [ ] The [Image Segmentation Using Hierarchical Merge Tree paper](https://arxiv.org/pdf/1505.06389.pdf) seems to build uppon the previous paper, and does give the [C++ code](https://github.com/tingliu/glia). (would require making a python wrapping)
- [ ] [This paper](https://arxiv.org/pdf/1603.04530v1.pdf) seems simpler (auto-encoder), but detects objects rather than boudaries. ([code](https://github.com/captanlevi/Contour-Detection-Pytorch))
- [ ] [This github](https://github.com/MarkMoHR/Awesome-Edge-Detection-Papers) contains a list of edge detection papers.

Note: OpenCV also has a [function to detect contours](https://docs.opencv.org/3.4/df/d0d/tutorial_find_contours.html) (there is even [a tutorial](https://learnopencv.com/contour-detection-using-opencv-python-c/)). However it doesn't seem like it would be good enough.


### What I tried to far
- I tested [this gPb implementation](https://github.com/hoel-bagard/gPb-GSoC) since it gave a python wrapper, but it's slow, crahes halfway through and prone to segfaults.
- I tested the [mcg used for the SynthText dataset](https://github.com/jponttuset/mcg), but I couldn't get it to work (see the rant).
- I tested (trained) [the pytorch implementation of COB](https://github.com/hoel-bagard/cobnet?organization=hoel-bagard&organization=hoel-bagard), but the results as good as advertised.

### Matlab rant
Matlab doesn't run on Arch (see their [list of supported OS](https://uk.mathworks.com/support/requirements/matlab-system-requirements.html)). I still tried but the installation failed.\
The SynthText mcg code can't even be git cloned on Windows because of the way they named their folder. And Matlab online can't be used because the files are too big. Which forced me to use a Ubuntu VM. And their code doesn't even run (after having wasted days trying to run multiple repos from that author, I've learned to avoid anything with his name in it...)...
