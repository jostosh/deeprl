# deeprl
MSc. project on Deep Reinforcement Learning. [A3C](http://proceedings.mlr.press/v48/mniha16.pdf) was chosen as the baseline algorithm.

## This repo
Currently, the master branch only supports a minimal working example of the most relevant contributions, which are:
- Learning policy quantization, run with `python3 train.py --model a3clpq` or `python train.py --model a3cglpq`
- Local weight sharing layers, run with `python3 train.py --model a3clws`
- Spatial softmax and what-and-where architectures, run with `python3 train.py --model a3css` or 
`python train.py --model a3cww`

### Learning policy quantization
Learning policy quantization is an output layer that is largely inspired by learning vector quantization. It replaces
the softmax operator with an LVQ. The learning mechanism must be slightly adapted, since LVQ normally works on 
supervised learning problems. By using a soft class assignment as in 
[robust soft learning vector quantization](http://ieeexplore.ieee.org/document/6790243/), we can still apply default
actor-critic update for the policy weights. The sign and the magnitude of the advantage are then implicitly 
'supervising' the direction of the prototypes.

The video here demonstrates an agent with an LPQ layer:

[![LPQ on Breakout](https://img.youtube.com/vi/4k5s9KrVp98/0.jpg)](https://www.youtube.com/watch?v=4k5s9KrVp98)

It certainly shows potential on the simple Catch game:
[LPQ reults](https://raw.githubusercontent.com/jostosh/deeprl/master/doc/lpq.png)

### Local weight sharing
Formerly, I called this spatial interpolation soft weight sharing layers. For further info see 
[this repo](github.com/jostosh/sisws).

### Spatial softmax and what-and-where architectures
These architectures use spatial softmax layers. Generally performs worse than the default A3C architecture. The video 
below demonstrates a trained spatial softmax agent on Pong:

[![A3CSS on Pong](https://img.youtube.com/vi/m4RcohCW4t4/0.jpg)](https://www.youtube.com/watch?v=m4RcohCW4t4)

The what-and-where architecture works better for certain games, e.g.:

![alt text](https://raw.githubusercontent.com/jostosh/deeprl/master/doc/ww.png)
[WW results](./doc/ww.png)

## Usage
`train.py` is the default script. Check out the `Config` class in `common/config.py` to see the command line arguments 
that you can specify. Any member of `Config` can be set through the command line.