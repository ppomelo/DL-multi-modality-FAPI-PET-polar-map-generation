import numpy as np

def inter_from_256(x):
    return np.interp(x=x,xp=[0,255],fp=[0,1])

cdict = {
    'red':tuple(reversed(((1.0,inter_from_256(255),inter_from_256(255)),
           (1/6*5,inter_from_256(240),inter_from_256(240)),
           (1/6*4,inter_from_256(200),inter_from_256(200)),
           (1/6*3,inter_from_256(85),inter_from_256(85)),
           (1/6*2,inter_from_256(63),inter_from_256(63)),
           (1/6*1,inter_from_256(32),inter_from_256(32)),
           (1/6 * 0,inter_from_256(8),inter_from_256(8))))),
    'green': tuple(reversed(((1.0, inter_from_256(253), inter_from_256(253)),
            (1/6 * 5, inter_from_256(180), inter_from_256(180)),
            (1/6 * 4, inter_from_256(107), inter_from_256(107)),
            (1/6 * 3, inter_from_256(57), inter_from_256(57)),
            (1/6 * 2, inter_from_256(79), inter_from_256(79)),
            (1/6 * 1, inter_from_256(102), inter_from_256(102)),
            (1/6 * 0, inter_from_256(24), inter_from_256(24))))),
    'blue': tuple(reversed(((1/6 * 6, inter_from_256(251), inter_from_256(251)),
              (1/6 * 5, inter_from_256(114), inter_from_256(114)),
              (1/6 * 4, inter_from_256(87), inter_from_256(87)),
              (1/6 * 3, inter_from_256(129), inter_from_256(129)),
              (1/6 * 2, inter_from_256(144), inter_from_256(144)),
              (1/6 * 1, inter_from_256(95), inter_from_256(95)),
              (1/6 * 0, inter_from_256(23), inter_from_256(23))))),
}